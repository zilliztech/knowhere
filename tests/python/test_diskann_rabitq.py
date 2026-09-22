# Copyright (C) 2026 Zilliz. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import struct

import knowhere
import numpy as np
import pytest


@pytest.mark.parametrize("metric", ["L2", "IP", "COSINE"])
@pytest.mark.parametrize("kind,codec", [("DISKANN_RABITQ", None), ("DISKANN", "RABITQ"), ("DISKANN", None)])
def test_navigation_roundtrip(tmp_path, metric, kind, codec):
    rng = np.random.default_rng(42)
    base = rng.normal(size=(1000, 64)).astype("float32")
    query = rng.normal(size=(10, 64)).astype("float32")
    if metric == "COSINE":
        base *= rng.uniform(0.1, 10, size=(1000, 1)).astype("float32")
        query *= rng.uniform(0.1, 10, size=(10, 1)).astype("float32")
    source = tmp_path / "base.fbin"
    with source.open("wb") as output:
        output.write(struct.pack("<II", *base.shape))
        output.write(base.tobytes())
    config = dict(dim=64, metric_type=metric, index_prefix=str(tmp_path / "index"), data_path=str(source),
                  max_degree=32, search_list_size=100, pq_code_budget_gb=0.00002, build_dram_budget_gb=1,
                  disk_pq_dims=0, search_cache_budget_gb=0, search_cache_budget_gb_ratio=0, rbq_bits=4)
    if codec is not None:
        config["navigation_codec"] = codec
    if kind == "DISKANN_RABITQ" or codec == "RABITQ":
        config.pop("pq_code_budget_gb")
    version = knowhere.GetCurrentVersion()
    index = knowhere.CreateIndex(kind, version)
    assert knowhere.Status(index.Build(knowhere.GetNullDataSet(), json.dumps(config))) == knowhere.Status.success
    binary = knowhere.GetBinarySet()
    assert knowhere.Status(index.Serialize(binary)) == knowhere.Status.success
    del index
    # A fresh generic DiskANN node recovers its navigation from stored files,
    # without the original codec, bits, PQ budget or other build parameters.
    restored = knowhere.CreateIndex("DISKANN", version)
    load = dict(metric_type=metric, index_prefix=config["index_prefix"],
                search_cache_budget_gb=0, search_cache_budget_gb_ratio=0, warm_up=True)
    assert knowhere.Status(restored.Deserialize(binary, json.dumps(load))) == knowhere.Status.success
    search = dict(dim=64, metric_type=metric, k=10, search_list_size=100, beamwidth=4, rbq_bits_query=4)
    result, status = restored.Search(knowhere.ArrayToDataSet(query), json.dumps(search), knowhere.GetNullBitSetView())
    assert knowhere.Status(status) == knowhere.Status.success
    distances, ids = knowhere.DataSetToArray(result)
    assert ids.shape == (10, 10)
    assert np.all(ids >= 0) and np.all(np.isfinite(distances))
    exact = ((query[:, None, :] - base[None, :, :]) ** 2).sum(2) if metric == "L2" else -(query @ base.T)
    if metric == "COSINE":
        exact /= np.linalg.norm(query, axis=1)[:, None] * np.linalg.norm(base, axis=1)[None, :]
    truth = np.argsort(exact, axis=1)[:, :10]
    recall = sum(len(set(actual) & set(expected)) for actual, expected in zip(ids, truth)) / 100
    assert recall > 0.8
    if metric == "COSINE":
        expected_scores = -np.take_along_axis(exact, ids, axis=1)
        np.testing.assert_allclose(distances, expected_scores, rtol=1e-4, atol=1e-5)
    if metric in ("IP", "COSINE"):
        zeros = np.zeros((1, 64), dtype="float32")
        empty, status = restored.Search(knowhere.ArrayToDataSet(zeros), json.dumps(search), knowhere.GetNullBitSetView())
        assert knowhere.Status(status) == knowhere.Status.success
        empty_distances, empty_ids = knowhere.DataSetToArray(empty)
        assert np.all(empty_ids == -1) and np.all(empty_distances == -1)
