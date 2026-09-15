# Copyright (C) 2026 Zilliz. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import struct

import knowhere
import numpy as np
import pytest


@pytest.mark.parametrize("metric", ["L2", "IP"])
@pytest.mark.parametrize("kind,codec", [("DISKANN_RABITQ", None), ("DISKANN", "RABITQ"), ("DISKANN", None)])
def test_navigation_roundtrip(tmp_path, metric, kind, codec):
    rng = np.random.default_rng(42)
    base = rng.normal(size=(1000, 64)).astype("float32")
    query = rng.normal(size=(10, 64)).astype("float32")
    source = tmp_path / "base.fbin"
    with source.open("wb") as output:
        output.write(struct.pack("<II", *base.shape))
        output.write(base.tobytes())
    config = dict(dim=64, metric_type=metric, index_prefix=str(tmp_path / "index"), data_path=str(source),
                  max_degree=32, search_list_size=100, pq_code_budget_gb=0.00002, build_dram_budget_gb=1,
                  disk_pq_dims=0, search_cache_budget_gb=0, search_cache_budget_gb_ratio=0, rbq_bits=4)
    if codec is not None:
        config["navigation_codec"] = codec
    version = knowhere.GetCurrentVersion()
    index = knowhere.CreateIndex(kind, version)
    assert knowhere.Status(index.Build(knowhere.GetNullDataSet(), json.dumps(config))) == knowhere.Status.success
    binary = knowhere.GetBinarySet()
    assert knowhere.Status(index.Serialize(binary)) == knowhere.Status.success
    del index
    restored = knowhere.CreateIndex(kind, version)
    assert knowhere.Status(restored.Deserialize(binary, json.dumps(dict(config, warm_up=True)))) == knowhere.Status.success
    search = dict(dim=64, metric_type=metric, k=10, search_list_size=100, beamwidth=4, rbq_bits_query=4)
    result, status = restored.Search(knowhere.ArrayToDataSet(query), json.dumps(search), knowhere.GetNullBitSetView())
    assert knowhere.Status(status) == knowhere.Status.success
    distances, ids = knowhere.DataSetToArray(result)
    assert ids.shape == (10, 10)
    assert np.all(ids >= 0) and np.all(np.isfinite(distances))
    exact = ((query[:, None, :] - base[None, :, :]) ** 2).sum(2) if metric == "L2" else -(query @ base.T)
    truth = np.argsort(exact, axis=1)[:, :10]
    recall = sum(len(set(actual) & set(expected)) for actual, expected in zip(ids, truth)) / 100
    assert recall > 0.8
