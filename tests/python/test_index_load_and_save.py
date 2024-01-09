import knowhere
import json
import pytest
import os
import numpy as np
from bfloat16 import bfloat16

test_data = [
    (
        "FLAT",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
        },
    ),
]
index_file = "test_index_load_and_save.index"


@pytest.mark.parametrize("name,config", test_data)
def test_save_and_load(gen_data, faiss_ans, recall, error, name, config):
    # simple load and save not work for ivf nm
    print(name, config)
    version = knowhere.GetCurrentVersion()
    build_idx = knowhere.CreateIndex(name, version, np.float32)
    xb, xq = gen_data(10_000, 100, 256)

    # build, serialize and dump
    build_idx.Build(
        knowhere.ArrayToDataSet(xb),
        json.dumps(config),
    )
    binset = knowhere.GetBinarySet()
    build_idx.Serialize(binset)
    knowhere.Dump(binset, index_file)

    # load and deserialize
    new_binset = knowhere.GetBinarySet()
    knowhere.Load(new_binset, index_file)
    search_idx = knowhere.CreateIndex(name, version, np.float32)
    search_idx.Deserialize(new_binset)

    # test the loaded index
    ans, _ = search_idx.Search(
        knowhere.ArrayToDataSet(xq), json.dumps(config), knowhere.GetNullBitSetView()
    )
    k_dis, k_ids = knowhere.DataSetToArray(ans)
    f_dis, f_ids = faiss_ans(xb, xq, config["metric_type"], config["k"])
    assert recall(f_ids, k_ids) >= 0.99
    assert error(f_dis, f_dis) <= 0.01

    # delete the index_file
    os.remove(index_file)

@pytest.mark.parametrize("name,config", test_data)
def test_float16_save_and_load(gen_data_with_type, faiss_ans, recall, error, name, config):
    # simple load and save not work for ivf nm
    print(name, config)
    version = knowhere.GetCurrentVersion()
    build_idx = knowhere.CreateIndex(name, version, np.float16)
    xb, xq = gen_data_with_type(10_000, 100, 256, np.float16)

    # build, serialize and dump
    build_idx.Build(
        knowhere.ArrayToDataSet(xb),
        json.dumps(config),
    )
    binset = knowhere.GetBinarySet()
    build_idx.Serialize(binset)
    knowhere.Dump(binset, index_file)

    # load and deserialize
    new_binset = knowhere.GetBinarySet()
    knowhere.Load(new_binset, index_file)
    search_idx = knowhere.CreateIndex(name, version, np.float16)
    search_idx.Deserialize(new_binset)

    # test the loaded index
    ans, _ = search_idx.Search(
        knowhere.ArrayToDataSet(xq), json.dumps(config), knowhere.GetNullBitSetView()
    )
    k_dis, k_ids = knowhere.DataSetToArray(ans)
    f_dis, f_ids = faiss_ans(xb, xq, config["metric_type"], config["k"])
    assert recall(f_ids, k_ids) >= 0.99
    assert error(f_dis, f_dis) <= 0.01

    # delete the index_file
    os.remove(index_file)

@pytest.mark.parametrize("name,config", test_data)
def test_bfloat16_save_and_load(gen_data_with_type, faiss_ans, recall, error, name, config):
    # simple load and save not work for ivf nm
    print(name, config)
    version = knowhere.GetCurrentVersion()
    build_idx = knowhere.CreateIndex(name, version, bfloat16)
    xb, xq = gen_data_with_type(10_000, 100, 256, bfloat16)

    # build, serialize and dump
    build_idx.Build(
        knowhere.ArrayToDataSet(xb),
        json.dumps(config),
    )
    binset = knowhere.GetBinarySet()
    build_idx.Serialize(binset)
    knowhere.Dump(binset, index_file)

    # load and deserialize
    new_binset = knowhere.GetBinarySet()
    knowhere.Load(new_binset, index_file)
    search_idx = knowhere.CreateIndex(name, version, bfloat16)
    search_idx.Deserialize(new_binset)

    # test the loaded index
    ans, _ = search_idx.Search(
        knowhere.ArrayToDataSet(xq), json.dumps(config), knowhere.GetNullBitSetView()
    )
    k_dis, k_ids = knowhere.DataSetToArray(ans)
    f_dis, f_ids = faiss_ans(xb, xq, config["metric_type"], config["k"])
    assert recall(f_ids, k_ids) >= 0.99
    assert error(f_dis, f_dis) <= 0.01

    # delete the index_file
    os.remove(index_file)
