import knowhere
import json
import pytest
import shutil
import ctypes
import os
import numpy as np
import time

def fbin_write(x, fname):
    assert x.dtype == np.float32
    f = open(fname, "wb")
    n, d = x.shape
    np.array([n, d], dtype='uint32').tofile(f)
    x.tofile(f)

def test_index(gen_data, faiss_ans, recall, error):
    version = "knowhere-v2.2.0"
    index_name = "DISKANN"
    diskann_dir = "diskann_test"
    data_path = os.path.join(diskann_dir, "diskann_data")
    index_path = os.path.join(diskann_dir, "diskann_index")
    ndim = 128
    nb = 10000
    nq = 100

    # create file path and data
    try:
        shutil.rmtree(diskann_dir)
    except:
        pass
    os.mkdir(diskann_dir)
    os.mkdir(index_path)
    xb, xq = gen_data(nb, nq, ndim)
    fbin_write(xb, data_path)

    # create config
    pq_code_size = ctypes.sizeof(ctypes.c_float) * ndim * nb * 0.125 / (1024 * 1024 * 1024)
    diskann_config = {
        "build_config": {
            "dim": 128,
            "metric_type": "L2",
            "index_prefix": index_path,
            "data_path": data_path,
            "max_degree": 56,
            "search_list_size": 128,
            "pq_code_budget_gb": pq_code_size,
            "build_dram_budget_gb":32.0,
            "num_threads": 8
        },
        "search_config": {
            "dim":128,
            "metric_type":"L2",
            "k":10,
            "search_list_size": 100,
            "beamwidth":8
        },
        "deserialize_config": {
            "metric_type":"L2",
            "index_prefix": index_path,
            "search_cache_budget_gb": pq_code_size,
        }
    }

    print(index_name, diskann_config["build_config"])
    diskann = knowhere.CreateIndex(index_name, version)
    build_status = diskann.Build(
        knowhere.GetNullDataSet(),
        json.dumps(diskann_config["build_config"]),
    )
    assert knowhere.Status(build_status) == knowhere.Status.success
    diskann.Deserialize(knowhere.GetBinarySet(), json.dumps(diskann_config["deserialize_config"]))
    ans, _ = diskann.Search(
        knowhere.ArrayToDataSet(xq),
        json.dumps(diskann_config["search_config"]),
        knowhere.GetNullBitSetView()
    )
    k_dis, k_ids = knowhere.DataSetToArray(ans)
    f_dis, f_ids = faiss_ans(xb, xq, diskann_config["search_config"]["metric_type"], diskann_config["search_config"]["k"])
    assert recall(f_ids, k_ids) >= 0.60
    assert error(f_dis, f_dis) <= 0.01
    shutil.rmtree(diskann_dir)
