import knowhere
import json
import pytest

test_data = [
    (
        "FLAT",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
        },
    ),
    (
        "IVFFLAT",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
            "nlist": 1024,
            "nprobe": 1024,
        },
    ),
    (
        "IVFFLATCC",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
            "nlist": 1024,
            "nprobe": 1024,
            "ssize": 48
        },
    ),
    (
        "SCANN",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
            "m": 128,
            "nbits": 4,
            "nlist": 1024,
            "nprobe": 1024,
            "reorder_k": 1500,
            "with_raw_data": True,
        },
    ),
    # (
    #     "IVFSQ",
    #     {
    #         "dim": 256,
    #         "k": 15,
    #         "metric_type": "L2",
    #         "nlist": 1024,
    #         "nprobe": 1024,
    #     },
    # ),
    # (
    #     "IVFPQ",
    #     {
    #         "dim": 256,
    #         "k": 15,
    #         "metric_type": "L2",
    #         "nlist": 1024,
    #         "nprobe": 1024,
    #         "m": 32,
    #         "nbits": 8,
    #     },
    # ),
    (
        "HNSW",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
            "M": 64,
            "efConstruction": 256,
            "ef": 256,
        },
    ),
]


@pytest.mark.parametrize("name,config", test_data)
def test_index(gen_data, faiss_ans, recall, error, name, config):
    print(name, config)
    idx = knowhere.CreateIndex(name)
    xb, xq = gen_data(10000, 100, 256)

    idx.Build(
        knowhere.ArrayToDataSet(xb),
        json.dumps(config),
    )
    ans, _ = idx.Search(
        knowhere.ArrayToDataSet(xq),
        json.dumps(config),
        knowhere.GetNullBitSetView()
    )
    k_dis, k_ids = knowhere.DataSetToArray(ans)
    f_dis, f_ids = faiss_ans(xb, xq, config["metric_type"], config["k"])
    assert recall(f_ids, k_ids) >= 0.9
    assert error(f_dis, f_dis) <= 0.01
