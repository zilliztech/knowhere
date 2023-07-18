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
            "ssize" : 48
        },
    ),
    (
        "IVFSQ",
        {
            "dim": 256,
            "k": 15,
            "metric_type": "L2",
            "nlist": 1024,
            "nprobe": 1024,
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
    if (name != "IVFSQ"):
        assert recall(f_ids, k_ids) >= 0.99
    else:
        assert recall(f_ids, k_ids) >= 0.70
    assert error(f_dis, f_dis) <= 0.01

    bitset = knowhere.CreateBitSet(xb.shape[0])
    for id in k_ids[:10,:1].ravel():
        bitset.SetBit(int(id))
    ans, _ = idx.Search(
        knowhere.ArrayToDataSet(xq),
        json.dumps(config),
        bitset.GetBitSetView()
    )

    k_dis, k_ids = knowhere.DataSetToArray(ans)
    if (name != "IVFSQ"):
        assert recall(f_ids, k_ids) >= 0.7
    else:
        assert recall(f_ids, k_ids) >= 0.5
    assert error(f_dis, f_dis) <= 0.01
