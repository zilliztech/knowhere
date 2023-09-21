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
]
@pytest.mark.parametrize("name,config", test_data)
def test_save_and_load(gen_data, faiss_ans, recall, error, name, config):
    # simple load and save not work for ivf nm
    print(name, config)
    version = "knowhere-v2.2.0"
    build_idx = knowhere.CreateIndex(name, version)
    xb, xq = gen_data(10000, 100, 256)

    build_idx.Build(
        knowhere.ArrayToDataSet(xb),
        json.dumps(config),
    )
    binset = knowhere.GetIndexSequence()
    build_idx.Serialize(binset)
    search_idx = knowhere.CreateIndex(name, version)
    search_idx.Deserialize(binset)
    ans, _ = search_idx.Search(
        knowhere.ArrayToDataSet(xq),
        json.dumps(config),
        knowhere.GetNullBitSetView()
    )
    k_dis, k_ids = knowhere.DataSetToArray(ans)
    f_dis, f_ids = faiss_ans(xb, xq, config["metric_type"], config["k"])
    assert recall(f_ids, k_ids) >= 0.99
    assert error(f_dis, f_dis) <= 0.01
