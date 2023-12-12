import pytest
import numpy as np
import faiss
from bfloat16 import bfloat16


@pytest.fixture()
def gen_data():
    def wrap(xb_rows, xq_rows, dim):
        return (
            np.random.randn(xb_rows, dim).astype(np.float32),
            np.random.randn(xq_rows, dim).astype(np.float32),
        )

    return wrap

@pytest.fixture()
def gen_data_with_type():
    def wrap(xb_rows, xq_rows, dim, type):
        if type == np.float16 or bfloat16:
            xb = np.random.randn(xb_rows, dim).astype(type)
            xq = np.random.randn(xq_rows, dim).astype(type)
            # To fix nan or inf when type is equal to float16
            min_value = -10.0
            max_value = 10.0
            xb = np.clip(xb, min_value, max_value).astype(type)
            xq = np.clip(xq, min_value, max_value).astype(type)
            return xb,xq
        else:
            return (
                np.random.randn(xb_rows, dim).astype(type),
                np.random.randn(xq_rows, dim).astype(type),
            )

    return wrap


@pytest.fixture()
def faiss_ans():
    def wrap(xb, xq, metric_type, k):
        index = None
        if metric_type == "L2":
            index = faiss.IndexFlat(xb.shape[1], faiss.METRIC_L2)
        if metric_type == "L1":
            index = faiss.IndexFlat(xb.shape[1], faiss.METRIC_L1)
        index.add(xb)
        return index.search(xq, k)

    return wrap


@pytest.fixture()
def recall():
    def wrap(a, b):
        return (a == b).mean()

    return wrap


@pytest.fixture()
def error():
    def wrap(a, b):
        return np.fabs(a - b).sum()

    return wrap
