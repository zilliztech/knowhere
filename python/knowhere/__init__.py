from . import swigknowhere
from .swigknowhere import Status
from .swigknowhere import GetNullDataSet, GetNullBitSetView, GetIndexSequence
import numpy as np

def CreateIndex(name, version):
    return swigknowhere.IndexWrap(name, version)


def CreateBitSet(bits_num):
    return swigknowhere.BitSet(bits_num)

def ArrayToDataSet(arr):
    if arr.ndim == 1:
        return swigknowhere.Array2DataSetIds(arr)
    if arr.ndim == 2:
        if arr.dtype == np.int32:
            return swigknowhere.Array2DataSetI(arr)
        if arr.dtype == np.float32:
            return swigknowhere.Array2DataSetF(arr)
    raise ValueError(
        """
        ArrayToDataSet only support numpy array dtype float32 and int32.
        """
    )

def DataSetToArray(ans):
    dim = swigknowhere.DataSet_Dim(ans)
    rows = swigknowhere.DataSet_Rows(ans)
    dis = np.zeros([rows, dim]).astype(np.float32)
    ids = np.zeros([rows, dim]).astype(np.int32)
    swigknowhere.DataSet2Array(ans, dis, ids)
    return dis, ids


def RangeSearchDataSetToArray(ans):
    rows = swigknowhere.DataSet_Rows(ans)
    lims = np.zeros(
        [
            rows + 1,
        ],
        dtype=np.int32,
    )
    swigknowhere.DumpRangeResultLimits(ans, lims)
    dis = np.zeros(
        [
            lims[-1],
        ],
        dtype=np.float32,
    )
    swigknowhere.DumpRangeResultDis(ans, dis)
    ids = np.zeros(
        [
            lims[-1],
        ],
        dtype=np.int32,
    )
    swigknowhere.DumpRangeResultIds(ans, ids)

    dis_list = []
    ids_list = []
    for idx in range(rows):
        dis_list.append(dis[lims[idx] : lims[idx + 1]])
        ids_list.append(ids[lims[idx] : lims[idx + 1]])

    return dis_list, ids_list

def GetVectorDataSetToArray(ans):
    dim = swigknowhere.DataSet_Dim(ans)
    rows = swigknowhere.DataSet_Rows(ans)
    data = np.zeros([rows, dim]).astype(np.float32)
    swigknowhere.DataSetTensor2Array(ans, data)
    return data
