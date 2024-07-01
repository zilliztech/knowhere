from . import swigknowhere
from .swigknowhere import Status
from .swigknowhere import CreateBinarySet, GetBinarySet, GetNullDataSet, GetNullBitSetView
from .swigknowhere import BruteForceSearchFloat, BruteForceRangeSearchFloat
from .swigknowhere import BruteForceSearchFP16, BruteForceRangeSearchFP16
from .swigknowhere import BruteForceSearchBF16, BruteForceRangeSearchBF16
from .swigknowhere import BruteForceSearchBin, BruteForceRangeSearchBin

import numpy as np
from bfloat16 import bfloat16


def CreateIndex(name, version, type=np.float32):
    if type == np.float32:
        return swigknowhere.IndexWrapFloat(name, version)
    if type == np.float16:
        return swigknowhere.IndexWrapFP16(name, version)
    if type == bfloat16:
        return swigknowhere.IndexWrapBF16(name, version)
    if type == np.uint8:
        return swigknowhere.IndexWrapBin(name, version)

def BruteForceSearch(type=np.float32, *args):
    if type == np.float32:
        return BruteForceSearchFloat(*args)
    if type == np.float16:
        return BruteForceSearchFP16(*args)
    if type == bfloat16:
        return BruteForceSearchBF16(*args)
    if type == np.uint8:
        return BruteForceSearchBin(*args)

def BruteForceRangeSearch(type=np.float32, *args):
    if type == np.float32:
        return BruteForceRangeSearchFloat(*args)
    if type == np.float16:
        return BruteForceRangeSearchFP16(*args)
    if type == bfloat16:
        return BruteForceRangeSearchBF16(*args)
    if type == np.uint8:
        return BruteForceRangeSearchBin(*args)


def GetCurrentVersion():
    return swigknowhere.CurrentVersion()


def CreateBitSet(bits_num):
    return swigknowhere.BitSet(bits_num)


def Load(binset, file_name):
    return swigknowhere.Load(binset, file_name)


def Dump(binset, file_name):
    return swigknowhere.Dump(binset, file_name)


def WriteIndexToDisk(binset, index_type, data_path):
    return swigknowhere.WriteIndexToDisk(binset, index_type, data_path)

def ArrayToBinary(arr):
    if arr.dtype == np.uint8:
        return swigknowhere.Array2Binary(arr)
    raise ValueError(
        """
        ArrayToBinary only support numpy array dtype uint8.
        """
    )

def ArrayToDataSet(arr):
    if arr.ndim == 1:
        return swigknowhere.Array2DataSetIds(arr)
    if arr.ndim == 2:
        if arr.dtype == np.uint8:
            return swigknowhere.Array2DataSetI(arr)
        if arr.dtype == np.float32:
            return swigknowhere.Array2DataSetF(arr)
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
            return swigknowhere.Array2DataSetFP16(arr)
        if arr.dtype == bfloat16:
            arr = arr.astype(np.float32)
            return swigknowhere.Array2DataSetBF16(arr)
    raise ValueError(
        """
        ArrayToDataSet only support numpy array dtype float32,uint8,float16 and bfloat16.
        """
    )

# follow csr_matrix format
# row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their
# corresponding values are stored in ``data[indptr[i]:indptr[i+1]]
def ArrayToSparseDataSet(data, indices, indptr):
    if data.ndim == 1 and indices.ndim == 1 and indptr.ndim == 1:
        assert data.shape[0] == indices.shape[0]
        assert indptr.shape[0] > 1
        return swigknowhere.Array2SparseDataSet(data, indices, indptr)
    raise ValueError(
        """
        ArrayToSparseDataSet input type wrong.
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

def GetFloat16VectorDataSetToArray(ans):
    dim = swigknowhere.DataSet_Dim(ans)
    rows = swigknowhere.DataSet_Rows(ans)
    data = np.zeros([rows, dim]).astype(np.float32)
    swigknowhere.Float16DataSetTensor2Array(ans, data)
    data = data.astype(np.float16)
    return data

def GetBFloat16VectorDataSetToArray(ans):
    dim = swigknowhere.DataSet_Dim(ans)
    rows = swigknowhere.DataSet_Rows(ans)
    data = np.zeros([rows, dim]).astype(np.float32)
    swigknowhere.BFloat16DataSetTensor2Array(ans, data)
    data = data.astype(bfloat16)
    return data

def GetBinaryVectorDataSetToArray(ans):
    dim = int(swigknowhere.DataSet_Dim(ans) / 8)
    rows = swigknowhere.DataSet_Rows(ans)
    data = np.zeros([rows, dim]).astype(np.uint8)
    swigknowhere.BinaryDataSetTensor2Array(ans, data)
    return data

def SetSimdType(type):
    swigknowhere.SetSimdType(type)

def SetBuildThreadPool(num_threads):
    swigknowhere.SetBuildThreadPool(num_threads)

def SetSearchThreadPool(num_threads):
    swigknowhere.SetSearchThreadPool(num_threads)
