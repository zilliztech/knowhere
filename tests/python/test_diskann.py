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

def remove_ncs_uploaded_files(index, index_path):
    """
    Remove files from disk that have been uploaded to NCS.
    This mimics VectorDiskAnnIndex::filterIndexFiles() logic - when NCS is enabled,
    files matching these patterns can be removed from disk since they're now in NCS.
    
    Args:
        index: The DiskANN index object
        index_path: Path to the index directory
        
    Returns:
        Number of files removed
    """
    # Get list of file patterns that were uploaded to NCS
    filter_out_patterns = index.ListFilesForNcsUpload()
    print(f"File patterns uploaded to NCS (will be removed from disk): {filter_out_patterns}")
    
    # Remove unnecessary files from disk that are now in NCS
    files_removed = 0
    for root, dirs, files in os.walk(index_path):
        for file in files:
            # Check if this file matches any of the NCS upload patterns (substring match)
            should_remove = any(pattern in file for pattern in filter_out_patterns)
            if should_remove:
                file_path = os.path.join(root, file)
                print(f"Removing unnecessary file from disk (now in NCS): {file}")
                os.remove(file_path)
                files_removed += 1
    
    return files_removed

def test_index(gen_data, faiss_ans, recall, error):
    version = knowhere.GetCurrentVersion()
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

def test_index_with_ncs(gen_data, faiss_ans, recall, error):
    version = knowhere.GetCurrentVersion()
    index_name = "DISKANN"
    diskann_dir = "diskann_test_ncs"
    data_path = os.path.join(diskann_dir, "diskann_data")
    index_path = os.path.join(diskann_dir, "diskann_index")
    ndim = 128
    nb = 10000
    nq = 100

    # Initialize NCS (in-memory) before building index
    print("Initializing NCS...")
    knowhere.InitNcs("in_memory")
    
    # create file path and data
    try:
        shutil.rmtree(diskann_dir)
    except:
        pass
    os.mkdir(diskann_dir)
    os.mkdir(index_path)
    xb, xq = gen_data(nb, nq, ndim)
    fbin_write(xb, data_path)

    # create config with NCS (in-memory neighbor cache)
    pq_code_size = ctypes.sizeof(ctypes.c_float) * ndim * nb * 0.125 / (1024 * 1024 * 1024)
    bucket_id = int(time.time() * 1000) % (2**32)  # Use numeric bucket ID
    
    # Create NCS bucket
    print(f"Creating NCS bucket {bucket_id}...")
    result = knowhere.CreateNcsBucket(bucket_id)
    assert result == 0, f"Failed to create NCS bucket, result: {result}"
    print(f"NCS bucket {bucket_id} created successfully")
    diskann_config = {
        "build_config": {
            "dim": 128,
            "metric_type": "L2",
            "index_prefix": index_path,
            "data_path": data_path,
            "max_degree": 56,
            "search_list_size": 128,
            "pq_code_budget_gb": pq_code_size,
            "build_dram_budget_gb": 32.0,
            "num_threads": 8,
            "ncs_descriptor": {
                "ncsKind_": "in_memory",
                "bucketId_": bucket_id,
                "extras_": {}
            }
        },
        "search_config": {
            "dim": 128,
            "metric_type": "L2",
            "k": 10,
            "search_list_size": 100,
            "beamwidth": 8
        },
        "deserialize_config": {
            "metric_type": "L2",
            "index_prefix": index_path,
            "search_cache_budget_gb": pq_code_size,
            "ncs_enable": True,
            "ncs_descriptor": {
                "ncsKind_": "in_memory",
                "bucketId_": bucket_id,
                "extras_": {}
            }
        }
    }

    print(index_name + " with NCS", diskann_config["build_config"])
    diskann = knowhere.CreateIndex(index_name, version)
    build_status = diskann.Build(
        knowhere.GetNullDataSet(),
        json.dumps(diskann_config["build_config"]),
    )
    assert knowhere.Status(build_status) == knowhere.Status.success
    
    # Verify bucket exists before upload
    bucket_exists = knowhere.IsNcsBucketExist(bucket_id)
    assert bucket_exists, f"NCS bucket {bucket_id} does not exist"
    
    # Upload index to NCS after build
    print(f"Uploading index to NCS bucket {bucket_id}...")
    upload_status = diskann.NcsUpload(json.dumps(diskann_config["build_config"]))
    assert knowhere.Status(upload_status) == knowhere.Status.success
    
    # Remove unnecessary files from disk that are now in NCS (before deserialize)
    # This saves disk space and matches the behavior of filterIndexFiles()
    # Note: Files are created in the parent directory with index_prefix as filename prefix
    files_removed = remove_ncs_uploaded_files(diskann, diskann_dir)
    diskann.Deserialize(knowhere.GetBinarySet(), json.dumps(diskann_config["deserialize_config"]))
    ans, _ = diskann.Search(
        knowhere.ArrayToDataSet(xq),
        json.dumps(diskann_config["search_config"]),
        knowhere.GetNullBitSetView()
    )
    k_dis, k_ids = knowhere.DataSetToArray(ans)
    f_dis, f_ids = faiss_ans(xb, xq, diskann_config["search_config"]["metric_type"], diskann_config["search_config"]["k"])
    # NCS should have better or similar recall compared to non-NCS
    assert recall(f_ids, k_ids) >= 0.60
    assert error(f_dis, f_dis) <= 0.01
    
    # Cleanup
    knowhere.DeleteNcsBucket(bucket_id)
    shutil.rmtree(diskann_dir)
