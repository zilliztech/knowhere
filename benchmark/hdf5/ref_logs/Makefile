###################################################################################################
# Test Knowhere binary index
test_binary: test_binary_idmap test_binary_ivf_flat test_binary_hnsw

test_binary_idmap:
	./benchmark_binary --gtest_filter="Benchmark_binary.TEST_BINARY_IDMAP" | tee test_binary_idmap.log
test_binary_ivf_flat:
	./benchmark_binary --gtest_filter="Benchmark_binary.TEST_BINARY_IVF_FLAT" | tee test_binary_ivf_flat.log
test_binary_hnsw:
	./benchmark_binary --gtest_filter="Benchmark_binary.TEST_BINARY_HNSW" | tee test_binary_hnsw.log

###################################################################################################
# Test Knowhere binary index range
test_binary_range: test_binary_range_idmap test_binary_range_ivf_flat test_binary_range_hnsw

test_binary_range_idmap:
	./benchmark_binary_range --gtest_filter="Benchmark_binary_range.TEST_BINARY_IDMAP" | tee test_binary_range_idmap.log
test_binary_range_ivf_flat:
	./benchmark_binary_range --gtest_filter="Benchmark_binary_range.TEST_BINARY_IVF_FLAT" | tee test_binary_range_ivf_flat.log
test_binary_range_hnsw:
	./benchmark_binary_range --gtest_filter="Benchmark_binary_range.TEST_BINARY_HNSW" | tee test_binary_range_hnsw.log

###################################################################################################
# Test Knowhere float index
test_float: test_float_idmap test_float_ivf_flat test_float_ivf_sq8 test_float_ivf_pq test_float_hnsw test_float_diskann
test_float_gpu: test_float_ivf_flat test_float_ivf_pq

test_float_idmap:
	./benchmark_float --gtest_filter="Benchmark_float.TEST_IDMAP" | tee test_float_idmap.log
test_float_ivf_flat:
	./benchmark_float --gtest_filter="Benchmark_float.TEST_IVF_FLAT" | tee test_float_ivf_flat.log
test_float_ivf_sq8:
	./benchmark_float --gtest_filter="Benchmark_float.TEST_IVF_SQ8" | tee test_float_ivf_sq8.log
test_float_ivf_pq:
	./benchmark_float --gtest_filter="Benchmark_float.TEST_IVF_PQ" | tee test_float_ivf_pq.log
test_float_hnsw:
	./benchmark_float --gtest_filter="Benchmark_float.TEST_HNSW" | tee test_float_hnsw.log
test_float_diskann:
	./benchmark_float --gtest_filter="Benchmark_float.TEST_DISKANN" | tee test_float_diskann.log

###################################################################################################
# Test Knowhere float index bitset
test_float_bitset: test_float_bitset_ivf_flat test_float_bitset_ivf_sq8 test_float_bitset_ivf_pq test_float_bitset_hnsw test_float_bitset_diskann
test_float_bitset_gpu: test_float_bitset_ivf_flat test_float_bitset_ivf_pq

test_float_bitset_ivf_flat:
	./benchmark_float_bitset --gtest_filter="Benchmark_float_bitset.TEST_IVF_FLAT" | tee test_float_bitset_ivf_flat.log
test_float_bitset_ivf_sq8:
	./benchmark_float_bitset --gtest_filter="Benchmark_float_bitset.TEST_IVF_SQ8" | tee test_float_bitset_ivf_sq8.log
test_float_bitset_ivf_pq:
	./benchmark_float_bitset --gtest_filter="Benchmark_float_bitset.TEST_IVF_PQ" | tee test_float_bitset_ivf_pq.log
test_float_bitset_hnsw:
	./benchmark_float_bitset --gtest_filter="Benchmark_float_bitset.TEST_HNSW" | tee test_float_bitset_hnsw.log
test_float_bitset_diskann:
	./benchmark_float_bitset --gtest_filter="Benchmark_float_bitset.TEST_DISKANN" | tee test_float_bitset_diskann.log

###################################################################################################
# Test Knowhere float index range
test_float_range: test_float_range_idmap test_float_range_ivf_flat test_float_range_ivf_sq8 test_float_range_ivf_pq test_float_range_hnsw test_float_range_diskann

test_float_range_idmap:
	./benchmark_float_range --gtest_filter="Benchmark_float_range.TEST_IDMAP" | tee test_float_range_idmap.log
test_float_range_ivf_flat:
	./benchmark_float_range --gtest_filter="Benchmark_float_range.TEST_IVF_FLAT" | tee test_float_range_ivf_flat.log
test_float_range_ivf_sq8:
	./benchmark_float_range --gtest_filter="Benchmark_float_range.TEST_IVF_SQ8" | tee test_float_range_ivf_sq8.log
test_float_range_ivf_pq:
	./benchmark_float_range --gtest_filter="Benchmark_float_range.TEST_IVF_PQ" | tee test_float_range_ivf_pq.log
test_float_range_hnsw:
	./benchmark_float_range --gtest_filter="Benchmark_float_range.TEST_HNSW" | tee test_float_range_hnsw.log
test_float_range_diskann:
	./benchmark_float_range --gtest_filter="Benchmark_float_range.TEST_DISKANN" | tee test_float_range_diskann.log

###################################################################################################
# Test Knowhere float index range bitset
test_float_range_bitset: test_float_range_bitset_ivf_flat test_float_range_bitset_ivf_sq8 test_float_range_bitset_ivf_pq test_float_range_bitset_hnsw test_float_range_bitset_diskann

test_float_range_bitset_ivf_flat:
	./benchmark_float_range_bitset --gtest_filter="Benchmark_float_range_bitset.TEST_IVF_FLAT" | tee test_float_range_bitset_ivf_flat.log
test_float_range_bitset_ivf_sq8:
	./benchmark_float_range_bitset --gtest_filter="Benchmark_float_range_bitset.TEST_IVF_SQ8" | tee test_float_range_bitset_ivf_sq8.log
test_float_range_bitset_ivf_pq:
	./benchmark_float_range_bitset --gtest_filter="Benchmark_float_range_bitset.TEST_IVF_PQ" | tee test_float_range_bitset_ivf_pq.log
test_float_range_bitset_hnsw:
	./benchmark_float_range_bitset --gtest_filter="Benchmark_float_range_bitset.TEST_HNSW" | tee test_float_range_bitset_hnsw.log
test_float_range_bitset_diskann:
	./benchmark_float_range_bitset --gtest_filter="Benchmark_float_range_bitset.TEST_DISKANN" | tee test_float_range_bitset_diskann.log

###################################################################################################
# Test Knowhere float index range multi
test_float_range_multi: test_float_range_multi_idmap test_float_range_multi_ivf_flat test_float_range_multi_ivf_sq8 test_float_range_multi_ivf_pq test_float_range_multi_hnsw

test_float_range_multi_idmap:
	./benchmark_float_range_multi --gtest_filter="Benchmark_float_range_multi.TEST_IDMAP" | tee test_float_range_multi_idmap.log
test_float_range_multi_ivf_flat:
	./benchmark_float_range_multi --gtest_filter="Benchmark_float_range_multi.TEST_IVF_FLAT" | tee test_float_range_multi_ivf_flat.log
test_float_range_multi_ivf_sq8:
	./benchmark_float_range_multi --gtest_filter="Benchmark_float_range_multi.TEST_IVF_SQ8" | tee test_float_range_multi_ivf_sq8.log
test_float_range_multi_ivf_pq:
	./benchmark_float_range_multi --gtest_filter="Benchmark_float_range_multi.TEST_IVF_PQ" | tee test_float_range_multi_ivf_pq.log
test_float_range_multi_hnsw:
	./benchmark_float_range_multi --gtest_filter="Benchmark_float_range_multi.TEST_HNSW" | tee test_float_range_multi_hnsw.log

###################################################################################################
# Test Knowhere float index qps
test_float_qps: test_float_qps_ivf_flat test_float_qps_ivf_sq8 test_float_qps_hnsw

test_float_qps_ivf_flat:
	./benchmark_float_qps --gtest_filter="Benchmark_float_qps.TEST_IVF_FLAT" | tee test_float_qps_ivf_flat.log
test_float_qps_ivf_sq8:
	./benchmark_float_qps --gtest_filter="Benchmark_float_qps.TEST_IVF_SQ8" | tee test_float_qps_ivf_sq8.log
test_float_qps_hnsw:
	./benchmark_float_qps --gtest_filter="Benchmark_float_qps.TEST_HNSW" | tee test_float_qps_hnsw.log

###################################################################################################
# Test Knowhere float index range qps
test_float_range_qps: test_float_range_qps_ivf_flat test_float_range_qps_ivf_sq8 test_float_range_qps_hnsw

test_float_range_qps_ivf_flat:
	./benchmark_float_range_qps --gtest_filter="Benchmark_float_range_qps.TEST_IVF_FLAT" | tee test_float_range_qps_ivf_flat.log
test_float_range_qps_ivf_sq8:
	./benchmark_float_range_qps --gtest_filter="Benchmark_float_range_qps.TEST_IVF_SQ8" | tee test_float_range_qps_ivf_sq8.log
test_float_range_qps_hnsw:
	./benchmark_float_range_qps --gtest_filter="Benchmark_float_range_qps.TEST_HNSW" | tee test_float_range_qps_hnsw.log

###################################################################################################
# Test Knowhere float index range multi qps
test_float_range_multi_qps: test_float_range_multi_qps_ivf_flat test_float_range_multi_qps_ivf_sq8 test_float_range_multi_qps_hnsw

test_float_range_multi_qps_ivf_flat:
	./benchmark_float_range_multi_qps --gtest_filter="Benchmark_float_range_multi_qps.TEST_IVF_FLAT" | tee test_float_range_multi_qps_ivf_flat.log
test_float_range_multi_qps_ivf_sq8:
	./benchmark_float_range_multi_qps --gtest_filter="Benchmark_float_range_multi_qps.TEST_IVF_SQ8" | tee test_float_range_multi_qps_ivf_sq8.log
test_float_range_multi_qps_hnsw:
	./benchmark_float_range_multi_qps --gtest_filter="Benchmark_float_range_multi_qps.TEST_HNSW" | tee test_float_range_multi_qps_hnsw.log
