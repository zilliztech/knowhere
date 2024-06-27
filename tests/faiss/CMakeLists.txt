set(FAISS_TEST_SRCS
  ../../thirdparty/faiss/tests/test_omp_threads.cpp
  ../../thirdparty/faiss/tests/test_heap.cpp
  ../../thirdparty/faiss/tests/test_partitioning.cpp
  ../../thirdparty/faiss/tests/test_simdlib.cpp
  ../../thirdparty/faiss/tests/test_approx_topk.cpp
  ../../thirdparty/faiss/tests/test_code_distance.cpp
  ../../thirdparty/faiss/tests/test_cppcontrib_uintreader.cpp
  ../../thirdparty/faiss/tests/test_distances_simd.cpp
  ../../thirdparty/faiss/tests/test_hnsw.cpp
  ../../thirdparty/faiss/tests/test_binary_flat.cpp
  ../../thirdparty/faiss/tests/test_cppcontrib_sa_decode.cpp
  ../../thirdparty/faiss/tests/test_ondisk_ivf.cpp
  ../../thirdparty/faiss/tests/test_pq_encoding.cpp
  ../../thirdparty/faiss/tests/test_RCQ_cropping.cpp
  ../../thirdparty/faiss/tests/test_ivfpq_codec.cpp
  ../../thirdparty/faiss/tests/test_ivfpq_indexing.cpp
  ../../thirdparty/faiss/tests/test_merge.cpp
  ../../thirdparty/faiss/tests/test_threaded_index.cpp
  ../../thirdparty/faiss/tests/test_pairs_decoding.cpp
  ../../thirdparty/faiss/tests/test_dealloc_invlists.cpp
  ../../thirdparty/faiss/tests/test_params_override.cpp
  ../../thirdparty/faiss/tests/test_sliding_ivf.cpp
  ../../thirdparty/faiss/tests/test_transfer_invlists.cpp
  ../../thirdparty/faiss/tests/test_distances_if.cpp
  ../../thirdparty/faiss/tests/test_fastscan_perf.cpp
  ../../thirdparty/faiss/tests/test_ivf_index.cpp
  ../../thirdparty/faiss/tests/test_disable_pq_sdc_tables.cpp
  ../../thirdparty/faiss/tests/test_common_ivf_empty_index.cpp
  ../../thirdparty/faiss/tests/test_callback.cpp
  )

find_package(GTest REQUIRED)

if(__X86_64)
  add_executable(faiss_tests ${FAISS_TEST_SRCS})

  add_dependencies(faiss_tests faiss)
  target_compile_options(
    faiss_tests
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -msse4.2
            -mavx2
            -mfma
            -mf16c
            -Wno-sign-compare
            -Wno-unused-variable
            -Wno-reorder
            -Wno-unused-local-typedefs
            -Wno-unused-function
            -Wno-strict-aliasing>)
  target_link_libraries(
    faiss_tests PUBLIC faiss knowhere GTest::gtest GTest::gtest_main)
  target_compile_definitions(faiss_tests PRIVATE FINTEGER=int)
endif()

if(__AARCH64)
  add_executable(faiss_tests ${FAISS_TEST_SRCS})
  target_compile_options(
    faiss_tests
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
            -Wno-sign-compare
            -Wno-unused-variable
            -Wno-reorder
            -Wno-unused-local-typedefs
            -Wno-unused-function
            -Wno-strict-aliasing>)

  add_dependencies(faiss_tests faiss)
  target_link_libraries(faiss_tests PUBLIC faiss knowhere GTest::gtest GTest::gtest_main)
  target_compile_definitions(faiss_tests PRIVATE FINTEGER=int)
endif()
