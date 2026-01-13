add_definitions(-DKNOWHERE_WITH_DISKANN)
find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIR})
find_package(aio REQUIRED)
include_directories(${AIO_INCLUDE})
find_package(fmt REQUIRED)

include_directories(${URING_INCLUDE})
include_directories(thirdparty/DiskANN/include)

find_package(double-conversion REQUIRED)
include_directories(${double-conversion_INCLUDE_DIRS})

set(DISKANN_SOURCES
    thirdparty/DiskANN/src/ann_exception.cpp
    thirdparty/DiskANN/src/aux_utils.cpp
    thirdparty/DiskANN/src/distance.cpp
    thirdparty/DiskANN/src/index.cpp
    thirdparty/DiskANN/src/linux_aligned_file_reader.cpp
    thirdparty/DiskANN/src/math_utils.cpp
    thirdparty/DiskANN/src/memory_mapper.cpp
    thirdparty/DiskANN/src/partition_and_pq.cpp
    thirdparty/DiskANN/src/pq_flash_index.cpp
    thirdparty/DiskANN/src/pq_flash_aisaq_index.cpp
    thirdparty/DiskANN/src/aisaq_utils.cpp
    thirdparty/DiskANN/src/aisaq_pq_reader.cpp
    thirdparty/DiskANN/src/logger.cpp
    thirdparty/DiskANN/src/utils.cpp)

find_package(folly REQUIRED)
set(DISKANN_LINKER_LIBS PUBLIC ${AIO_LIBRARIES} ${DISKANN_BOOST_PROGRAM_OPTIONS_LIB} nlohmann_json::nlohmann_json
         	Folly::folly fmt::fmt-header-only prometheus-cpp::core prometheus-cpp::push glog::glog)
if (WITH_CUVS)
    list(APPEND DISKANN_LINKER_LIBS PRIVATE cuvs::cuvs)
    list(APPEND DISKANN_SOURCES thirdparty/DiskANN/src/diskann_gpu.cpp)
endif()

add_library(diskann STATIC ${DISKANN_SOURCES})
target_link_libraries(diskann ${DISKANN_LINKER_LIBS})

if(__X86_64)
  target_compile_options(
    diskann PRIVATE -fno-builtin-malloc -fno-builtin-calloc
                    -fno-builtin-realloc -fno-builtin-free)
endif()

if (WITH_CUVS)
    target_link_libraries(diskann PRIVATE cuvs::cuvs)
endif()

list(APPEND KNOWHERE_LINKER_LIBS diskann)
