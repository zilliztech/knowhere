add_definitions(-DKNOWHERE_WITH_DISKANN)
find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIR})
find_package(aio REQUIRED)
include_directories(${AIO_INCLUDE})
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
    thirdparty/DiskANN/src/logger.cpp
    thirdparty/DiskANN/src/utils.cpp)

find_package(folly REQUIRED)

add_library(diskann STATIC ${DISKANN_SOURCES})
target_link_libraries(diskann PUBLIC ${AIO_LIBRARIES}
                                     ${DISKANN_BOOST_PROGRAM_OPTIONS_LIB}
                                     nlohmann_json::nlohmann_json
                                     Folly::folly
                                     glog::glog)
if(__X86_64)
  target_compile_options(
    diskann PRIVATE -fno-builtin-malloc -fno-builtin-calloc
                    -fno-builtin-realloc -fno-builtin-free)
endif()
list(APPEND KNOWHERE_LINKER_LIBS diskann)
