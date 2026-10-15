# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_STANDARD_REQUIRED on )

if(MSVC)
	add_subdirectory(dll)
else()
	#file(GLOB CPP_SOURCES *.cpp)
	set(CPP_SOURCES ann_exception.cpp aux_utils.cpp distance.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp memory_mapper.cpp
        partition_and_pq.cpp  ssd_graph_index.cpp navigation_runtime.cpp pq_navigation.cpp rabitq_navigation.cpp logger.cpp utils.cpp
		distance_neon.cpp)
	if(NOT TARGET faiss)
        find_package(faiss CONFIG REQUIRED)
    endif()
	add_library(${PROJECT_NAME} STATIC ${CPP_SOURCES})
    target_link_libraries(${PROJECT_NAME} PUBLIC faiss)
	set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
	# add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()
install()
install( TARGETS  ${PROJECT_NAME} ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
