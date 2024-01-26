# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED on )

if(MSVC)
	add_subdirectory(dll)
else()
	#file(GLOB CPP_SOURCES *.cpp)
	set(CPP_SOURCES ann_exception.cpp aux_utils.cpp distance.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp memory_mapper.cpp aio_context_pool.cpp
        partition_and_pq.cpp  pq_flash_index.cpp logger.cpp utils.cpp
		distance_neon.cpp)
	add_library(${PROJECT_NAME} STATIC ${CPP_SOURCES})
	set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
	# add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()
install()
install( TARGETS  ${PROJECT_NAME} ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
