# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.10...3.31)
set(AUTHOR_DOMAIN "org.tenuous")
set(COPYRIGHT_STRING "Copyright (c) 2023 Todd Ingalls")

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

# Set directory for support folder
set(ARMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../armadillo-code)
set(MLPACK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../mlpack)
set(ENSMALLEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../ensmallen)
set(CEREAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../cereal)

#############################################################
# MAX EXTERNAL
#############################################################
find_path(MLPACK_HEADERS mlpack)

include_directories( 
	"${C74_INCLUDES}"
    "${MLPACK_DIR}/src"
    "${ARMA_DIR}/include"
    "${MLPACK_DIR}/src"
    "${ENSMALLEN_DIR}/include"
 	"${CEREAL_DIR}/include"
    "/usr/local/include"
    "../shared"
)


set( SOURCE_FILES
	${PROJECT_NAME}.cpp
)


add_library( 
	${PROJECT_NAME} 
	MODULE
	${SOURCE_FILES}
)

find_library(
    ACCELERATE_LIB
    accelerate
)

if (APPLE)
	target_link_libraries(${PROJECT_NAME} PRIVATE general "${ACCELERATE_LIB}")
endif ()

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

#############################################################
# UNIT TEST
#############################################################

#include(${C74_MIN_API_DIR}/test/min-object-unittest.cmake)
