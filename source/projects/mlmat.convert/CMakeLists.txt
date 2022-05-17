# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.0)

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

# Set directory for support folder
set(ARMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../armadillo)
set(MLPACK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../mlpack)

#############################################################
# MAX EXTERNAL
#############################################################
find_path(MLPACK_HEADERS mlpack)

include_directories( 
	"${C74_INCLUDES}"
    "${MLMAT_SUPPORT_DIR}/mlpack/build/include"
    "${ARMA_DIR}/include"
    "${MLPACK_DIR}/build/include"
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


#target_link_libraries(${PROJECT_NAME} PRIVATE debug "${ARMA_DIR}/build/Debug/libarmadillo.a")
#target_link_libraries(${PROJECT_NAME} PRIVATE debug "${MLPACK_DIR}/build/lib/Debug/libmlpack.a")
#
#
#target_link_libraries(${PROJECT_NAME} PRIVATE optimized "${ARMA_DIR}/build/Release/libarmadillo.a")
#target_link_libraries(${PROJECT_NAME} PRIVATE optimized "${MLPACK_DIR}/build/lib/Release/libmlpack.a")

if (APPLE)
	target_link_libraries(${PROJECT_NAME} PRIVATE general "${ACCELERATE_LIB}")
endif ()

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

#############################################################
# UNIT TEST
#############################################################

include(${C74_MIN_API_DIR}/test/min-object-unittest.cmake)
