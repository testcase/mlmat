# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.0)

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

# Set directory for support folder
set(ARMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../armadillo)
set(MLPACK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../mlpack)
set(ENSMALLEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../ensmallen)
set(CEREAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../cereal)
set(BOOST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../boost_1_79_0)

#############################################################
# MAX EXTERNAL
#############################################################
find_path(MLPACK_HEADERS mlpack)

include_directories( 
	"${C74_INCLUDES}"
    "${MLMAT_SUPPORT_DIR}/mlpack/build/include"
    "${ARMA_DIR}/include"
    "${MLPACK_DIR}/build/include"
    "${ENSMALLEN_DIR}/include"
 	"${CEREAL_DIR}/include"
 	"${BOOST_DIR}"
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


if(APPLE)
	target_link_libraries(${PROJECT_NAME} PRIVATE debug "${ARMA_DIR}/build/Debug/libarmadillo.a")
	target_link_libraries(${PROJECT_NAME} PRIVATE debug "${MLPACK_DIR}/build/lib/Debug/libmlpack.a")
	target_link_libraries(${PROJECT_NAME} PRIVATE optimized "${ARMA_DIR}/build/Release/libarmadillo.a")
	target_link_libraries(${PROJECT_NAME} PRIVATE optimized "${MLPACK_DIR}/build/lib/Release/libmlpack.a")
	target_link_libraries(${PROJECT_NAME} PRIVATE general "${ACCELERATE_LIB}")
endif ()



if (WIN32)
	target_link_libraries(${PROJECT_NAME} PRIVATE debug "${ARMA_DIR}/build/Debug/armadillo.lib")
	target_link_libraries(${PROJECT_NAME} PRIVATE debug "${MLPACK_DIR}/build/Debug/mlpack.lib")
	target_link_libraries(${PROJECT_NAME} PRIVATE optimized "${ARMA_DIR}/build/Release/armadillo.lib")
	target_link_libraries(${PROJECT_NAME} PRIVATE optimized "${MLPACK_DIR}/build/Release/mlpack.lib")
endif ()

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

#############################################################
# UNIT TEST
#############################################################

#include(${C74_MIN_API_DIR}/test/min-object-unittest.cmake)

#target_link_libraries(${PROJECT_NAME}_test PRIVATE debug "${ARMA_DIR}/build/Debug/libarmadillo.a")
#target_link_libraries(${PROJECT_NAME}_test PRIVATE debug "${MLPACK_DIR}/build/lib/Debug/libmlpack.a")

#target_link_libraries(${PROJECT_NAME}_test PRIVATE optimized "${ARMA_DIR}/build/Release/libarmadillo.a")
#target_link_libraries(${PROJECT_NAME}_test PRIVATE optimized "${MLPACK_DIR}/build/lib/Release/libmlpack.a")

#if (APPLE)
#	target_link_libraries(${PROJECT_NAME} PRIVATE general "${ACCELERATE_LIB}")
#endif ()