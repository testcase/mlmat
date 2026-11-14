# Copyright 2018 The Min-DevKit Authors. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

cmake_minimum_required(VERSION 3.10...3.31)
set(AUTHOR_DOMAIN "org.tenuous")
set(COPYRIGHT_STRING "Copyright (c) 2023 Todd Ingalls")

set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../min-api)
include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

#############################################################
# MAX EXTERNAL
#############################################################

set( SOURCE_FILES
	${PROJECT_NAME}.cpp
)

add_library(
	${PROJECT_NAME}
	MODULE
	${SOURCE_FILES}
)

include_directories(
	"${C74_INCLUDES}"
	"../shared"
)

# bqfft/bqvec FFT backend (uses vDSP via Accelerate on macOS).
# See source/cmake/mlmat_deps.cmake.
target_link_libraries(${PROJECT_NAME} PRIVATE mlmat::fft)

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /bigobj)
endif ()

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)
