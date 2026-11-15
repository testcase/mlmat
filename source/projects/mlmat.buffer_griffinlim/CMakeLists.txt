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


#############################################################
# REFERENCE PAGE GENERATOR
#
# Regenerates docs/mlmat.buffer_griffinlim.maxref.xml at build time, outside of Max:
# stock doc_generate against the min-api mock kernel, then the mlmat
# enhancements (attribute defaults, enum lists for the Inspector,
# list sizes, class discussion) -- the same output the old
# "better docs" min-api patch used to produce, with no min-api
# modification. Runs as a POST_BUILD step of the external, so the
# refpage always ends up newer than the .mxo and doc_update inside
# Max leaves the enhanced page alone.
#############################################################

if (NOT TARGET mock_kernel)
    set(C74_MOCK_TARGET_DIR "${CMAKE_BINARY_DIR}")
    add_subdirectory(${C74_MIN_API_DIR}/test/mock ${CMAKE_BINARY_DIR}/mock)
endif ()

add_executable(${PROJECT_NAME}_docgen
    ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/docgen/mlmat.buffer_griffinlim_docgen.cpp
)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_docgen)

target_include_directories(${PROJECT_NAME}_docgen PRIVATE
    "${C74_INCLUDES}"
    "${C74_MIN_API_DIR}/test"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/docgen"
    "../shared"
)
target_link_libraries(${PROJECT_NAME}_docgen PRIVATE mock_kernel mlmat::fft)
target_compile_definitions(${PROJECT_NAME}_docgen PRIVATE MIN_TEST)
set_property(TARGET ${PROJECT_NAME}_docgen PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME}_docgen PROPERTY CXX_STANDARD_REQUIRED ON)

if (APPLE)
    # jit and MSP buffer symbols referenced by the compiled-in code are
    # never called; resolve them weakly against the SDK's frameworks
    # like the min object unit tests do
    set_target_properties(${PROJECT_NAME}_docgen PROPERTIES
        LINK_FLAGS "-Wl,-F'${MAX_SDK_JIT_INCLUDES}', -Wl,-F'${MAX_SDK_MSP_INCLUDES}', -weak_framework JitterAPI, -weak_framework MaxAudioAPI")
elseif (WIN32)
    # jit symbols come from the SDK stub library, as for the externals;
    # the kernel (MaxAPI) side is provided by the mock kernel
    target_link_libraries(${PROJECT_NAME}_docgen PRIVATE ${Jitter_LIB})
endif ()


include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

# run the reference page generator LAST in the post-build sequence: the
# codesign step (registered by min-posttarget above) rewrites the
# external, so the page must be written after it to stay newer

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND $<TARGET_FILE:${PROJECT_NAME}_docgen> "${CMAKE_SOURCE_DIR}/docs/mlmat.buffer_griffinlim.maxref.xml"
    COMMENT "generating enhanced reference page for mlmat.buffer_griffinlim"
)

