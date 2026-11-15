# Copyright 2026 Todd Ingalls. All rights reserved.
# Use of this source code is governed by the MIT License found in the License.md file.

# ---------------------------------------------------------------------------
# mlmat central dependency configuration
#
# Included once from the top-level CMakeLists.txt. Provides two interface
# targets that the individual object projects link against:
#
#   mlmat::deps  — for objects built on mlpack
#       mlpack, ensmallen and cereal are consumed header-only straight from
#       their git submodules. Armadillo is used in "no wrapper" mode
#       (ARMA_DONT_USE_WRAPPER): its headers call BLAS/LAPACK directly, so
#       there is no libarmadillo to build and no third-party runtime
#       libraries to distribute.
#         * macOS   : Apple's Accelerate framework (system BLAS/LAPACK)
#         * Windows : static OpenBLAS (BLAS) + clapack (LAPACK), provided by
#                     the vcpkg manifest (vcpkg.json) — linked statically
#                     into each external, so nothing is shipped except .mxe
#
#   mlmat::fft   — for the spectral objects (buffer_stft, buffer_istft,
#                  buffer_griffinlim)
#       bqfft + bqvec, vendored as git submodules. On macOS bqfft is
#       compiled with HAVE_VDSP so it uses Accelerate's vDSP internally;
#       on Windows it falls back to its built-in double-precision FFT.
#       Only power-of-two FFT sizes are supported.
# ---------------------------------------------------------------------------

# --- mlpack-based objects --------------------------------------------------

add_library(mlmat_deps INTERFACE)
add_library(mlmat::deps ALIAS mlmat_deps)

target_include_directories(mlmat_deps INTERFACE
    "${CMAKE_SOURCE_DIR}/source/mlpack/src"
    "${CMAKE_SOURCE_DIR}/source/armadillo-code/include"
    "${CMAKE_SOURCE_DIR}/source/ensmallen/include"
    "${CMAKE_SOURCE_DIR}/source/cereal/include"
    "${CMAKE_SOURCE_DIR}/source/projects/shared"
)

# Use armadillo as a pure template library: link BLAS/LAPACK directly.
target_compile_definitions(mlmat_deps INTERFACE ARMA_DONT_USE_WRAPPER=1)

# The heavily templated mlpack headers produce large object files on MSVC.
target_compile_options(mlmat_deps INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
    $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>
)

if(APPLE)
    find_library(MLMAT_ACCELERATE_LIB Accelerate REQUIRED)
    target_link_libraries(mlmat_deps INTERFACE ${MLMAT_ACCELERATE_LIB})
elseif(WIN32)
    # BLAS and LAPACK come from vcpkg (see vcpkg.json). With the
    # x64-windows-static triplet this resolves to static OpenBLAS and
    # clapack, both of which are linked into the externals themselves.
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    target_link_libraries(mlmat_deps INTERFACE
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
    )
endif()

# --- spectral objects (bqfft) ----------------------------------------------

if (MSVC)
    # max-pretarget.cmake compiles every project target with the static
    # CRT (/MT), matching the vcpkg x64-windows-static ports (also
    # built against the static CRT). Targets defined here at the top
    # level see only CMake's default (/MD); match pretarget or the
    # buffer externals/docgens fail with LNK2038 (RuntimeLibrary
    # mismatch) linking mlmat::fft.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif ()

add_library(mlmat_fft STATIC
    "${CMAKE_SOURCE_DIR}/source/bqfft/src/FFT.cpp"
    "${CMAKE_SOURCE_DIR}/source/bqvec/src/Allocators.cpp"
    "${CMAKE_SOURCE_DIR}/source/bqvec/src/VectorOpsComplex.cpp"
)
add_library(mlmat::fft ALIAS mlmat_fft)

target_include_directories(mlmat_fft PUBLIC
    "${CMAKE_SOURCE_DIR}/source/bqfft"
    "${CMAKE_SOURCE_DIR}/source/bqvec"
)

# MSVC's <cmath> needs _USE_MATH_DEFINES before M_PI is visible.
target_compile_definitions(mlmat_fft PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>
)

if(APPLE)
    # Let bqfft use vDSP from the Accelerate framework on macOS.
    target_compile_definitions(mlmat_fft PRIVATE HAVE_VDSP=1)
    find_library(MLMAT_ACCELERATE_LIB2 Accelerate REQUIRED)
    target_link_libraries(mlmat_fft PRIVATE ${MLMAT_ACCELERATE_LIB2})
else()
    # bqfft needs an explicit backend: on non-Apple platforms use its
    # self-contained built-in double-precision FFT (no external
    # dependency). Without one, FFT.cpp #errors at compile time -- the
    # buffer objects were never buildable on Windows before this.
    target_compile_definitions(mlmat_fft PRIVATE USE_BUILTIN_FFT=1)
endif()
