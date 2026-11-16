# Regression tests for the spectral objects' shared core
# (source/projects/shared/mlmat_stft.hpp).
#
# spectral_test.cpp checks STFT/ISTFT round trips, bit-identical window
# caching, and the Griffin-Lim reconstruction: the fixed algorithm (with
# the momentum formulation) must reduce the consistency error, beat the
# historical pre-fix iteration, match the textbook algorithm for momentum 0,
# and be deterministic.

add_executable(mlmat_spectral_test
    spectral_test.cpp
)

# the stub c74_min.h in this directory must shadow the real min header
target_include_directories(mlmat_spectral_test
    BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(mlmat_spectral_test PRIVATE mlmat::fft)

# MSVC's <cmath> only defines M_PI with _USE_MATH_DEFINES; this test
# target sees neither the C74 pretarget's definitions (project targets
# only) nor mlmat::fft's private ones.
target_compile_definitions(mlmat_spectral_test PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>)

set_property(TARGET mlmat_spectral_test PROPERTY CXX_STANDARD 17)
set_property(TARGET mlmat_spectral_test PROPERTY CXX_STANDARD_REQUIRED ON)

add_test(NAME mlmat.spectral COMMAND mlmat_spectral_test)
