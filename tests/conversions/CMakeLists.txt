# Regression tests for the arma <-> jit matrix conversion layer.
#
# conversions_test.cpp compares the live implementation
# (source/projects/shared/matrix_conversions.hpp) bit-for-bit against
# matrix_conversions_reference.hpp, a frozen copy of the original
# element-by-element implementation, using a fake of the Jitter API so the
# tests run outside of Max. Any change to the conversion layer must keep
# parity with the reference (or update the reference deliberately, with a
# explanation of the intended behavior change).
#
# Run with ctest from the build directory, or directly:
#   ./mlmat_conversions_test          parity checks only
#   ./mlmat_conversions_test bench    also print conversion timings

add_executable(mlmat_conversions_test
    conversions_test.cpp
    fakejit.cpp
)

# the stub c74_min.h in this directory must shadow any real min header
target_include_directories(mlmat_conversions_test
    BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# armadillo headers + BLAS/LAPACK linkage (Accelerate on macOS)
target_link_libraries(mlmat_conversions_test PRIVATE mlmat::deps)

set_property(TARGET mlmat_conversions_test PROPERTY CXX_STANDARD 17)
set_property(TARGET mlmat_conversions_test PROPERTY CXX_STANDARD_REQUIRED ON)

add_test(NAME mlmat.conversions COMMAND mlmat_conversions_test)
