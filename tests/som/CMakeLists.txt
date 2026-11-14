# Regression test for the mlmat.som SOM implementation.
#
# som_test.cpp compares the optimized implementation
# (source/projects/shared/mlmat_som.hpp) against som_reference.hpp, a
# frozen copy of the original implementation, with identical starting maps
# and seeds. The optimization changes floating point evaluation order, so
# trained maps are compared within a tight tolerance; determinism of the
# new implementation itself is checked bit-for-bit.
#
# Run with ctest from the build directory, or directly:
#   ./mlmat_som_test         parity checks only
#   ./mlmat_som_test bench   also print training timings

add_executable(mlmat_som_test
    som_test.cpp
)

target_link_libraries(mlmat_som_test PRIVATE mlmat::deps)

set_property(TARGET mlmat_som_test PROPERTY CXX_STANDARD 17)
set_property(TARGET mlmat_som_test PROPERTY CXX_STANDARD_REQUIRED ON)

add_test(NAME mlmat.som COMMAND mlmat_som_test)
