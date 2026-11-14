/// @file
/// Regression test for the mlmat.som SOM implementation
/// (source/projects/shared/mlmat_som.hpp).
///
/// The optimized implementation is compared against som_reference.hpp, a
/// frozen copy of the original implementation, with identical starting
/// maps and seeds. The optimization changes floating point evaluation
/// order (BLAS-based BMU search with an exact near-tie recheck, separable
/// influence lookup tables), so trained maps are compared within a tight
/// tolerance rather than bit-for-bit. Determinism of the new
/// implementation is checked bit-for-bit.

#include <armadillo>
#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>
#include <cereal/cereal.hpp>

namespace REF {
#include "som_reference.hpp"
}
namespace NEW {
#include "../../../source/projects/shared/mlmat_som.hpp"
}

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>

static int failures = 0;
static int checks = 0;

static void check(bool ok, const char* what, double detail = 0.0)
{
    ++checks;
    if (!ok) {
        ++failures;
        printf("FAIL  %s (detail %.3e)\n", what, detail);
    }
}

static void run_parity()
{
    const long cols = 8, rows = 6, weights = 12;
    const long epochs = 3;
    const long points = 200;

    arma::mat data(weights, points, arma::fill::none);
    mlpack::RandomSeed(4242);
    data.randu();

    // ---------------- BMU selection parity ----------------
    {
        mlpack::RandomSeed(1234);
        NEW::SOM som_new(cols, rows, weights, epochs, 0, 0.01, NEW::SOM::uniform);
        mlpack::RandomSeed(1234);
        REF::SOM som_ref(cols, rows, weights, epochs, 0, 0.01, REF::SOM::uniform);
        *som_new.m_nodes = *som_ref.m_nodes;   // identical starting maps
        som_new.recompute_node_norms();

        long mismatches = 0;
        for (long k = 0; k < points; ++k) {
            const size_t b_ref = som_ref.find_bmu(data.col(k));
            const size_t b_new = som_new.find_bmu(data.col(k));
            if (b_ref != b_new)
                ++mismatches;
        }
        check(mismatches == 0, "find_bmu matches reference on random points", double(mismatches));
    }

    // ---------------- online training parity ----------------
    {
        mlpack::RandomSeed(1234);
        REF::SOM som_ref(cols, rows, weights, epochs, 0, 0.01, REF::SOM::uniform);
        NEW::SOM som_new(cols, rows, weights, epochs, 0, 0.01, NEW::SOM::uniform);
        *som_new.m_nodes = *som_ref.m_nodes;

        som_ref.run_epochs(data);
        som_new.run_epochs(data);

        const arma::mat& a = *som_ref.m_nodes;
        const arma::mat& b = *som_new.m_nodes;
        double max_diff = 0.0;
        for (size_t i = 0; i < a.n_elem; ++i)
            max_diff = std::max(max_diff, std::fabs(a(i) - b(i)));
        const double scale = std::max(1e-30, arma::max(arma::max(arma::abs(a))));
        check(max_diff / scale < 1e-9, "run_epochs map matches reference (rel tol 1e-9)", max_diff / scale);
    }

    // ---------------- batch training parity ----------------
    {
        mlpack::RandomSeed(5678);
        REF::SOM som_ref(cols, rows, weights, epochs, 0, 0.01, REF::SOM::uniform);
        NEW::SOM som_new(cols, rows, weights, epochs, 0, 0.01, NEW::SOM::uniform);
        *som_new.m_nodes = *som_ref.m_nodes;

        som_ref.run_batch_knn(data);
        som_new.run_batch_knn(data);

        const arma::mat& a = *som_ref.m_nodes;
        const arma::mat& b = *som_new.m_nodes;
        double max_diff = 0.0;
        for (size_t i = 0; i < a.n_elem; ++i)
            max_diff = std::max(max_diff, std::fabs(a(i) - b(i)));
        const double scale = std::max(1e-30, arma::max(arma::max(arma::abs(a))));
        check(max_diff / scale < 1e-9, "run_batch_knn map matches reference (rel tol 1e-9)", max_diff / scale);
    }

    // ---------------- sample initialization parity ----------------
    {
        // both instances start from identical (uniform) nodes and sample
        // the same training columns (the training path consumes no other
        // randomness, so resetting the seed gives identical picks)
        mlpack::RandomSeed(999);
        REF::SOM som_ref(cols, rows, weights, epochs, 0, 0.01, REF::SOM::sample);
        NEW::SOM som_new(cols, rows, weights, epochs, 0, 0.01, NEW::SOM::sample);
        *som_new.m_nodes = *som_ref.m_nodes;

        mlpack::RandomSeed(555);
        som_ref.run_epochs(data);
        mlpack::RandomSeed(555);
        som_new.run_epochs(data);

        const arma::mat& a = *som_ref.m_nodes;
        const arma::mat& b = *som_new.m_nodes;
        double max_diff = 0.0;
        for (size_t i = 0; i < a.n_elem; ++i)
            max_diff = std::max(max_diff, std::fabs(a(i) - b(i)));
        const double scale = std::max(1e-30, arma::max(arma::max(arma::abs(a))));
        check(max_diff / scale < 1e-9, "sample init + training matches reference (rel tol 1e-9)", max_diff / scale);
    }

    // ---------------- gaussian initialization parity ----------------
    {
        mlpack::RandomSeed(31337);
        REF::SOM som_ref(cols, rows, weights, epochs, 0, 0.01, REF::SOM::gaussian);
        mlpack::RandomSeed(31337);
        NEW::SOM som_new(cols, rows, weights, epochs, 0, 0.01, NEW::SOM::gaussian);
        double max_diff = 0.0;
        const arma::mat& a = *som_ref.m_nodes;
        const arma::mat& b = *som_new.m_nodes;
        for (size_t i = 0; i < a.n_elem; ++i)
            max_diff = std::max(max_diff, std::fabs(a(i) - b(i)));
        check(max_diff == 0.0, "gaussian init is identical for identical seed", max_diff);
    }

    // ---------------- determinism of the new implementation ----------------
    {
        arma::mat map_a, map_b;
        mlpack::RandomSeed(777);
        {
            NEW::SOM s(cols, rows, weights, epochs, 0, 0.01, NEW::SOM::uniform);
            s.run_epochs(data);
            map_a = *s.m_nodes;
        }
        mlpack::RandomSeed(777);
        {
            NEW::SOM s(cols, rows, weights, epochs, 0, 0.01, NEW::SOM::uniform);
            s.run_epochs(data);
            map_b = *s.m_nodes;
        }
        bool ok = map_a.n_elem == map_b.n_elem &&
                  memcmp(map_a.memptr(), map_b.memptr(), map_a.n_elem * sizeof(double)) == 0;
        check(ok, "new implementation is deterministic for a fixed seed");
    }

    // ---------------- degenerate 1x1 map (new guard) ----------------
    {
        NEW::SOM s(1, 1, weights, epochs, 0, 0.01, NEW::SOM::uniform);
        s.run_epochs(data);
        bool finite = true;
        for (size_t i = 0; i < s.m_nodes->n_elem; ++i)
            if (!std::isfinite((*s.m_nodes)(i)))
                finite = false;
        check(finite, "1x1 map trains without NaN (new guard)");
    }
}

// ---------------------------------------------------------------------------
// optional benchmarks ("bench" argument)
// ---------------------------------------------------------------------------
template <typename F>
static double bench_ms(F&& fn, int iters)
{
    fn(); // warmup
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static void run_bench()
{
    const long cols = 16, rows = 16, weights = 32;
    const long epochs = 3;
    const long points = 500;

    arma::mat data(weights, points, arma::fill::randu);

    printf("\nbenchmark (16x16 map, %ld weights, %ld points, %ld epochs):\n",
           weights, points, epochs);
    {
        REF::SOM som_ref(cols, rows, weights, epochs, 0, 0.01, REF::SOM::uniform);
        NEW::SOM som_new(cols, rows, weights, epochs, 0, 0.01, NEW::SOM::uniform);
        *som_new.m_nodes = *som_ref.m_nodes;
        double t_ref = bench_ms([&] {
            REF::SOM s(cols, rows, weights, epochs, 0, 0.01, REF::SOM::uniform);
            *s.m_nodes = *som_ref.m_nodes;
            s.run_epochs(data);
        }, 5);
        double t_new = bench_ms([&] {
            NEW::SOM s(cols, rows, weights, epochs, 0, 0.01, NEW::SOM::uniform);
            *s.m_nodes = *som_new.m_nodes;
            s.run_epochs(data);
        }, 5);
        printf("  run_epochs:     reference %8.1f   current %8.1f   (%.2fx)\n",
               t_ref, t_new, t_ref / t_new);
    }
    {
        REF::SOM som_ref(cols, rows, weights, epochs, 0, 0.01, REF::SOM::uniform);
        NEW::SOM som_new(cols, rows, weights, epochs, 0, 0.01, NEW::SOM::uniform);
        *som_new.m_nodes = *som_ref.m_nodes;
        double t_ref = bench_ms([&] {
            REF::SOM s(cols, rows, weights, epochs, 0, 0.01, REF::SOM::uniform);
            *s.m_nodes = *som_ref.m_nodes;
            s.run_batch_knn(data);
        }, 5);
        double t_new = bench_ms([&] {
            NEW::SOM s(cols, rows, weights, epochs, 0, 0.01, NEW::SOM::uniform);
            *s.m_nodes = *som_new.m_nodes;
            s.run_batch_knn(data);
        }, 5);
        printf("  run_batch_knn:  reference %8.1f   current %8.1f   (%.2fx)\n",
               t_ref, t_new, t_ref / t_new);
    }
}

int main(int argc, char** argv)
{
    run_parity();
    printf("som parity: %d checks, %d failures\n", checks, failures);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "bench") == 0)
            run_bench();
    }
    return failures ? 1 : 0;
}
