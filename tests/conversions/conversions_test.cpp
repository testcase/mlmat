/// @file
/// Regression test for the arma <-> jit matrix conversion layer
/// (source/projects/shared/matrix_conversions.hpp).
///
/// The live implementation is compared bit-for-bit against
/// matrix_conversions_reference.hpp, a frozen copy of the original
/// element-by-element implementation. Both run against a small fake of the
/// Jitter API (fakejit.h/.cpp) so the tests run outside of Max.
///
/// Run with the "bench" argument to also print conversion timings.

#include "c74_min.h"
#include <armadillo>
#include "fakejit.h"

// pre-include these so that the headers included inside the namespaces
// below do not nest their contents in those namespaces
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>

namespace REF {
#include "matrix_conversions_reference.hpp"
}
namespace NEW {
#include "../../../source/projects/shared/matrix_conversions.hpp"
}

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

static int failures = 0;
static int checks = 0;

static void check(bool ok, const char* what)
{
    ++checks;
    if (!ok) {
        ++failures;
        printf("FAIL  %s\n", what);
    }
}

static bool same_matrix(const fakejit::FakeMatrix& a, const fakejit::FakeMatrix& b)
{
    if (a.info.dimcount != b.info.dimcount) return false;
    if (a.info.planecount != b.info.planecount) return false;
    if (a.info.type != b.info.type) return false;
    for (long i = 0; i < a.info.dimcount; ++i)
        if (a.info.dim[i] != b.info.dim[i]) return false;
    if (a.data.size() != b.data.size()) return false;
    return memcmp(a.data.data(), b.data.data(), a.data.size()) == 0;
}

static c74::max::t_jit_matrix_info base_info(c74::max::t_symbol* type,
                                             long dimcount, long d0, long d1, long d2, long pc)
{
    c74::max::t_jit_matrix_info info;
    memset(&info, 0, sizeof(info));
    info.type = type;
    info.flags = 0;
    info.dimcount = dimcount;
    info.dim[0] = d0;
    info.dim[1] = d1;
    info.dim[2] = d2;
    info.planecount = pc;
    return info;
}

static fakejit::FakeMatrix* new_dest()
{
    auto* m = new fakejit::FakeMatrix();
    m->info = base_info(c74::max::jit_sym_float64, 1, 1, 1, 0, 1);
    m->realize();
    return m;
}

template <typename ARMA>
static void test_arma_to_jit(const char* what,
                             int mode,
                             const ARMA& source,
                             c74::max::t_jit_matrix_info target,
                             bool is_coords = false,
                             long x = 0)
{
    auto* d_ref = new_dest();
    auto* d_new = new_dest();
    auto t1 = target;
    auto t2 = target;
    auto& src = const_cast<ARMA&>(source);
    REF::arma_to_jit(mode, src, (c74::max::t_object*)d_ref, t1, is_coords, x);
    NEW::arma_to_jit(mode, src, (c74::max::t_object*)d_new, t2, is_coords, x);
    check(same_matrix(*d_ref, *d_new), what);
    delete d_ref;
    delete d_new;
}

// build a source matrix with a packed or padded (unpacked) layout filled
// with a deterministic pattern (long patterns include negative values to
// exercise the int32 -> size_t conversion)
static fakejit::FakeMatrix* make_source(c74::max::t_symbol* type,
                                        long d0, long d1, long pc,
                                        long pad_bytes)
{
    auto* m = new fakejit::FakeMatrix();
    size_t es = (type == c74::max::jit_sym_float64) ? 8 : 4;
    m->info = base_info(type, 2, d0, d1, 0, pc);
    m->info.dimstride[0] = long(pc * es);
    m->info.dimstride[1] = long(d0 * pc * es + pad_bytes);
    m->info.dimcount = 2;
    size_t total = (size_t(d1) - 1) * size_t(m->info.dimstride[1]) + size_t(d0) * pc * es;
    m->info.size = long(total);
    m->data.assign(total, 0xAB);
    for (long jcol = 0; jcol < d1; ++jcol)
        for (long jrow = 0; jrow < d0; ++jrow)
            for (long p = 0; p < pc; ++p) {
                size_t idx = size_t(jcol) * size_t(m->info.dimstride[1]) +
                             size_t(jrow) * size_t(m->info.dimstride[0]) + size_t(p) * es;
                if (es == 8) {
                    double v = double((jcol * d0 + jrow) * pc + p) * 0.25 - 100.0;
                    memcpy(m->data.data() + idx, &v, 8);
                } else {
                    c74::max::t_int32 v = c74::max::t_int32((jcol * d0 + jrow) * pc + p) - 500;
                    memcpy(m->data.data() + idx, &v, 4);
                }
            }
    return m;
}

template <typename ARMA>
static void test_jit_to_arma(const char* what, int mode, fakejit::FakeMatrix* src)
{
    ARMA a_ref, a_new;
    REF::jit_to_arma(mode, (c74::max::t_object*)src, a_ref);
    NEW::jit_to_arma(mode, (c74::max::t_object*)src, a_new);
    bool ok = (a_ref.n_rows == a_new.n_rows && a_ref.n_cols == a_new.n_cols &&
               a_ref.n_elem == a_new.n_elem &&
               memcmp(a_ref.memptr(), a_new.memptr(), a_ref.n_elem * sizeof(a_ref[0])) == 0);
    check(ok, what);
}

// ---------------------------------------------------------------------------
// parity tests
// ---------------------------------------------------------------------------
static void run_parity()
{
    using c74::max::jit_sym_float64;
    using c74::max::jit_sym_long;

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    // ---------------- arma_to_jit ----------------
    {
        arma::Row<double> r24(24, arma::fill::none);
        for (auto& v : r24) v = dist(rng);

        // Row mode 0: 1-cell multi-plane
        test_arma_to_jit("arma_to_jit Row<double> m0 1d", 0, r24,
                         base_info(jit_sym_float64, 1, 1, 1, 0, 24));
        test_arma_to_jit("arma_to_jit Row<double> m0 1d coords pow2", 0, r24,
                         base_info(jit_sym_float64, 1, 12, 1, 0, 2), true, 8);
        test_arma_to_jit("arma_to_jit Row<double> m0 1d coords !pow2", 0, r24,
                         base_info(jit_sym_float64, 1, 12, 1, 0, 2), true, 7);
        test_arma_to_jit("arma_to_jit Row<double> m0 2d coords", 0, r24,
                         base_info(jit_sym_float64, 2, 12, 2, 0, 2), true, 8);
        test_arma_to_jit("arma_to_jit Row<double> m1", 1, r24, base_info(jit_sym_float64, 1, 4, 4, 0, 1));
        test_arma_to_jit("arma_to_jit Row<double> m2", 2, r24, base_info(jit_sym_float64, 1, 4, 4, 0, 1));

        arma::Row<size_t> ri24(24, arma::fill::none);
        for (size_t i = 0; i < 24; ++i) ri24(i) = i * 3 + 1;
        test_arma_to_jit("arma_to_jit Row<size_t> m0 1d", 0, ri24,
                         base_info(jit_sym_long, 1, 1, 1, 0, 24));
        test_arma_to_jit("arma_to_jit Row<size_t> m0 1d coords pow2", 0, ri24,
                         base_info(jit_sym_long, 1, 12, 1, 0, 2), true, 8);
        test_arma_to_jit("arma_to_jit Row<size_t> m0 2d coords", 0, ri24,
                         base_info(jit_sym_long, 2, 4, 6, 0, 2), true, 5);
        test_arma_to_jit("arma_to_jit Row<size_t> m1", 1, ri24, base_info(jit_sym_long, 1, 4, 4, 0, 1));
        test_arma_to_jit("arma_to_jit Row<size_t> m2", 2, ri24, base_info(jit_sym_long, 1, 4, 4, 0, 1));

        arma::Mat<double> m67(6, 7, arma::fill::none);
        for (size_t c = 0; c < 7; ++c)
            for (size_t r = 0; r < 6; ++r) m67(r, c) = dist(rng);
        test_arma_to_jit("arma_to_jit Mat<double> m0 1d", 0, m67,
                         base_info(jit_sym_float64, 1, 42, 1, 0, 1));
        // mode 0 reads planecount * d0 * d1 sequential source elements
        arma::Mat<double> m84d(12, 7, arma::fill::none);
        for (size_t c = 0; c < 7; ++c)
            for (size_t r = 0; r < 12; ++r) m84d(r, c) = dist(rng);
        arma::Mat<double> m126d(18, 7, arma::fill::none);
        for (size_t c = 0; c < 7; ++c)
            for (size_t r = 0; r < 18; ++r) m126d(r, c) = dist(rng);
        test_arma_to_jit("arma_to_jit Mat<double> m0 2d pc2", 0, m84d,
                         base_info(jit_sym_float64, 2, 3, 14, 0, 2));
        test_arma_to_jit("arma_to_jit Mat<double> m0 2d pc3", 0, m126d,
                         base_info(jit_sym_float64, 2, 7, 2, 0, 3));
        test_arma_to_jit("arma_to_jit Mat<double> m1", 1, m67, base_info(jit_sym_float64, 2, 4, 4, 0, 1));
        test_arma_to_jit("arma_to_jit Mat<double> m2", 2, m67, base_info(jit_sym_float64, 2, 4, 4, 0, 1));
        // 3d: every slice gets the same source data (reference semantics)
        test_arma_to_jit("arma_to_jit Mat<double> m0 3d pc2", 0, m84d,
                         base_info(jit_sym_float64, 3, 3, 14, 2, 2));
        // 3d coords: per-slice source rows
        arma::Mat<double> mcoords(2, 12, arma::fill::none);
        for (size_t c = 0; c < 12; ++c)
            for (size_t r = 0; r < 2; ++r) mcoords(r, c) = double(r * 12 + c);
        test_arma_to_jit("arma_to_jit Mat<double> m0 3d coords", 0, mcoords,
                         base_info(jit_sym_float64, 3, 4, 3, 2, 2), true, 8);

        arma::Mat<size_t> mi67(6, 7, arma::fill::none);
        for (size_t c = 0; c < 7; ++c)
            for (size_t r = 0; r < 6; ++r) mi67(r, c) = r * 7 + c;
        arma::Mat<size_t> mi84(12, 7, arma::fill::none);
        for (size_t c = 0; c < 7; ++c)
            for (size_t r = 0; r < 12; ++r) mi84(r, c) = r * 7 + c;
        arma::Mat<size_t> mi84p(12, 7, arma::fill::none);
        for (size_t c = 0; c < 7; ++c)
            for (size_t r = 0; r < 12; ++r) mi84p(r, c) = r * 7 + c;
        test_arma_to_jit("arma_to_jit Mat<size_t> m0 2d pc2", 0, mi84p,
                         base_info(jit_sym_long, 2, 3, 14, 0, 2));
        test_arma_to_jit("arma_to_jit Mat<size_t> m0 3d", 0, mi84,
                         base_info(jit_sym_long, 3, 3, 14, 2, 2));
        test_arma_to_jit("arma_to_jit Mat<size_t> m1", 1, mi67, base_info(jit_sym_long, 2, 4, 4, 0, 2));
        test_arma_to_jit("arma_to_jit Mat<size_t> m2", 2, mi67, base_info(jit_sym_long, 2, 4, 4, 0, 2));
    }

    // ---------------- jit_to_arma ----------------
    {
        for (long pc : {1L, 3L}) {
            auto* src = make_source(jit_sym_float64, 6, 4, pc, 0);      // packed
            auto* pad = make_source(jit_sym_float64, 6, 4, pc, 32);     // unpacked rows
            char what[128];
            snprintf(what, sizeof(what), "jit_to_arma Mat<double> m0 pc%ld packed", pc);
            test_jit_to_arma<arma::mat>(what, 0, src);
            snprintf(what, sizeof(what), "jit_to_arma Mat<double> m0 pc%ld unpacked", pc);
            test_jit_to_arma<arma::mat>(what, 0, pad);
            delete src;
            delete pad;
        }
        for (int mode : {1, 2}) {
            auto* src = make_source(jit_sym_float64, 6, 4, 1, 0);
            auto* pad = make_source(jit_sym_float64, 6, 4, 1, 32);
            char what[128];
            snprintf(what, sizeof(what), "jit_to_arma Mat<double> m%d packed", mode);
            test_jit_to_arma<arma::mat>(what, mode, src);
            snprintf(what, sizeof(what), "jit_to_arma Mat<double> m%d unpacked", mode);
            test_jit_to_arma<arma::mat>(what, mode, pad);
            delete src;
            delete pad;
        }
        // long matrices (negative values exercise the int32 -> uword wrap)
        for (int mode : {0, 1, 2}) {
            auto* src = make_source(jit_sym_long, 6, 4, mode == 0 ? 3 : 1, 0);
            auto* pad = make_source(jit_sym_long, 6, 4, mode == 0 ? 3 : 1, 16);
            char what[128];
            snprintf(what, sizeof(what), "jit_to_arma Mat<size_t> m%d packed", mode);
            test_jit_to_arma<arma::Mat<size_t>>(what, mode, src);
            snprintf(what, sizeof(what), "jit_to_arma Mat<size_t> m%d unpacked", mode);
            test_jit_to_arma<arma::Mat<size_t>>(what, mode, pad);
            delete src;
            delete pad;
        }
        // rows
        for (int mode : {0, 1, 2}) {
            auto* src = make_source(jit_sym_long, 24, 1, 1, 0);
            char what[128];
            snprintf(what, sizeof(what), "jit_to_arma Row<size_t> m%d packed", mode);
            test_jit_to_arma<arma::Row<size_t>>(what, mode, src);
            delete src;
        }
        auto* padrow = make_source(jit_sym_long, 24, 1, 1, 16);
        test_jit_to_arma<arma::Row<size_t>>("jit_to_arma Row<size_t> m1 unpacked", 1, padrow);
        delete padrow;

        // Row mode 0 with planecount > 1: the reference implementation wrote
        // past the end of its output row here; verify the fixed behavior is
        // well defined and flattens every plane.
        {
            auto* src = make_source(jit_sym_long, 8, 2, 3, 0);
            arma::Row<size_t> r;
            NEW::jit_to_arma(0, (c74::max::t_object*)src, r);
            bool ok = (r.n_elem == 48);
            for (long jcol = 0; ok && jcol < 2; ++jcol)
                for (long jrow = 0; ok && jrow < 8; ++jrow)
                    for (long p = 0; ok && p < 3; ++p) {
                        size_t idx = (size_t(jcol) * 8 + size_t(jrow)) * 3 + size_t(p);
                        c74::max::t_int32 expect = c74::max::t_int32((jcol * 8 + jrow) * 3 + p) - 500;
                        ok = (r(idx) == size_t(expect));
                    }
            check(ok, "jit_to_arma Row<size_t> m0 pc3 (fixed; reference overflew)");
            delete src;
        }
    }

    // ---------------- jit_to_arma_limit ----------------
    {
        auto limit_case = [&](const char* what, int mode, long pc, bool packed) {
            auto* src_ref = make_source(jit_sym_long, 6, 4, pc, packed ? 0 : 16);
            auto* src_new = make_source(jit_sym_long, 6, 4, pc, packed ? 0 : 16);
            arma::Col<arma::uword> c_ref, c_new;
            REF::jit_to_arma_limit(mode, (c74::max::t_object*)src_ref, c_ref, 5, 3);
            NEW::jit_to_arma_limit(mode, (c74::max::t_object*)src_new, c_new, 5, 3);
            bool ok = c_ref.n_elem == c_new.n_elem &&
                      memcmp(c_ref.memptr(), c_new.memptr(), c_ref.n_elem * sizeof(arma::uword)) == 0;
            check(ok, what);
            delete src_ref;
            delete src_new;
        };
        limit_case("jit_to_arma_limit m0 pc1 packed", 0, 1, true);
        limit_case("jit_to_arma_limit m0 pc1 unpacked", 0, 1, false);
        limit_case("jit_to_arma_limit m0 pc2 packed", 0, 2, true);
        limit_case("jit_to_arma_limit m1 packed", 1, 1, true);
        limit_case("jit_to_arma_limit m1 unpacked", 1, 1, false);
        limit_case("jit_to_arma_limit m2 packed", 2, 1, true);
        limit_case("jit_to_arma_limit m2 unpacked", 2, 1, false);
    }

    // ---------------- fill_jit_matrix on unpacked matrices ----------------
    {
        auto fill_case = [&](const char* what, int mode, long pc, bool is_coords, long x,
                             c74::max::t_symbol* type, bool long_arma) {
            auto* d_ref = new fakejit::FakeMatrix();
            auto* d_new = new fakejit::FakeMatrix();
            size_t es = (type == jit_sym_float64) ? 8 : 4;
            auto info = base_info(type, 2, 6, 4, 0, pc);
            info.dimstride[0] = long(pc * es);
            info.dimstride[1] = long(6 * pc * es + 24); // padded rows
            info.size = long((size_t(4) - 1) * size_t(info.dimstride[1]) + 6 * pc * es);
            for (auto* d : {d_ref, d_new}) {
                d->info = info;
                d->data.assign(size_t(info.size), 0xAB);
            }
            arma::Mat<double> md(24, 2, arma::fill::none);
            for (size_t c = 0; c < 2; ++c)
                for (size_t r = 0; r < 24; ++r) md(r, c) = double(r * 2 + c) - 10.0;
            arma::Mat<size_t> mi(24, 2, arma::fill::none);
            for (size_t c = 0; c < 2; ++c)
                for (size_t r = 0; r < 24; ++r) mi(r, c) = r * 2 + c;
            if (long_arma) {
                REF::fill_jit_matrix<c74::max::t_int32, arma::Mat<size_t>>((c74::max::t_object*)d_ref, mi, mode, is_coords, x);
                NEW::fill_jit_matrix<c74::max::t_int32, arma::Mat<size_t>>((c74::max::t_object*)d_new, mi, mode, is_coords, x);
            } else {
                REF::fill_jit_matrix<double, arma::Mat<double>>((c74::max::t_object*)d_ref, md, mode, is_coords, x);
                NEW::fill_jit_matrix<double, arma::Mat<double>>((c74::max::t_object*)d_new, md, mode, is_coords, x);
            }
            check(same_matrix(*d_ref, *d_new), what);
            delete d_ref;
            delete d_new;
        };
        fill_case("fill_jit_matrix double m0 pc2 unpacked", 0, 2, false, 0, jit_sym_float64, false);
        fill_case("fill_jit_matrix double m1 unpacked", 1, 1, false, 0, jit_sym_float64, false);
        fill_case("fill_jit_matrix double m2 unpacked", 2, 1, false, 0, jit_sym_float64, false);
        fill_case("fill_jit_matrix double m0 pc2 coords unpacked", 0, 2, true, 8, jit_sym_float64, false);
        fill_case("fill_jit_matrix size_t m0 pc2 unpacked", 0, 2, false, 0, jit_sym_long, true);
        fill_case("fill_jit_matrix size_t m1 unpacked", 1, 1, false, 0, jit_sym_long, true);
        fill_case("fill_jit_matrix size_t m2 unpacked", 2, 1, false, 0, jit_sym_long, true);
    }
}

// ---------------------------------------------------------------------------
// optional benchmarks (informational only, "bench" argument)
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
    using c74::max::jit_sym_float64;

    const int N = 512;
    arma::Mat<double> m(N, N, arma::fill::randu);
    auto* dest = new_dest();

    printf("\nbenchmark (ms per call, %dx%d double matrix):\n", N, N);
    for (int mode = 0; mode <= 2; ++mode) {
        auto target = (mode == 0) ? base_info(jit_sym_float64, 2, N, N, 0, 1)
                                  : base_info(jit_sym_float64, 2, 4, 4, 0, 1);
        double t_ref = bench_ms([&] {
            REF::arma_to_jit(mode, m, (c74::max::t_object*)dest, target, false, 0);
        }, 30);
        double t_new = bench_ms([&] {
            NEW::arma_to_jit(mode, m, (c74::max::t_object*)dest, target, false, 0);
        }, 30);
        printf("  arma_to_jit  mode %d:  reference %8.3f   current %8.3f   (%.2fx)\n",
               mode, t_ref, t_new, t_ref / t_new);
    }

    auto* src = make_source(jit_sym_float64, N, N, 1, 0);
    for (int mode = 0; mode <= 2; ++mode) {
        arma::mat a;
        double t_ref = bench_ms([&] {
            REF::jit_to_arma(mode, (c74::max::t_object*)src, a);
        }, 30);
        double t_new = bench_ms([&] {
            NEW::jit_to_arma(mode, (c74::max::t_object*)src, a);
        }, 30);
        printf("  jit_to_arma  mode %d:  reference %8.3f   current %8.3f   (%.2fx)\n",
               mode, t_ref, t_new, t_ref / t_new);
    }

    delete dest;
    delete src;
}

int main(int argc, char** argv)
{
    run_parity();
    printf("conversions parity: %d checks, %d failures\n", checks, failures);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "bench") == 0)
            run_bench();
    }
    return failures ? 1 : 0;
}
