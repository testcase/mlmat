/// @file
/// Regression tests for the spectral objects' shared core
/// (source/projects/shared/mlmat_stft.hpp): STFT/ISTFT round trips,
/// window caching, and the Griffin-Lim phase reconstruction (including
/// the momentum formulation and a comparison against the pre-fix
/// iteration scheme the object used to use).

#include "c74_min.h"

// pre-include everything the header below pulls in, so nothing is nested
// inside the NEW namespace
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>
#include "bqfft/FFT.h"

namespace NEW {
#include "../../source/projects/shared/mlmat_stft.hpp"
}

using NEW::STFT;
using NEW::ISTFT;
using NEW::GriffinLim;

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

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

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static std::vector<float> make_signal(size_t n, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = float(0.6 * std::sin(2.0 * M_PI * 13.0 * i / n) +
                     0.3 * std::sin(2.0 * M_PI * 47.0 * i / n + 0.7) +
                     0.1 * dist(rng));
    }
    return x;
}

// mean absolute difference of the magnitudes of a re-analysis against the
// reference magnitudes (the Griffin-Lim consistency error)
static double consistency_error(STFT& stft,
                               const std::vector<float>& samples,
                               const std::vector<float>& reference_magnitudes,
                               size_t num_bins,
                               int fftsize, int overlap,
                               const std::string& window_string,
                               bool full_spectrum)
{
    const size_t step = size_t(fftsize) / size_t(overlap);
    const size_t analysis_frames = (samples.size() + step - 1) / step;
    const long spec_dim = full_spectrum ? fftsize : fftsize / 2;
    std::vector<float> mags(analysis_frames * spec_dim, 0.0f);
    std::vector<float> phases(analysis_frames * spec_dim, 0.0f);
    stft.process(const_cast<std::vector<float>&>(samples), mags, phases, fftsize, overlap, window_string, true, full_spectrum);

    double err = 0.0;
    for (size_t i = 0; i < num_bins; ++i)
        err += std::fabs(double(mags[i]) - double(reference_magnitudes[i]));
    return err / num_bins;
}

// classic textbook Griffin-Lim (reference for momentum = 0)
static void textbook_gl(std::vector<float>& magnitudes,
                        std::vector<float>& phases,
                        size_t num_bins,
                        std::vector<float>& out_samples,
                        int fftsize, int overlap,
                        const std::string& window_string,
                        long iters,
                        bool full_spectrum)
{
    STFT stft;
    ISTFT istft;
    std::vector<float> rect_real(num_bins, 0.0f), rect_imag(num_bins, 0.0f);
    for (size_t i = 0; i < num_bins; ++i) {
        rect_real[i] = magnitudes[i] * std::cos(phases[i]);
        rect_imag[i] = magnitudes[i] * std::sin(phases[i]);
    }
    const size_t step = size_t(fftsize) / size_t(overlap);
    const size_t analysis_frames = (out_samples.size() + step - 1) / step;
    const long spec_dim = full_spectrum ? fftsize : fftsize / 2;
    std::vector<float> c_mag(analysis_frames * spec_dim, 0.0f);
    std::vector<float> c_phase(analysis_frames * spec_dim, 0.0f);
    for (long iter = 0; iter < iters; ++iter) {
        istft.process(rect_real, rect_imag, out_samples, fftsize, overlap, window_string, false, full_spectrum);
        stft.process(out_samples, c_mag, c_phase, fftsize, overlap, window_string, true, full_spectrum);
        for (size_t i = 0; i < num_bins; ++i) {
            rect_real[i] = magnitudes[i] * std::cos(c_phase[i]);
            rect_imag[i] = magnitudes[i] * std::sin(c_phase[i]);
        }
    }
    istft.process(rect_real, rect_imag, out_samples, fftsize, overlap, window_string, false, full_spectrum);
}

// historical reference: the iteration the object used before the magnitude
// restoration fix. the input magnitudes are never restored (and the polar
// -> rect conversion in the istft keeps re-interpreting them), so the
// reconstruction does not converge to a consistent spectrogram.
static void historical_gl(std::vector<float>& in_real,
                          std::vector<float>& in_imag,
                          size_t num_bins,
                          std::vector<float>& out_samples,
                          int fftsize, int overlap,
                          const std::string& window_string,
                          long iters,
                          bool full_spectrum)
{
    STFT stft;
    ISTFT istft;
    const size_t step = size_t(fftsize) / size_t(overlap);
    const size_t analysis_frames = (out_samples.size() + step - 1) / step;
    const long spec_dim = full_spectrum ? fftsize : fftsize / 2;
    std::vector<float> out_real(analysis_frames * spec_dim, 0.0f);
    std::vector<float> out_imag(analysis_frames * spec_dim, 0.0f);
    for (long iter = 0; iter < iters; ++iter) {
        istft.process(in_real, in_imag, out_samples, fftsize, overlap, window_string, true, full_spectrum);
        stft.process(out_samples, out_real, out_imag, fftsize, overlap, window_string, true, full_spectrum);
        in_imag = out_imag;
    }
    istft.process(in_real, in_imag, out_samples, fftsize, overlap, window_string, true, full_spectrum);
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------
static void run_parity()
{
    const int fftsize = 512;
    const int overlap = 4;
    const size_t n = 4096;
    const std::string win = "hanning";

    auto signal = make_signal(n, 12345);

    // -------- STFT / ISTFT round trip --------
    {
        const size_t step = size_t(fftsize) / size_t(overlap);
        const size_t frames = (n + step - 1) / step;
        const long spec_dim = fftsize / 2;
        const size_t bins = frames * spec_dim;

        STFT stft;
        ISTFT istft;
        std::vector<float> mags(bins, 0.0f), phases(bins, 0.0f);
        stft.process(signal, mags, phases, fftsize, overlap, win, true, false);

        std::vector<float> samples(bins + step, 0.0f);
        istft.process(mags, phases, samples, fftsize, overlap, win, true, false);

        // the interior of the signal should be recovered. (the analysis /
        // synthesis window and scaling convention leaves a small ripple,
        // about 0.3% for these settings)
        double max_diff = 0.0;
        for (size_t i = fftsize; i + fftsize < n; ++i)
            max_diff = std::max(max_diff, std::fabs(double(samples[i]) - double(signal[i])));
        check(max_diff < 1e-2, "stft -> istft round trip recovers the signal", max_diff);
    }

    // -------- window caching: repeated calls are bit-identical --------
    {
        const size_t step = size_t(fftsize) / size_t(overlap);
        const size_t frames = (n + step - 1) / step;
        const size_t bins = frames * (fftsize / 2);

        STFT stft_a;
        std::vector<float> mags1(bins, 0.0f), phases1(bins, 0.0f);
        std::vector<float> mags2(bins, 0.0f), phases2(bins, 0.0f);
        stft_a.process(signal, mags1, phases1, fftsize, overlap, win, true, false);
        stft_a.process(signal, mags2, phases2, fftsize, overlap, win, true, false);  // cached window
        bool ok = memcmp(mags1.data(), mags2.data(), bins * 4) == 0 &&
                  memcmp(phases1.data(), phases2.data(), bins * 4) == 0;
        check(ok, "stft window caching is bit-identical");

        // a different window in between, then back
        stft_a.process(signal, mags2, phases2, fftsize, overlap, "hamming", true, false);
        stft_a.process(signal, mags2, phases2, fftsize, overlap, win, true, false);
        ok = memcmp(mags1.data(), mags2.data(), bins * 4) == 0 &&
             memcmp(phases1.data(), phases2.data(), bins * 4) == 0;
        check(ok, "stft window reversion is bit-identical");

        ISTFT istft_a;
        const size_t out_len = bins + step;
        std::vector<float> s1(out_len, 0.0f), s2(out_len, 0.0f);
        // note: the polar -> rect conversion happens in place, so each call
        // gets its own copy of the input
        std::vector<float> mags_a = mags1, phases_a = phases1;
        std::vector<float> mags_b = mags1, phases_b = phases1;
        istft_a.process(mags_a, phases_a, s1, fftsize, overlap, win, true, false);
        istft_a.process(mags_b, phases_b, s2, fftsize, overlap, win, true, false);
        ok = memcmp(s1.data(), s2.data(), out_len * 4) == 0;
        check(ok, "istft window caching is bit-identical");
    }

    // -------- Griffin-Lim --------
    {
        const size_t step = size_t(fftsize) / size_t(overlap);
        const size_t frames = (n + step - 1) / step;
        const long spec_dim = fftsize / 2;
        const size_t bins = frames * spec_dim;

        STFT stft;
        std::vector<float> mags(bins, 0.0f), phases(bins, 0.0f);
        stft.process(signal, mags, phases, fftsize, overlap, win, true, false);

        const long iters = 10;
        const size_t out_len = bins + step;

        // initial estimate: magnitudes with zero phases
        std::vector<float> zero_phases(bins, 0.0f);
        double initial_err = consistency_error(stft, signal, mags, bins, fftsize, overlap, win, false);
        // (the initial error is measured against a zero-phase synthesis)
        {
            ISTFT istft;
            std::vector<float> samples0(out_len, 0.0f);
            istft.process(mags, zero_phases, samples0, fftsize, overlap, win, true, false);
            initial_err = consistency_error(stft, samples0, mags, bins, fftsize, overlap, win, false);
        }

        // historical (pre-fix) iteration
        double historical_err;
        {
            std::vector<float> in_real = mags, in_imag = zero_phases;
            std::vector<float> samples(out_len, 0.0f);
            historical_gl(in_real, in_imag, bins, samples, fftsize, overlap, win, iters, false);
            historical_err = consistency_error(stft, samples, mags, bins, fftsize, overlap, win, false);
        }

        // fixed algorithm, classic (momentum 0)
        double fixed_err;
        {
            GriffinLim gl;
            std::vector<float> in_mags = mags, in_phases = zero_phases;
            std::vector<float> samples(out_len, 0.0f);
            gl.process(in_mags, in_phases, bins, samples, fftsize, overlap, win, iters, 0.0, false);
            fixed_err = consistency_error(stft, samples, mags, bins, fftsize, overlap, win, false);
        }

        // fixed algorithm, momentum 0.9
        double fast_err;
        {
            GriffinLim gl;
            std::vector<float> in_mags = mags, in_phases = zero_phases;
            std::vector<float> samples(out_len, 0.0f);
            gl.process(in_mags, in_phases, bins, samples, fftsize, overlap, win, iters, 0.9, false);
            fast_err = consistency_error(stft, samples, mags, bins, fftsize, overlap, win, false);
        }

        printf("  consistency errors: initial %.4f  historical %.4f  fixed(m=0) %.6f  fixed(m=0.9) %.6f\n",
               initial_err, historical_err, fixed_err, fast_err);

        check(fixed_err < initial_err, "griffin-lim reduces the consistency error", fixed_err);
        check(fixed_err < historical_err, "fixed griffin-lim beats the historical iteration", fixed_err);
        // (with a zero-phase initial estimate, momentum converges more
        // slowly per iteration than the classic update; it only pays off
        // for large iteration counts. it must still converge.)
        check(fast_err < initial_err, "momentum griffin-lim still converges", fast_err);

        // momentum 0 matches the textbook algorithm
        {
            std::vector<float> t_mags = mags, t_phases = zero_phases;
            std::vector<float> t_samples(out_len, 0.0f);
            textbook_gl(t_mags, t_phases, bins, t_samples, fftsize, overlap, win, iters, false);

            GriffinLim gl;
            std::vector<float> in_mags = mags, in_phases = zero_phases;
            std::vector<float> samples(out_len, 0.0f);
            gl.process(in_mags, in_phases, bins, samples, fftsize, overlap, win, iters, 0.0, false);

            double max_diff = 0.0;
            for (size_t i = 0; i < out_len; ++i)
                max_diff = std::max(max_diff, std::fabs(double(samples[i]) - double(t_samples[i])));
            check(max_diff < 1e-4, "momentum 0 matches textbook griffin-lim", max_diff);
        }

        // determinism
        {
            GriffinLim gl_a, gl_b;
            std::vector<float> a_mags = mags, a_phases = zero_phases;
            std::vector<float> b_mags = mags, b_phases = zero_phases;
            std::vector<float> a_samples(out_len, 0.0f), b_samples(out_len, 0.0f);
            gl_a.process(a_mags, a_phases, bins, a_samples, fftsize, overlap, win, iters, 0.9, false);
            gl_b.process(b_mags, b_phases, bins, b_samples, fftsize, overlap, win, iters, 0.9, false);
            bool ok = memcmp(a_samples.data(), b_samples.data(), out_len * 4) == 0;
            check(ok, "griffin-lim is deterministic");
        }

        // invalid (non power of two) fftsize throws
        {
            GriffinLim gl;
            std::vector<float> in_mags = mags, in_phases = zero_phases;
            std::vector<float> samples(out_len, 0.0f);
            bool threw = false;
            try {
                gl.process(in_mags, in_phases, bins, samples, 1000, overlap, win, iters, 0.0, false);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            check(threw, "non power-of-two fftsize is rejected");
        }
    }
}

int main()
{
    run_parity();
    printf("spectral parity: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
