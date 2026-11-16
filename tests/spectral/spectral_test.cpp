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

static void run_resample()
{
    // resample(): a 1 kHz sine at 44.1k must stay 1 kHz in the 48k world
    // (checked through an STFT peak-bin lookup), and a same-rate resample
    // must be the identity.
    const double in_rate = 44100.0, out_rate = 48000.0, f0 = 1000.0;
    const size_t n = 8192;
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i)
        x[i] = float(std::sin(2.0 * M_PI * f0 * i / in_rate));

    std::vector<float> same;
    NEW::mlmat::resample(x, same, in_rate, in_rate);
    check(same.size() == n, "resample same-rate length", same.size());
    bool identity = true;
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(same[i] - x[i]) > 1e-6) {
            identity = false;
            break;
        }
    }
    check(identity, "resample same-rate is identity");

    std::vector<float> y;
    NEW::mlmat::resample(x, y, in_rate, out_rate);
    check(y.size() == size_t(llround(n * out_rate / in_rate)), "resample output length", y.size());

    // spectral peak of the resampled signal: middle frame. (the
    // magnitudes/phases vectors are out-params and must be pre-sized by
    // the caller, like the objects do)
    const int fftsize = 1024;
    const size_t bins = fftsize / 2;
    const size_t step = fftsize / 1;
    const size_t frames_expected = (y.size() + step - 1) / step;
    STFT stft;
    std::vector<float> mags(frames_expected * bins), phases(frames_expected * bins);
    stft.process(y, mags, phases, fftsize, 1, "hanning", true, false);
    const size_t frames = bins > 0 ? mags.size() / bins : 0;
    check(frames == frames_expected, "resample stft frames", frames);
    if (frames >= 3) {
        const size_t mid = (frames / 2) * bins;
        size_t peak = 0;
        for (size_t k = 1; k < bins; ++k) {
            if (mags[mid + k] > mags[mid + peak])
                peak = k;
        }
        const double peak_hz = peak * out_rate / fftsize;
        check(std::fabs(peak_hz - f0) <= 1.5 * out_rate / fftsize,
              "resample keeps 1kHz peak", peak_hz - f0);
    }
}


static void run_spectral_ops()
{
    // ------------------------------------------------------------------
    // spectral_find_peaks
    // ------------------------------------------------------------------

    // a log-gaussian magnitude frame is exactly parabolic in log
    // space, so the interpolation must recover the peak exactly
    {
        const size_t bins = 32;
        const int fftsize = 64;
        const double sr = 48000.0;
        const double c_true = 15.4;
        const double a_true = 0.7;
        std::vector<float> m(bins);
        for (size_t k = 0; k < bins; ++k)
            m[k] = float(a_true * std::exp(-1.0 * (double(k) - c_true) * (double(k) - c_true)));

        std::vector<NEW::mlmat::spectral_peak> peaks;
        NEW::mlmat::spectral_find_peaks(m, sr, fftsize, -60.0f, 4, peaks);
        check(peaks.size() == 1, "gaussian frame: one peak", peaks.size());
        if (peaks.size() == 1) {
            check(std::fabs(peaks[0].frequency - c_true * sr / fftsize) < 1e-3,
                  "gaussian frame: exact frequency", peaks[0].frequency - c_true * sr / fftsize);
            check(std::fabs(peaks[0].amplitude - a_true) < 1e-3,
                  "gaussian frame: exact amplitude", peaks[0].amplitude - a_true);
        }
    }

    // a real windowed 1 kHz sine analyzed by the stack's STFT: the peak
    // must interpolate well below a bin
    {
        const int fftsize = 1024;
        const double sr = 48000.0;
        std::vector<float> x(fftsize);
        for (int i = 0; i < fftsize; ++i)
            x[i] = float(0.5 * std::sin(2.0 * M_PI * 1000.0 * i / sr));

        STFT stft;
        std::vector<float> mags(fftsize / 2), phases(fftsize / 2);
        stft.process(x, mags, phases, fftsize, 1, "hanning", true, false);

        std::vector<NEW::mlmat::spectral_peak> peaks;
        NEW::mlmat::spectral_find_peaks(mags, sr, fftsize, -60.0f, 8, peaks);
        check(peaks.size() >= 1, "sine frame: peaks found", peaks.size());
        if (!peaks.empty()) {
            check(std::fabs(peaks[0].frequency - 1000.0) <= 3.0,
                  "sine frame: peak at 1kHz", peaks[0].frequency - 1000.0);
            check(peaks[0].amplitude > 0.1, "sine frame: peak amplitude sane", peaks[0].amplitude);
        }
    }

    // ------------------------------------------------------------------
    // spectral_multiply: the half-spectrum packing (real DC in re[0],
    // real Nyquist in im[0]) must agree with multiplying the true
    // (full complex) spectra
    // ------------------------------------------------------------------
    {
        const int N = 16;
        const int half = N / 2;
        std::vector<double> a(N), b(N);
        for (int i = 0; i < N; ++i) {
            a[i] = std::sin(2.0 * M_PI * (i + 1) / 5.3) + 0.3 * std::cos(2.0 * M_PI * i / 3.1);
            b[i] = 0.6 * std::sin(2.0 * M_PI * (i + 2) / 7.7) + 0.2 * std::sin(2.0 * M_PI * i / 2.3);
        }
        // true spectra, k = 0..N/2
        std::vector<double> ra(half + 1), ia(half + 1), rb(half + 1), ib(half + 1);
        for (int k = 0; k <= half; ++k) {
            for (int t = 0; t < N; ++t) {
                const double c = std::cos(-2.0 * M_PI * double(k) * double(t) / N);
                const double si = std::sin(-2.0 * M_PI * double(k) * double(t) / N);
                ra[k] += a[t] * c; ia[k] += a[t] * si;
                rb[k] += b[t] * c; ib[k] += b[t] * si;
            }
        }

        // half-spectrum packing of the inputs
        std::vector<float> re1(half), im1(half), re2(half), im2(half);
        re1[0] = (float)ra[0]; im1[0] = (float)ra[half];
        re2[0] = (float)rb[0]; im2[0] = (float)rb[half];
        for (int k = 1; k < half; ++k) {
            re1[k] = (float)ra[k]; im1[k] = (float)ia[k];
            re2[k] = (float)rb[k]; im2[k] = (float)ib[k];
        }

        std::vector<float> re_out(half), im_out(half);
        NEW::mlmat::spectral_multiply(re1.data(), im1.data(), re2.data(), im2.data(),
                                     re_out.data(), im_out.data(), half, false, 1.0f);

        double max_err = 0.0;
        // index 0 packs product DC (re) and product Nyquist (im)
        max_err = std::max(max_err, std::fabs(re_out[0] - ra[0] * rb[0]));
        max_err = std::max(max_err, std::fabs(im_out[0] - ra[half] * rb[half]));
        for (int k = 1; k < half; ++k) {
            const double pr = ra[k] * rb[k] - ia[k] * ib[k];
            const double pi = ra[k] * ib[k] + ia[k] * rb[k];
            max_err = std::max(max_err, std::fabs(re_out[k] - pr));
            max_err = std::max(max_err, std::fabs(im_out[k] - pi));
        }
        check(max_err < 1e-3, "half-spectrum multiply matches true spectra", max_err);

        // full-spectrum multiply: conjugate-symmetric extension, all
        // bins complex
        std::vector<float> fre1(N), fim1(N), fre2(N), fim2(N), f_re_out(N), f_im_out(N);
        for (int k = 0; k <= half; ++k) {
            fre1[k] = (float)ra[k]; fim1[k] = (float)ia[k];
            fre2[k] = (float)rb[k]; fim2[k] = (float)ib[k];
        }
        for (int k = 1; k < half; ++k) {
            fre1[N - k] = (float)ra[k]; fim1[N - k] = (float)-ia[k];
            fre2[N - k] = (float)rb[k]; fim2[N - k] = (float)-ib[k];
        }
        NEW::mlmat::spectral_multiply(fre1.data(), fim1.data(), fre2.data(), fim2.data(),
                                     f_re_out.data(), f_im_out.data(), N, true, 1.0f);
        double max_err_full = 0.0;
        for (int k = 0; k <= half; ++k) {
            const double pr = ra[k] * rb[k] - ia[k] * ib[k];
            const double pi = ra[k] * ib[k] + ia[k] * rb[k];
            max_err_full = std::max(max_err_full, std::fabs(f_re_out[k] - pr));
            max_err_full = std::max(max_err_full, std::fabs(f_im_out[k] - pi));
        }
        check(max_err_full < 1e-3, "full-spectrum multiply matches true spectra", max_err_full);
    }

    // ------------------------------------------------------------------
    // end to end smoke: STFT a, STFT b, spectral multiply, ISTFT
    // ------------------------------------------------------------------
    {
        const int fftsize = 1024;
        const int overlap = 4;
        const size_t step = fftsize / overlap;
        const size_t nsamples = 4096;
        const size_t frames = (nsamples * overlap + fftsize - 1) / fftsize;
        const size_t bins = fftsize / 2;

        STFT stft_a, stft_b;
        ISTFT istft;
        std::vector<float> a(nsamples), b(nsamples);
        for (size_t i = 0; i < nsamples; ++i) {
            a[i] = float(0.4 * std::sin(2.0 * M_PI * 220.0 * i / 48000.0));
            b[i] = float(0.4 * std::sin(2.0 * M_PI * 330.0 * i / 48000.0));
        }
        std::vector<float> re_a(frames * bins), im_a(frames * bins);
        std::vector<float> re_b(frames * bins), im_b(frames * bins);
        stft_a.process(a, re_a, im_a, fftsize, overlap, "hanning", false, false);
        stft_b.process(b, re_b, im_b, fftsize, overlap, "hanning", false, false);

        // frame-wise multiply (the object's core operation)
        const size_t total = frames * bins;
        std::vector<float> re_p(total + step, 0.0f), im_p(total + step, 0.0f);
        for (size_t f = 0; f < frames; ++f) {
            NEW::mlmat::spectral_multiply(re_a.data() + f * bins, im_a.data() + f * bins,
                                          re_b.data() + f * bins, im_b.data() + f * bins,
                                          re_p.data() + f * bins, im_p.data() + f * bins,
                                          bins, false, 1.0f);
        }

        std::vector<float> samples(total + step, 0.0f);
        istft.process(re_p, im_p, samples, fftsize, overlap, "hanning", false, false);

        float mx = 0.0f;
        for (float v : samples)
            mx = std::max(mx, std::fabs(v));
        check(mx > 1e-4f, "stft x stft -> istft produces signal", mx);
    }
}

int main()
{
    run_parity();
    run_resample();
    run_spectral_ops();
    printf("spectral parity: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
