/// @file
/// Regression tests for the spectral objects' shared core
/// (source/projects/shared/mlmat_stft.hpp): STFT/ISTFT round trips,
/// window caching, the Griffin-Lim phase reconstruction (including
/// the momentum formulation and a comparison against the pre-fix
/// iteration scheme the object used to use), phase-based frequency
/// reassignment (including through the STFT stack itself), the
/// window coherent-gain amplitude correction used by
/// mlmat.spectral_peaks and mlmat.spectral_reassign, the two-input
/// cores of mlmat.spectral_mask and mlmat.spectral_op, peak pruning
/// and partial tracking, and the McAulay-Quatieri synthesis core of
/// mlmat.buffer_mqsynth.

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

static void run_reassignment()
{
    // ------------------------------------------------------------------
    // reassigned_frequency: the phase-based instantaneous frequency
    // estimate for a sinusoid at a non-bin-center frequency must give
    // the exact frequency (not an approximation like the parabola)
    // ------------------------------------------------------------------
    {
        const int fftsize = 1024;
        const int overlap = 4;
        const double sr = 48000.0;
        const double hop = double(fftsize) / overlap;

        // sinusoid at 1000.3 Hz: bin 21.335 (not a bin center)
        const double f0 = 1000.3;

        // generate two consecutive STFT frames and extract the phase
        // of the nearest bin
        std::vector<float> x1(fftsize), x2(fftsize);
        for (int i = 0; i < fftsize; ++i) {
            x1[i] = float(0.5 * std::sin(2.0 * M_PI * f0 * i / sr));
            x2[i] = float(0.5 * std::sin(2.0 * M_PI * f0 * (i + hop) / sr));
        }

        // find the nearest bin
        const long k = long(f0 * fftsize / sr);   // 21

        // compute the phase of bin k in each frame using the DFT
        auto bin_phase = [&](const std::vector<float>& sig) -> double {
            double re = 0, im = 0;
            for (int t = 0; t < fftsize; ++t) {
                const double c = std::cos(-2.0 * M_PI * double(k) * double(t) / fftsize);
                const double si = std::sin(-2.0 * M_PI * double(k) * double(t) / fftsize);
                re += sig[t] * c;
                im += sig[t] * si;
            }
            return std::atan2(im, re);
        };

        const double phi1 = bin_phase(x1);
        const double phi2 = bin_phase(x2);

        const double freq = NEW::mlmat::reassigned_frequency(
            k, phi2, phi1, fftsize, overlap, sr);

        // for a pure sinusoid the reassignment is exact (within
        // float precision of the phase computation)
        check(std::fabs(freq - f0) < 0.5,
              "reassigned frequency of 1000.3Hz sine", freq - f0);
    }

    // a bin-center frequency should also be exact (zero phase deviation)
    {
        const int fftsize = 1024;
        const int overlap = 4;
        const double sr = 48000.0;
        const double hop = double(fftsize) / overlap;
        const long k = 21;   // bin 21 = 984.375 Hz exactly
        const double f0 = double(k) * sr / fftsize;

        std::vector<float> x1(fftsize), x2(fftsize);
        for (int i = 0; i < fftsize; ++i) {
            x1[i] = float(0.5 * std::sin(2.0 * M_PI * f0 * i / sr));
            x2[i] = float(0.5 * std::sin(2.0 * M_PI * f0 * (i + hop) / sr));
        }

        auto bin_phase = [&](const std::vector<float>& sig) -> double {
            double re = 0, im = 0;
            for (int t = 0; t < fftsize; ++t) {
                const double c = std::cos(-2.0 * M_PI * double(k) * double(t) / fftsize);
                const double si = std::sin(-2.0 * M_PI * double(k) * double(t) / fftsize);
                re += sig[t] * c;
                im += sig[t] * si;
            }
            return std::atan2(im, re);
        };

        const double freq = NEW::mlmat::reassigned_frequency(
            k, bin_phase(x2), bin_phase(x1), fftsize, overlap, sr);

        check(std::fabs(freq - f0) < 0.1,
              "reassigned frequency of bin-center sine", freq - f0);
    }

    // ------------------------------------------------------------------
    // through the STFT stack: the phases of the same bin in consecutive
    // frames of a full analysis give the exact frequency of a
    // non-bin-center sinusoid (this is the path mlmat.spectral_reassign
    // takes through mlmat.buffer_stft's output)
    // ------------------------------------------------------------------
    {
        const int fftsize = 1024;
        const int overlap = 4;
        const double sr = 48000.0;
        const size_t step = size_t(fftsize) / size_t(overlap);
        const size_t spec_dim = size_t(fftsize) / 2;

        const double f0 = 5.4 * sr / fftsize;   // 253.125 Hz: not a bin center
        const long k = 5;                      // nearest bin

        const size_t n = 8 * step + size_t(fftsize);
        std::vector<float> x(n);
        for (size_t i = 0; i < n; ++i)
            x[i] = float(0.5 * std::sin(2.0 * M_PI * f0 * double(i) / sr));

        const size_t frames = (n + step - 1) / step;
        STFT stft;
        std::vector<float> mags(frames * spec_dim, 0.0f), phases(frames * spec_dim, 0.0f);
        stft.process(x, mags, phases, fftsize, overlap, "hanning", true, false);

        // two consecutive frames from the middle of the analysis
        const size_t f1 = frames / 2;
        const double freq = NEW::mlmat::reassigned_frequency(
            k, double(phases[(f1 + 1) * spec_dim + size_t(k)]),
            double(phases[f1 * spec_dim + size_t(k)]),
            fftsize, overlap, sr);

        check(std::fabs(freq - f0) < 0.5,
              "stack phases reassign the off-bin sine exactly", freq - f0);

        // the peak magnitude of the bin-centered case is handled in
        // run_amplitude_correction(); here sanity-check the raw
        // magnitude is what the correction formula expects
        check(mags[f1 * spec_dim + size_t(k)] > 0.1f,
              "stack phases: peak magnitude sane", mags[f1 * spec_dim + size_t(k)]);
    }
}

// ---------------------------------------------------------------------------
// amplitude correction (mlmat.spectral_reassign's plane 0)
// ---------------------------------------------------------------------------
static void run_amplitude_correction()
{
    const int fftsize = 1024;
    const double sr = 48000.0;

    // ------------------------------------------------------------------
    // coherent gains of the five windows match their analytic values
    // (make_window reproduces the vDSP conventions: hanning is the
    // periodic Hann scaled by sqrt(8/3), so its gain is sqrt(8/3)/2)
    // ------------------------------------------------------------------
    {
        struct { const char* name; double expected; } wins[] = {
            {"square",   1.0},
            {"triangle", 0.5},
            {"hanning",  0.5 * std::sqrt(8.0 / 3.0)},
            {"hamming",  0.54},
            {"blackman", 0.42},
        };
        for (const auto& w : wins) {
            const double cg = NEW::mlmat::window_coherent_gain(fftsize, w.name);
            check(std::fabs(cg - w.expected) < 1e-3,
                  "coherent gain of the analytic value", cg - w.expected);
        }
    }

    // ------------------------------------------------------------------
    // a bin-centered sinusoid of amplitude A, analyzed with each
    // window and corrected by 2 * mag / (fftsize * cg), reads back as A
    // (the correction the object applies to every complex bin)
    // ------------------------------------------------------------------
    {
        const double A = 0.37;
        const size_t k0 = 5;
        const double f0 = double(k0) * sr / fftsize;
        std::vector<float> x(static_cast<size_t>(fftsize), 0.0f);
        for (int i = 0; i < fftsize; ++i)
            x[size_t(i)] = float(A * std::sin(2.0 * M_PI * f0 * i / sr));

        for (const char* wname : {"square", "triangle", "hanning", "hamming", "blackman"}) {
            STFT stft;
            std::vector<float> mags(size_t(fftsize) / 2, 0.0f), phases(size_t(fftsize) / 2, 0.0f);
            stft.process(x, mags, phases, fftsize, 1, wname, true, false);

            const double cg = NEW::mlmat::window_coherent_gain(fftsize, wname);
            const double amp = 2.0 * double(mags[k0]) / (double(fftsize) * cg);
            check(std::fabs(amp - A) < 5e-3,
                  "corrected magnitude recovers the sinusoid amplitude", amp - A);
        }
    }

    // ------------------------------------------------------------------
    // DC: with the half-spectrum rect convention the real DC lives in
    // re[0] with no factor 2 (all the energy in one real bin), so a
    // constant of D corrects as D = mag / (fftsize * cg). through the
    // polar output the bin 0 magnitude is |X[0]| -- the same value
    // ------------------------------------------------------------------
    {
        const double D = 0.42;
        std::vector<float> x(static_cast<size_t>(fftsize), static_cast<float>(D));

        STFT stft;
        std::vector<float> mags(size_t(fftsize) / 2, 0.0f), phases(size_t(fftsize) / 2, 0.0f);
        stft.process(x, mags, phases, fftsize, 1, "hanning", true, false);

        const double cg = NEW::mlmat::window_coherent_gain(fftsize, "hanning");
        const double amp = double(mags[0]) / (double(fftsize) * cg);
        check(std::fabs(amp - D) < 5e-3,
              "corrected DC recovers the constant's amplitude", amp - D);
    }

    // ------------------------------------------------------------------
    // the same correction applied to spectral_find_peaks' interpolated
    // peak amplitudes (what mlmat.spectral_peaks outputs in its
    // default amplitudes mode): a bin-centered sinusoid's peak reads
    // back at A through every window
    // ------------------------------------------------------------------
    {
        const double A = 0.5;
        const size_t k0 = 5;
        const double f0 = double(k0) * sr / fftsize;
        std::vector<float> x(static_cast<size_t>(fftsize), 0.0f);
        for (int i = 0; i < fftsize; ++i)
            x[static_cast<size_t>(i)] = float(A * std::sin(2.0 * M_PI * f0 * i / sr));

        for (const char* wname : {"square", "triangle", "hanning", "hamming", "blackman"}) {
            STFT stft;
            std::vector<float> mags(static_cast<size_t>(fftsize) / 2, 0.0f),
                              phases(static_cast<size_t>(fftsize) / 2, 0.0f);
            stft.process(x, mags, phases, fftsize, 1, wname, true, false);

            std::vector<NEW::mlmat::spectral_peak> frame_peaks;
            NEW::mlmat::spectral_find_peaks(mags, sr, fftsize, -60.0f, 4, frame_peaks);
            check(!frame_peaks.empty(), "peak correction: peak found", 0.0);
            if (!frame_peaks.empty()) {
                const double cg = NEW::mlmat::window_coherent_gain(fftsize, wname);
                const double pkamp = 2.0 * double(frame_peaks[0].amplitude)
                                   / (double(fftsize) * cg);
                check(std::fabs(pkamp - A) < 5e-3,
                      "corrected peak amplitude recovers the sinusoid", pkamp - A);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// spectral_mask (mlmat.spectral_mask's core)
// ---------------------------------------------------------------------------
static void run_spectral_mask()
{
    const size_t n = 5;
    const float ml[5] = {0.5f, 0.3f, 0.7f, 0.8f, 0.0f};
    const float pl[5] = {0.1f, 0.2f, 0.3f, 0.4f, 0.9f};
    const float mr[5] = {0.6f, 0.05f, 0.9f, 0.8f, 0.0f};
    const float pr[5] = {0.5f, 0.6f, 0.7f, 0.8f, 0.2f};
    float m[5], p[5];

    // mag_from: magnitude from the source, phase from the masker
    NEW::mlmat::spectral_mask(ml, pl, mr, pr, m, p, n,
                              NEW::mlmat::mask_mode::mag_from, -60.0f);
    bool ok = true;
    for (size_t k = 0; k < n; ++k)
        ok = ok && std::fabs(m[k] - ml[k]) < 1e-6 && std::fabs(p[k] - pr[k]) < 1e-6;
    check(ok, "mask mag_from: magnitude from left, phase from right");

    // phase_from: the reverse
    NEW::mlmat::spectral_mask(ml, pl, mr, pr, m, p, n,
                              NEW::mlmat::mask_mode::phase_from, -60.0f);
    ok = true;
    for (size_t k = 0; k < n; ++k)
        ok = ok && std::fabs(m[k] - mr[k]) < 1e-6 && std::fabs(p[k] - pl[k]) < 1e-6;
    check(ok, "mask phase_from: magnitude from right, phase from left");

    // hard: at -20 dB the threshold is 0.1 -- only bin 1 (masker 0.05)
    // is gated; the zeroed bin is magnitude and phase 0
    NEW::mlmat::spectral_mask(ml, pl, mr, pr, m, p, n,
                              NEW::mlmat::mask_mode::hard, -20.0f);
    ok = std::fabs(m[0] - 0.5f) < 1e-6 && std::fabs(p[0] - 0.1f) < 1e-6
      && m[1] == 0.0f && p[1] == 0.0f
      && std::fabs(m[2] - 0.7f) < 1e-6 && std::fabs(p[2] - 0.3f) < 1e-6
      && std::fabs(m[3] - 0.8f) < 1e-6
      && m[4] == 0.0f;
    check(ok, "mask hard: gates bins whose masker is below the threshold");

    // soft: the Wiener weight |r|^2 / (|l|^2 + |r|^2), phase from the
    // source; both-zero bins give zero (no 0/0)
    NEW::mlmat::spectral_mask(ml, pl, mr, pr, m, p, n,
                              NEW::mlmat::mask_mode::soft, -60.0f);
    ok = true;
    for (size_t k = 0; k < 4; ++k) {
        const double expect = double(ml[k]) * (double(mr[k]) * mr[k])
            / (double(ml[k]) * ml[k] + double(mr[k]) * mr[k]);
        ok = ok && std::fabs(double(m[k]) - expect) < 1e-5
                  && std::fabs(p[k] - pl[k]) < 1e-6;
    }
    ok = ok && m[4] == 0.0f;
    check(ok, "mask soft: Wiener weight, phase from the source");

    // subtract: positive differences kept, negative floored at zero,
    // phase from the source
    NEW::mlmat::spectral_mask(ml, pl, mr, pr, m, p, n,
                              NEW::mlmat::mask_mode::subtract, -60.0f);
    ok = m[0] == 0.0f                                   // 0.5 - 0.6 < 0
      && std::fabs(m[1] - 0.25f) < 1e-6 && std::fabs(p[1] - 0.2f) < 1e-6
      && m[2] == 0.0f                                   // 0.7 - 0.9 < 0
      && m[3] == 0.0f                                   // 0.8 - 0.8 == 0
      && m[4] == 0.0f;
    check(ok, "mask subtract: floors the difference at zero");
}

// ---------------------------------------------------------------------------
// spectral_op (mlmat.spectral_op's core)
// ---------------------------------------------------------------------------
static void run_spectral_op()
{
    const size_t n = 4;
    const float ml[4] = {0.5f, 0.3f, 0.0f, 0.8f};
    const float pl[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    const float mr[4] = {0.6f, 0.05f, 0.9f, 0.8f};
    const float pr[4] = {0.5f, 0.6f, 0.7f, 0.8f};
    float m[4], p[4];

    // mix endpoints: t=0 is the left input, t=1 the right. compared
    // as complex values: a zero-magnitude bin carries no meaningful
    // phase (the roundtrip yields atan2(0,0) = 0), but its complex
    // value (zero) is exact
    NEW::mlmat::spectral_op(ml, pl, mr, pr, m, p, n, NEW::mlmat::op_kind::mix, 0.0, 1.0f);
    bool ok = true;
    for (size_t k = 0; k < n; ++k)
        ok = ok && std::fabs(double(m[k]) * std::cos(p[k]) - double(ml[k]) * std::cos(pl[k])) < 1e-6
                  && std::fabs(double(m[k]) * std::sin(p[k]) - double(ml[k]) * std::sin(pl[k])) < 1e-6;
    check(ok, "op mix at t=0 is the left input");

    NEW::mlmat::spectral_op(ml, pl, mr, pr, m, p, n, NEW::mlmat::op_kind::mix, 1.0, 1.0f);
    ok = true;
    for (size_t k = 0; k < n; ++k)
        ok = ok && std::fabs(double(m[k]) * std::cos(p[k]) - double(mr[k]) * std::cos(pr[k])) < 1e-6
                  && std::fabs(double(m[k]) * std::sin(p[k]) - double(mr[k]) * std::sin(pr[k])) < 1e-6;
    check(ok, "op mix at t=1 is the right input");

    // max/min select the whole bin (magnitude and phase) of the
    // larger/smaller input; ties go to the left
    NEW::mlmat::spectral_op(ml, pl, mr, pr, m, p, n, NEW::mlmat::op_kind::max, 0.5, 1.0f);
    ok = std::fabs(m[0] - 0.6f) < 1e-6 && std::fabs(p[0] - 0.5f) < 1e-6
      && std::fabs(m[1] - 0.3f) < 1e-6 && std::fabs(p[1] - 0.2f) < 1e-6
      && std::fabs(m[2] - 0.9f) < 1e-6 && std::fabs(p[2] - 0.7f) < 1e-6
      && std::fabs(m[3] - 0.8f) < 1e-6 && std::fabs(p[3] - 0.4f) < 1e-6;
    check(ok, "op max selects the larger bin's full content");

    NEW::mlmat::spectral_op(ml, pl, mr, pr, m, p, n, NEW::mlmat::op_kind::min, 0.5, 1.0f);
    ok = std::fabs(m[0] - 0.5f) < 1e-6 && std::fabs(p[0] - 0.1f) < 1e-6
      && std::fabs(m[1] - 0.05f) < 1e-6 && std::fabs(p[1] - 0.6f) < 1e-6
      && std::fabs(m[2] - 0.0f) < 1e-6 && std::fabs(p[2] - 0.3f) < 1e-6
      && std::fabs(m[3] - 0.8f) < 1e-6 && std::fabs(p[3] - 0.4f) < 1e-6;
    check(ok, "op min selects the smaller bin's full content");

    // gain scales the output (max is magnitude-domain, so the mags
    // scale directly)
    NEW::mlmat::spectral_op(ml, pl, mr, pr, m, p, n, NEW::mlmat::op_kind::max, 0.5, 2.0f);
    ok = std::fabs(m[0] - 1.2f) < 1e-5 && std::fabs(m[2] - 1.8f) < 1e-5;
    check(ok, "op gain scales the output");

    // ------------------------------------------------------------------
    // add is linear in the STFT coefficients: the frame-wise complex
    // sum of two analyses must equal the analysis of the summed
    // signals, bin for bin
    // ------------------------------------------------------------------
    {
        const int fftsize = 512;
        const int overlap = 4;
        const size_t nsig = 2048;
        const size_t step = size_t(fftsize) / size_t(overlap);
        const size_t frames = (nsig + step - 1) / step;
        const size_t spec = size_t(fftsize) / 2;

        std::vector<float> x = make_signal(nsig, 777);
        std::vector<float> y = make_signal(nsig, 999);
        std::vector<float> xy(nsig);
        for (size_t i = 0; i < nsig; ++i)
            xy[i] = x[i] + y[i];

        STFT stft;
        std::vector<float> mx(frames * spec, 0.0f), px(frames * spec, 0.0f);
        std::vector<float> my(frames * spec, 0.0f), py(frames * spec, 0.0f);
        std::vector<float> mxy(frames * spec, 0.0f), pxy(frames * spec, 0.0f);
        stft.process(x, mx, px, fftsize, overlap, "hanning", true, false);
        stft.process(y, my, py, fftsize, overlap, "hanning", true, false);
        stft.process(xy, mxy, pxy, fftsize, overlap, "hanning", true, false);

        std::vector<float> ma(spec), pa(spec);
        double max_err = 0.0;
        for (size_t f = 0; f < frames; ++f) {
            NEW::mlmat::spectral_op(mx.data() + f * spec, px.data() + f * spec,
                                    my.data() + f * spec, py.data() + f * spec,
                                    ma.data(), pa.data(), spec,
                                    NEW::mlmat::op_kind::add, 0.0, 1.0f);
            for (size_t k = 0; k < spec; ++k) {
                const double re_add = double(ma[k]) * std::cos(pa[k]);
                const double im_add = double(ma[k]) * std::sin(pa[k]);
                const double re_ref = double(mxy[f * spec + k]) * std::cos(pxy[f * spec + k]);
                const double im_ref = double(mxy[f * spec + k]) * std::sin(pxy[f * spec + k]);
                max_err = std::max(max_err, std::fabs(re_add - re_ref));
                max_err = std::max(max_err, std::fabs(im_add - im_ref));
            }
        }
        check(max_err < 1e-2, "op add: stft(x) + stft(y) == stft(x + y)", max_err);

        // multiply must agree with spectral_multiply (the crosssynth
        // core) on the same frame, bin for bin
        {
            std::vector<float> re1(spec), im1(spec), re2(spec), im2(spec);
            for (size_t k = 0; k < spec; ++k) {
                re1[k] = float(mx[k] * std::cos(px[k]));
                im1[k] = float(mx[k] * std::sin(px[k]));
                re2[k] = float(my[k] * std::cos(py[k]));
                im2[k] = float(my[k] * std::sin(py[k]));
            }
            std::vector<float> re_p(spec), im_p(spec);
            NEW::mlmat::spectral_multiply(re1.data(), im1.data(), re2.data(), im2.data(),
                                          re_p.data(), im_p.data(), spec, false, 1.0f);
            NEW::mlmat::spectral_op(mx.data(), px.data(), my.data(), py.data(),
                                    ma.data(), pa.data(), spec,
                                    NEW::mlmat::op_kind::multiply, 0.0, 1.0f);
            double max_mul = 0.0;
            for (size_t k = 0; k < spec; ++k) {
                const double re_op = double(ma[k]) * std::cos(pa[k]);
                const double im_op = double(ma[k]) * std::sin(pa[k]);
                max_mul = std::max(max_mul, std::fabs(re_op - re_p[k]) / (1.0 + std::fabs(re_p[k])));
                max_mul = std::max(max_mul, std::fabs(im_op - im_p[k]) / (1.0 + std::fabs(im_p[k])));
            }
            check(max_mul < 1e-3, "op multiply matches spectral_multiply", max_mul);
        }
    }
}

// ---------------------------------------------------------------------------
// peak pruning (spectral_peaks' @spacing / @spacing_db core)
// ---------------------------------------------------------------------------
static void run_peak_pruning()
{
    const int fftsize = 64;
    const double sr = 48000.0;

    auto peak = [&](double bin, double amp) {
        NEW::mlmat::spectral_peak p;
        p.amplitude = static_cast<float>(amp);
        p.frequency = static_cast<float>(bin * sr / fftsize);
        return p;
    };
    auto prune = [&](std::vector<NEW::mlmat::spectral_peak>& peaks,
                     int spacing, double spacing_db) {
        NEW::mlmat::spectral_prune_peaks(peaks, fftsize, sr, spacing, spacing_db);
    };

    // spacing 1 (the default): never prunes, even a tiny neighbor
    {
        std::vector<NEW::mlmat::spectral_peak> peaks = { peak(10, 0.8), peak(11, 0.01) };
        prune(peaks, 1, -31.0);
        check(peaks.size() == 2, "pruning off at spacing 1");
    }

    // a side-lobe-level neighbor within spacing, below the gate: pruned
    // (0.01 / 0.8 = -38 dB < -31 dB)
    {
        std::vector<NEW::mlmat::spectral_peak> peaks = { peak(10, 0.8), peak(11, 0.01) };
        prune(peaks, 3, -31.0);
        check(peaks.size() == 1 && std::fabs(peaks[0].amplitude - 0.8f) < 1e-6,
              "side-lobe-level neighbor is pruned");
    }

    // a genuine partial within spacing but above the gate: survives
    // (0.4 / 0.8 = -6 dB > -31 dB)
    {
        std::vector<NEW::mlmat::spectral_peak> peaks = { peak(10, 0.8), peak(11, 0.4) };
        prune(peaks, 3, -31.0);
        check(peaks.size() == 2, "genuine nearby partial survives the gate");
    }

    // gate 0 (off): the distance rule alone prunes the weaker neighbor
    {
        std::vector<NEW::mlmat::spectral_peak> peaks = { peak(10, 0.8), peak(11, 0.4) };
        prune(peaks, 3, 0.0);
        check(peaks.size() == 1, "gate 0 prunes on distance alone");
    }

    // farther than spacing: survives every setting
    {
        std::vector<NEW::mlmat::spectral_peak> peaks = { peak(10, 0.8), peak(14, 0.01) };
        prune(peaks, 3, -31.0);
        bool ok = peaks.size() == 2;
        prune(peaks, 3, 0.0);
        ok = ok && peaks.size() == 2;
        check(ok, "peaks farther than spacing are untouched");
    }

    // unsorted input: the strongest survives even when it comes last
    {
        std::vector<NEW::mlmat::spectral_peak> peaks = { peak(11, 0.01), peak(10, 0.8) };
        prune(peaks, 3, -31.0);
        const double f10 = 10.0 * sr / fftsize;
        check(peaks.size() == 1 && std::fabs(peaks[0].frequency - f10) < 1e-3,
              "unsorted input: strongest peak is kept");
    }

    // the strongest peak itself is never pruned
    {
        std::vector<NEW::mlmat::spectral_peak> peaks = { peak(10, 0.8) };
        prune(peaks, 3, -31.0);
        check(peaks.size() == 1 && std::fabs(peaks[0].amplitude - 0.8f) < 1e-6,
              "the main peak is never pruned");
    }
}

// ---------------------------------------------------------------------------
// partial tracking (spectral_peaks' @track core, McAulay-Quatieri matching)
// ---------------------------------------------------------------------------
static void run_partial_tracking()
{
    const size_t slots = 4;
    const double delta = 20.0;   // Hz matching tolerance
    NEW::mlmat::peak_tracker tr;
    std::vector<NEW::mlmat::spectral_peak> out;

    auto pk = [](double f, double a) {
        return NEW::mlmat::spectral_peak{static_cast<float>(a), static_cast<float>(f)};
    };

    // a stable sinusoid stays in slot 0 for every frame
    {
        tr.reset();
        bool ok = true;
        for (int f = 0; f < 20; ++f) {
            tr.process({pk(440.0, 0.5)}, slots, delta, 0, out);
            ok = ok && out.size() == slots
                  && std::fabs(out[0].frequency - 440.0f) < 1e-6
                  && std::fabs(out[0].amplitude - 0.5f) < 1e-6
                  && out[1].amplitude == 0.0f;
        }
        check(ok, "tracking: a stable sinusoid holds slot 0");
    }

    // two partials hold slots 0 and 1 even when their amplitudes swap
    // (birth order is by amplitude, so whichever is stronger in frame
    // 0 takes slot 0 -- what matters is that the assignment holds)
    {
        tr.reset();
        bool ok = true;
        double slot0_freq = 0.0;
        for (int f = 0; f < 20; ++f) {
            const double a0 = (f % 2) ? 0.5 : 0.3;
            const double a1 = (f % 2) ? 0.3 : 0.5;
            tr.process({pk(440.0, a0), pk(880.0, a1)}, slots, delta, 0, out);
            const double f0 = out[0].frequency;
            const double f1 = out[1].frequency;
            const bool pair = (std::fabs(f0 - 440.0) < 1e-6 && std::fabs(f1 - 880.0) < 1e-6)
                          || (std::fabs(f0 - 880.0) < 1e-6 && std::fabs(f1 - 440.0) < 1e-6);
            ok = ok && pair && (f == 0 || std::fabs(f0 - slot0_freq) < 1e-6);
            slot0_freq = f0;
        }
        check(ok, "tracking: slots are amplitude-swap invariant");
    }

    // a partial appearing mid-stream is born into the next free slot
    {
        tr.reset();
        bool ok = true;
        for (int f = 0; f < 15; ++f) {
            if (f < 10) {
                tr.process({pk(440.0, 0.5)}, slots, delta, 0, out);
                ok = ok && out[1].amplitude == 0.0f;
            } else {
                tr.process({pk(440.0, 0.5), pk(880.0, 0.4)}, slots, delta, 0, out);
                ok = ok && std::fabs(out[1].frequency - 880.0f) < 1e-6;
            }
        }
        check(ok, "tracking: a new partial is born into the next free slot");
    }

    // hold 0: a vanishing partial dies immediately (zeros), and the
    // slot is free for a rebirth
    {
        tr.reset();
        tr.process({pk(440.0, 0.5), pk(880.0, 0.4)}, slots, delta, 0, out);
        tr.process({pk(440.0, 0.5)}, slots, delta, 0, out);
        bool ok = out[1].amplitude == 0.0f && out[1].frequency == 0.0f;
        tr.process({pk(440.0, 0.5), pk(880.0, 0.4)}, slots, delta, 0, out);
        ok = ok && std::fabs(out[1].frequency - 880.0f) < 1e-6;
        check(ok, "tracking: hold 0 dies immediately and rebirths into the slot");
    }

    // hold 2: a vanishing partial coasts at zero amplitude holding its
    // frequency, and reacquires the same slot (no rebirth churn)
    {
        tr.reset();
        tr.process({pk(440.0, 0.5), pk(880.0, 0.4)}, slots, delta, 2, out);
        bool ok = true;
        for (int f = 0; f < 2; ++f) {
            tr.process({pk(440.0, 0.5)}, slots, delta, 2, out);
            ok = ok && out[1].amplitude == 0.0f
                  && std::fabs(out[1].frequency - 880.0f) < 1e-6;
        }
        tr.process({pk(440.0, 0.5), pk(880.0, 0.4)}, slots, delta, 2, out);
        ok = ok && std::fabs(out[1].amplitude - 0.4f) < 1e-6
              && std::fabs(out[1].frequency - 880.0f) < 1e-6
              && out[2].amplitude == 0.0f && out[3].amplitude == 0.0f;
        check(ok, "tracking: hold 2 coasts and reacquires the same slot");
    }

    // after the coast is exhausted the partial dies and its slot is
    // reused for the next birth
    {
        tr.reset();
        tr.process({pk(440.0, 0.5), pk(880.0, 0.4)}, slots, delta, 1, out);
        tr.process({pk(440.0, 0.5)}, slots, delta, 1, out);   // coasting
        tr.process({pk(440.0, 0.5)}, slots, delta, 1, out);   // dies
        bool ok = out[1].amplitude == 0.0f && out[1].frequency == 0.0f;
        tr.process({pk(440.0, 0.5), pk(880.0, 0.4)}, slots, delta, 1, out);
        ok = ok && std::fabs(out[1].frequency - 880.0f) < 1e-6
              && out[2].amplitude == 0.0f;
        check(ok, "tracking: death after the coast frees the slot");
    }

    // matching is bounded by delta: a partial 30 Hz away is a birth,
    // not a match (the old partial dies in the same frame at hold 0,
    // after the new peak has birthed into the next slot)
    {
        tr.reset();
        tr.process({pk(440.0, 0.5)}, slots, delta, 0, out);
        tr.process({pk(470.0, 0.5)}, slots, delta, 0, out);
        bool ok = out[1].amplitude == 0.5f
                  && std::fabs(out[1].frequency - 470.0f) < 1e-6
                  && out[0].amplitude == 0.0f && out[0].frequency == 0.0f;
        check(ok, "tracking: beyond-delta peaks are births, not matches");
    }

    // more partials than slots: the extras are dropped, the slots stay full
    {
        tr.reset();
        tr.process({pk(100.0, 0.5), pk(200.0, 0.4), pk(300.0, 0.3)},
                   2, delta, 0, out);
        bool ok = out.size() == 2
                  && std::fabs(out[0].frequency - 100.0f) < 1e-6
                  && std::fabs(out[1].frequency - 200.0f) < 1e-6;
        check(ok, "tracking: extra partials are dropped beyond max slots");
    }

    // reset() empties the state: the next partial starts over
    {
        tr.reset();
        tr.process({pk(440.0, 0.5), pk(880.0, 0.4)}, slots, delta, 0, out);
        tr.reset();
        tr.process({pk(1320.0, 0.5)}, slots, delta, 0, out);
        check(std::fabs(out[0].frequency - 1320.0f) < 1e-6,
              "tracking: reset() restarts slot assignment");
    }

    // end to end: two steady sinusoids through the STFT stack, found
    // per frame and tracked -- the columns hold the same partials for
    // the whole analysis
    {
        const int fftsize = 1024;
        const int overlap = 4;
        const double sr = 48000.0;
        const size_t step = size_t(fftsize) / size_t(overlap);
        const size_t n = 4096;
        const size_t frames = (n + step - 1) / step;
        const size_t spec = size_t(fftsize) / 2;

        std::vector<float> x(n);
        for (size_t i = 0; i < n; ++i)
            x[i] = float(0.5 * std::sin(2.0 * M_PI * 1000.0 * i / sr)
                       + 0.5 * std::sin(2.0 * M_PI * 1500.0 * i / sr));

        STFT stft;
        std::vector<float> mags(frames * spec, 0.0f), phases(frames * spec, 0.0f);
        stft.process(x, mags, phases, fftsize, overlap, "hanning", true, false);

        NEW::mlmat::peak_tracker tracker;
        std::vector<NEW::mlmat::spectral_peak> frame_peaks, tracked_out;
        bool ok = true;
        // the STFT fills floor((n-1)/step)+1 frames; the over-allocated
        // last frame of the output vectors stays all-zero
        const size_t filled = (n - 1) / step + 1;
        for (size_t f = 0; f < filled; ++f) {
            std::vector<float> frame(mags.begin() + f * spec,
                                     mags.begin() + (f + 1) * spec);
            // the two strongest peaks of every frame are the two
            // sinusoids (their side-lobes are far weaker); track into
            // exactly two slots
            NEW::mlmat::spectral_find_peaks(frame, sr, fftsize, -60.0f, 2, frame_peaks);
            tracker.process(frame_peaks, 2, sr / fftsize, 0, tracked_out);

            // the first two frames establish the partials; every later
            // frame must keep the two sinusoids in the same slots
            // (either assignment order -- which one leads depends on
            // the bin-centeredness of each sinusoid)
            // assert only on full-window frames: the parabolic
            // interpolation carries a few Hz of bias for the off-center
            // 1 kHz sinusoid (bin 21.33), and frames whose window
            // extends past the signal end (zero-padded) smear much
            // more -- the tracker still holds the slots there
            if (f >= 2 && f * step + size_t(fftsize) <= n) {
                const bool s0_1k = std::fabs(tracked_out[0].frequency - 1000.0) < 10.0;
                const bool s0_1k5 = std::fabs(tracked_out[0].frequency - 1500.0) < 10.0;
                const bool s1_1k = std::fabs(tracked_out[1].frequency - 1000.0) < 10.0;
                const bool s1_1k5 = std::fabs(tracked_out[1].frequency - 1500.0) < 10.0;
                ok = ok && ((s0_1k && s1_1k5) || (s0_1k5 && s1_1k))
                          && tracked_out[0].amplitude > 0.1
                          && tracked_out[1].amplitude > 0.1;
            }
        }
        check(ok, "tracking end to end: two sinusoids hold stable columns");
    }
}

// ---------------------------------------------------------------------------
// MQ synthesis (mlmat.buffer_mqsynth's core)
// ---------------------------------------------------------------------------
static void run_mq_synth()
{
    const int fftsize = 512;
    const int overlap = 4;
    const double sr = 48000.0;
    const size_t step = size_t(fftsize) / size_t(overlap);
    const size_t n = 8192;
    const size_t frames = (n - 1) / step + 1;
    const size_t bins = size_t(fftsize) / 2;
    const size_t slots = 4;

    // the objects' pipeline for one frame: find peaks, prune the
    // window side-lobes (spacing 9, distance-only gate), amplitude-
    // correct with the hanning coherent gain, track into identity
    // columns, and emit the (amplitude, frequency, phase) partial
    // data that the synth consumes
    NEW::mlmat::peak_tracker tracker;
    std::vector<NEW::mlmat::spectral_peak> frame_peaks, tracked;
    std::vector<float> pk_amps(frames * slots, 0.0f), pk_freqs(frames * slots, 0.0f);
    std::vector<float> pk_phases(frames * slots, 0.0f);
    const double cg = NEW::mlmat::window_coherent_gain(fftsize, "hanning");
    auto analyze_frame = [&](size_t f, const std::vector<float>& mags,
                             const std::vector<float>& phases) {
        frame_peaks.clear();
        std::vector<float> frame(mags.begin() + f * bins, mags.begin() + (f + 1) * bins);
        NEW::mlmat::spectral_find_peaks(frame, sr, fftsize, -60.0f, 8, frame_peaks);
        NEW::mlmat::spectral_prune_peaks(frame_peaks, fftsize, sr, 9, 0.0);
        for (auto& pk : frame_peaks)
            pk.amplitude = static_cast<float>(double(pk.amplitude) * 2.0 / (double(fftsize) * cg));
        tracker.process(frame_peaks, slots, sr / fftsize, 0, tracked);
        for (size_t s = 0; s < slots; ++s) {
            const float a = tracked[s].amplitude;
            const float fr = tracked[s].frequency;
            pk_amps[f * slots + s] = a;
            pk_freqs[f * slots + s] = fr;
            // the analysis phase at the partial's bin (plane 2)
            if (a == 0.0f && fr == 0.0f) {
                pk_phases[f * slots + s] = 0.0f;
            } else {
                long k = std::lround(double(fr) * double(fftsize) / sr);
                k = std::max(0L, std::min(k, long(bins) - 1));
                pk_phases[f * slots + s] = phases[f * bins + size_t(k)];
            }
        }
    };
    auto reset_pipeline = [&]() {
        tracker.reset();
        std::fill(pk_amps.begin(), pk_amps.end(), 0.0f);
        std::fill(pk_freqs.begin(), pk_freqs.end(), 0.0f);
        std::fill(pk_phases.begin(), pk_phases.end(), 0.0f);
    };

    // -------- identity: one steady bin-centered sinusoid reconstructed
    // near-sample-exact (the cubic phase makes this hold) --------
    {
        const double A = 0.5;
        const double f0 = 20.0 * sr / fftsize;   // bin 20 exactly
        std::vector<float> x(n);
        for (size_t i = 0; i < n; ++i)
            x[i] = float(A * std::sin(2.0 * M_PI * f0 * double(i) / sr));

        STFT stft;
        std::vector<float> mags(frames * bins, 0.0f), phases(frames * bins, 0.0f);
        stft.process(x, mags, phases, fftsize, overlap, "hanning", true, false);

        reset_pipeline();
        for (size_t f = 0; f < frames; ++f)
            analyze_frame(f, mags, phases);

        NEW::MQSynth mq;
        std::vector<float> out;
        mq.process(pk_amps, pk_freqs, pk_phases, frames, slots, fftsize, overlap, sr, out);

        check(out.size() == frames * step, "mq synth output length",
              double(out.size()) - double(frames * step));
        double max_err = 0.0;
        for (size_t i = 2 * step; i < (frames - 5) * step; ++i)
            max_err = std::max(max_err, std::fabs(double(out[i]) - double(x[i])));
        check(max_err < 5e-3, "mq synth identity: steady sinusoid reconstructed", max_err);
    }

    // -------- identity: two partials --------
    {
        const double A0 = 0.4, A1 = 0.3;
        const double f0 = 5.0 * sr / fftsize;    // bin 5
        const double f1 = 20.0 * sr / fftsize;   // bin 20
        std::vector<float> x(n);
        for (size_t i = 0; i < n; ++i)
            x[i] = float(A0 * std::sin(2.0 * M_PI * f0 * double(i) / sr)
                       + A1 * std::sin(2.0 * M_PI * f1 * double(i) / sr));

        STFT stft;
        std::vector<float> mags(frames * bins, 0.0f), phases(frames * bins, 0.0f);
        stft.process(x, mags, phases, fftsize, overlap, "hanning", true, false);

        reset_pipeline();
        for (size_t f = 0; f < frames; ++f)
            analyze_frame(f, mags, phases);

        NEW::MQSynth mq;
        std::vector<float> out;
        mq.process(pk_amps, pk_freqs, pk_phases, frames, slots, fftsize, overlap, sr, out);

        double max_err = 0.0;
        for (size_t i = 2 * step; i < (frames - 5) * step; ++i)
            max_err = std::max(max_err, std::fabs(double(out[i]) - double(x[i])));
        check(max_err < 5e-3, "mq synth identity: two partials reconstructed", max_err);
    }

    // -------- death envelope (direct core): the partial fades out
    // over the frame after its last frame, then the output is silent
    // -- and the run interior is phase-exact --------
    {
        const size_t t_frames = 10;
        const double f0 = 20.0 * sr / fftsize;
        std::vector<float> amps(t_frames, 0.0f), freqs(t_frames, 0.0f), phases(t_frames, 0.0f);
        for (size_t f = 0; f < 6; ++f) {
            amps[f] = 0.5f;
            freqs[f] = static_cast<float>(f0);
            // consistent analysis phases: 2*pi*f0*f*hop/sr
            phases[f] = float(std::fmod(2.0 * M_PI * f0 * double(f) * step / sr, 2.0 * M_PI));
        }

        NEW::MQSynth mq;
        std::vector<float> out;
        mq.process(amps, freqs, phases, t_frames, 1, fftsize, overlap, sr, out);

        check(out.size() == t_frames * step, "mq synth death: output length",
              double(out.size()) - double(t_frames * step));
        // the run interior matches the pure sinusoid
        double max_err = 0.0;
        for (size_t i = step; i < 5 * step; ++i)
            max_err = std::max(max_err, std::fabs(double(out[i])
                - 0.5 * std::cos(2.0 * M_PI * f0 * double(i) / sr)));
        check(max_err < 1e-2, "mq synth death: run interior is phase-exact", max_err);
        // after the fade frame, silence
        double tail = 0.0;
        for (size_t i = 6 * step; i < out.size(); ++i)
            tail = std::max(tail, std::fabs(double(out[i])));
        check(tail == 0.0, "mq synth death: silence after the fade", tail);
    }

    // -------- coast gap (direct core): zero-amplitude coast frames
    // with the frequency held fade down and back up within one run
    // (no death, no rebirth) --------
    {
        const size_t t_frames = 10;
        const double f0 = 20.0 * sr / fftsize;
        std::vector<float> amps(t_frames, 0.0f), freqs(t_frames, 0.0f), phases(t_frames, 0.0f);
        for (size_t f = 0; f < t_frames; ++f) {
            freqs[f] = static_cast<float>(f0);   // held through the gap
            phases[f] = float(std::fmod(2.0 * M_PI * f0 * double(f) * step / sr, 2.0 * M_PI));
        }
        for (size_t f = 0; f < 4; ++f) amps[f] = 0.5f;
        for (size_t f = 6; f < t_frames; ++f) amps[f] = 0.5f;
        // frames 4 and 5 coast at zero amplitude

        NEW::MQSynth mq;
        std::vector<float> out;
        mq.process(amps, freqs, phases, t_frames, 1, fftsize, overlap, sr, out);

        // the gap center is near silent (the fades span the whole
        // frame, so only the middle of the gap is at zero)
        double gap_peak = 0.0;
        for (size_t i = 5 * step - 8; i < 5 * step + 8; ++i)
            gap_peak = std::max(gap_peak, std::fabs(double(out[i])));
        check(gap_peak < 0.05, "mq synth coast: gap fades to silence", gap_peak);
        // ... and the partial recovers full amplitude after the gap
        double end_peak = 0.0;
        for (size_t i = 7 * step; i + 1 < out.size(); ++i)
            end_peak = std::max(end_peak, std::fabs(double(out[i])));
        check(end_peak > 0.4, "mq synth coast: partial recovers after the gap", end_peak);
    }
}

// ---------------------------------------------------------------------------
// Griffin-Lim input seeding: an input seeded with the true analysis
// phases is already consistent, so a single iteration reproduces the
// istft result (this is the warm start a rectangular input gives via
// the hypot/atan2 conversion)
// ---------------------------------------------------------------------------
static void run_gl_seeding()
{
    const int fftsize = 512;
    const int overlap = 4;
    const size_t n = 4096;
    const size_t step = size_t(fftsize) / size_t(overlap);
    const size_t frames = (n + step - 1) / step;
    const size_t spec = size_t(fftsize) / 2;
    const size_t total = frames * spec;

    auto signal = make_signal(n, 4242);

    STFT stft;
    ISTFT istft;
    std::vector<float> mags(total, 0.0f), phases(total, 0.0f);
    stft.process(signal, mags, phases, fftsize, overlap, "hanning", true, false);

    // the plain istft of the analysis is the reference. NOTE: istft's
    // polar -> rect conversion happens in place on the caller's
    // vectors, so it gets its own copies (see the window-caching test
    // for the same trap)
    std::vector<float> ref_mags = mags, ref_phases = phases;
    std::vector<float> ref(total + step, 0.0f);
    istft.process(ref_mags, ref_phases, ref, fftsize, overlap, "hanning", true, false);

    // Griffin-Lim seeded with the true phases: one iteration
    GriffinLim gl;
    std::vector<float> in_mags = mags, in_phases = phases;
    std::vector<float> out(total + step, 0.0f);
    gl.process(in_mags, in_phases, total, out, fftsize, overlap, "hanning", 1, 0.0, false);

    double max_diff = 0.0;
    for (size_t i = fftsize; i + fftsize < n; ++i)
        max_diff = std::max(max_diff, std::fabs(double(out[i]) - double(ref[i])));
    check(max_diff < 1e-2, "griffin-lim seeded with true phases matches the istft", max_diff);
}

int main()
{
    run_parity();
    run_resample();
    run_spectral_ops();
    run_reassignment();
    run_amplitude_correction();
    run_spectral_mask();
    run_spectral_op();
    run_peak_pruning();
    run_partial_tracking();
    run_mq_synth();
    run_gl_seeding();
    printf("spectral parity: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
