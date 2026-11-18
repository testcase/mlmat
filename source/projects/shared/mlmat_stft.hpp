/// @file mlmat_stft.hpp
/// @ingroup mlmat
/// @copyright Copyright 2026 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_min.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "bqfft/FFT.h"

// the MIN_DISCUSSION macro (consumed by the reference page generators
// via the has_class_discussion trait). Also defined in mlmat.hpp; the
// guard makes this header usable without the mlpack-heavy includes.
#ifndef MIN_DISCUSSION
#define MIN_DISCUSSION static constexpr const char* class_discussion
#endif

using namespace c74;
using namespace c74::min;

namespace mlmat {

/// bqfft's vDSP (macOS) and built-in (Windows) backends both support
/// power-of-two transform sizes only.
inline bool fftsize_is_valid(int fftsize) {
    return fftsize >= 2 && (fftsize & (fftsize - 1)) == 0;
}

/// Window generation. These reproduce the conventions of the vDSP window
/// generators used by earlier versions of this code:
///   hanning  : vDSP_hann_window(..., vDSP_HANN_NORM)  == sqrt(8/3) * periodic Hann
///   hamming  : vDSP_hamm_window(..., 0)               == periodic Hamming
///   blackman : vDSP_blkman_window(..., 0)            == periodic Blackman
///   triangle : the vDSP_vramp construction used previously
inline void make_window(std::vector<float>& window, int fftsize, const string& window_string) {
    const int fftsize_2 = fftsize / 2;

    if (window_string == "triangle") {
        const float tristart = 0.0f;
        const float tritop = 1.0f;
        const float up = 1.0f / fftsize_2;
        const float down = -1.0f / fftsize_2;
        // Up ramp across the first half (vDSP_vramp semantics).
        for (int i = 0; i < fftsize_2; ++i) {
            window[i] = tristart + up * i;
        }
        // Down ramp across the second half, starting one sample earlier so
        // that the center sample holds the peak value (as before).
        for (int i = 0; i < fftsize_2; ++i) {
            window[fftsize_2 - 1 + i] = tritop + down * i;
        }
    } else if (window_string == "hanning") {
        // vDSP_HANN_NORM is the periodic Hann window scaled by sqrt(8/3).
        const float scale = 0.5f * 1.6329931618554521f;
        for (int i = 0; i < fftsize; ++i) {
            window[i] = scale * (1.0f - std::cos((2.0f * static_cast<float>(M_PI) * i) / fftsize));
        }
    } else if (window_string == "hamming") {
        for (int i = 0; i < fftsize; ++i) {
            window[i] = 0.54f - 0.46f * std::cos((2.0f * static_cast<float>(M_PI) * i) / fftsize);
        }
    } else if (window_string == "blackman") {
        for (int i = 0; i < fftsize; ++i) {
            const float a = (2.0f * static_cast<float>(M_PI) * i) / fftsize;
            window[i] = 0.42f - 0.5f * std::cos(a) + 0.08f * std::cos(2.0f * a);
        }
    }
}

/// Coherent gain (mean amplitude) of an analysis window: the factor by
/// which windowing scales the DFT magnitude of a bin-centered
/// sinusoid. The unscaled forward DFT (bqfft's convention) puts a
/// sinusoid of amplitude A at A * fftsize * cg / 2 in the complex bins,
/// so dividing by that recovers true signal amplitudes; the real DC (and
/// Nyquist) bin holds all its energy in one bin, so it is A * fftsize *
/// cg there (no factor 2). "square" is the implicit all-ones window
/// (make_window leaves it empty, the STFT skips the multiply).
///
///   square   1.0     triangle 0.5
///   hanning  sqrt(8/3) / 2 (the sqrt(8/3) scale is the vDSP_HANN_NORM
///                        convention reproduced by make_window)
///   hamming  0.54    blackman 0.42
inline double window_coherent_gain(int fftsize, const string& window_string)
{
    if (window_string == "square") {
        return 1.0;
    }
    std::vector<float> window(static_cast<size_t>(fftsize), 0.0f);
    make_window(window, fftsize, window_string);
    double sum = 0.0;
    for (float w : window) {
        sum += static_cast<double>(w);
    }
    return sum / static_cast<double>(fftsize);
}

/// Cubic Hermite (Catmull-Rom) resampling of `input` (sampled at
/// `input_rate`) to `output_rate`. The output vector is resized to
/// round(input.size() * output_rate / input_rate) and holds the signal
/// resampled onto the output-rate sample grid.
inline void resample(const std::vector<float>& input,
                     std::vector<float>& output,
                     double input_rate,
                     double output_rate)
{
    if (input.empty() || input_rate <= 0.0 || output_rate <= 0.0) {
        output.clear();
        return;
    }
    const size_t in_size = input.size();
    const size_t out_size = static_cast<size_t>(
        std::llround(static_cast<double>(in_size) * output_rate / input_rate));
    if (out_size == 0) {
        output.clear();
        return;
    }
    output.resize(out_size);
    const double step = static_cast<double>(in_size) / static_cast<double>(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        const double pos = std::min(static_cast<double>(i) * step,
                                    static_cast<double>(in_size - 1));
        const size_t p = static_cast<size_t>(pos);
        const double t = pos - static_cast<double>(p);
        const float p0 = input[p > 0 ? p - 1 : 0];
        const float p1 = input[p];
        const float p2 = input[p + 1 < in_size ? p + 1 : in_size - 1];
        const float p3 = input[p + 2 < in_size ? p + 2 : in_size - 1];
        output[i] = static_cast<float>(0.5 * ((2.0 * p1)
            + (-static_cast<double>(p0) + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
            + (-static_cast<double>(p0) + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t));
    }
}

/// dB to linear amplitude.
inline float db_to_lin(float db)
{
    return static_cast<float>(std::pow(10.0, static_cast<double>(db) / 20.0));
}

/// One spectral peak: an amplitude with an (interpolated) frequency.
struct spectral_peak {
    float amplitude;
    float frequency;
};

/// Center frequency of bin k (k may be fractional: an interpolated
/// peak position).
inline double bin_frequency(double k, double samplerate, int fftsize)
{
    return k * samplerate / fftsize;
}

/// Find the spectral peaks of one frame of magnitudes: bins that are
/// local maxima above `threshold_db`, refined with parabolic
/// interpolation over the (log) magnitudes of the neighboring bins.
/// The result holds at most `max_peaks` peaks, sorted by descending
/// amplitude. Edge bins (0 and bins-1) cannot be interpolated.
inline void spectral_find_peaks(const std::vector<float>& magnitudes,
                                double samplerate,
                                int fftsize,
                                float threshold_db,
                                size_t max_peaks,
                                std::vector<spectral_peak>& out)
{
    const size_t bins = magnitudes.size();
    const float thr = db_to_lin(threshold_db);
    out.clear();
    for (size_t k = 0; k < bins; ++k) {
        const float m = magnitudes[k];
        if (m < thr) {
            continue;
        }
        const float left = k > 0 ? magnitudes[k - 1] : m;
        const float right = k + 1 < bins ? magnitudes[k + 1] : m;
        if (m < left || m < right) {
            continue; // not a local maximum
        }

        float delta = 0.0f;
        float amp = m;
        if (k > 0 && k + 1 < bins) {
            const double la = std::log(static_cast<double>(magnitudes[k - 1]));
            const double lb = std::log(static_cast<double>(m));
            const double lc = std::log(static_cast<double>(magnitudes[k + 1]));
            const double denom = la - 2.0 * lb + lc;
            if (std::fabs(denom) > 1e-12) {
                delta = static_cast<float>(0.5 * (la - lc) / denom);
                if (delta > 0.5f) {
                    delta = 0.5f;
                }
                if (delta < -0.5f) {
                    delta = -0.5f;
                }
                amp = static_cast<float>(std::exp(lb - 0.25 * (la - lc) * delta));
            }
        }

        spectral_peak p;
        p.amplitude = amp;
        p.frequency = static_cast<float>(bin_frequency(static_cast<long>(k) + delta,
                                                      samplerate, fftsize));
        out.push_back(p);
    }

    std::sort(out.begin(), out.end(), [](const spectral_peak& a, const spectral_peak& b) {
        return a.amplitude > b.amplitude;
    });
    if (out.size() > max_peaks) {
        out.resize(max_peaks);
    }
}

/// Phase-based (reassigned) instantaneous frequency for bin k.
/// prev_phase and curr_phase are the phases of the same bin in two
/// consecutive STFT frames; overlap is the STFT overlap factor (so the
/// hop is fftsize / overlap). Returns the precise frequency in Hz.
///
/// For a stable sinusoid this gives the exact frequency (not an
/// approximation like the parabolic interpolation); for transient or
/// broadband content it degrades gracefully.
inline double reassigned_frequency(long k, double curr_phase, double prev_phase,
                                   int fftsize, int overlap, double samplerate)
{
    const double hop = double(fftsize) / double(overlap);
    if (hop < 1.0) {
        return -1.0;
    }

    // expected phase advance for the bin center frequency
    const double expected = 2.0 * M_PI * double(k) * hop / double(fftsize);

    // actual phase difference minus the expected advance
    double dphi = curr_phase - prev_phase - expected;

    // unwrap to [-pi, pi]
    dphi = dphi - 2.0 * M_PI * std::round(dphi / (2.0 * M_PI));

    // frequency deviation from bin center, in bins
    const double delta = dphi * double(fftsize) / (2.0 * M_PI * hop);

    return (double(k) + delta) * samplerate / double(fftsize);
}

/// Frame-wise multiply of two spectra (spectral convolution). re/im
/// hold `bins` values each.
///
/// For the half-spectrum convention (bins == fftsize/2, as produced by
/// the buffer_stft / buffer_istft stack) index 0 packs the real DC in
/// re and the real Nyquist in im (the vDSP-era convention), so the two
/// multiply as real x real. For the full spectrum all bins are complex
/// pairs.
inline void spectral_multiply(const float* re1, const float* im1,
                              const float* re2, const float* im2,
                              float* re_out, float* im_out,
                              size_t bins,
                              bool full_spectrum,
                              float gain)
{
    for (size_t k = 1; k < bins; ++k) {
        re_out[k] = (re1[k] * re2[k] - im1[k] * im2[k]) * gain;
        im_out[k] = (re1[k] * im2[k] + im1[k] * re2[k]) * gain;
    }
    if (full_spectrum) {
        re_out[0] = (re1[0] * re2[0] - im1[0] * im2[0]) * gain;
        im_out[0] = (re1[0] * im2[0] + im1[0] * re2[0]) * gain;
    }
    else {
        re_out[0] = re1[0] * re2[0] * gain;
        im_out[0] = im1[0] * im2[0] * gain;
    }
}

} // namespace mlmat


class STFT
{

public:
    STFT() = default;
    ~STFT() = default;

    void process(std::vector<float>& input,
                 std::vector<float>& out_magnitudes,
                 std::vector<float>& out_phases,
                 int fftsize,
                 int overlap,
                 const string window_string,
                 bool output_polar = true,
                 bool full_spectrum = false)
    {
        if (!mlmat::fftsize_is_valid(fftsize)) {
            std::stringstream ss;
            ss << "fftsize " << fftsize << " is not valid.";
            throw std::invalid_argument(ss.str());
        }

        // (Re)create the FFT setup when the size changes.
        if (!m_fft || m_fft_size != fftsize) {
            try {
                m_fft.reset(new breakfastquay::FFT(fftsize));
                m_fft_size = fftsize;
            } catch (breakfastquay::FFT::Exception) {
                std::stringstream ss;
                ss << "fftsize " << fftsize << " is not valid.";
                throw std::invalid_argument(ss.str());
            }
        }

        size_t step = fftsize/overlap;
        size_t fftsize_2 = fftsize / 2;
        size_t frame_count = input.size();

        m_in_samples.resize(fftsize);
        m_spec_real.resize(fftsize_2 + 1);
        m_spec_imag.resize(fftsize_2 + 1, 0.0f);
        m_out_real.resize(fftsize_2);
        m_out_imag.resize(fftsize_2);
        m_window_vector.resize(fftsize);
        m_magnitudes.resize(fftsize_2);
        m_phases.resize(fftsize_2);

        // the window only depends on (fftsize, window type): rebuild it
        // only when either changes, not on every call
        if (m_window_fftsize != fftsize || m_window_type != window_string) {
            mlmat::make_window(m_window_vector, fftsize, window_string);
            m_window_fftsize = fftsize;
            m_window_type = window_string;
        }

        int current_pos = 0;

        for (auto i = 0; i < frame_count; i+=step) {
            //get next N samples
            if(i+fftsize > frame_count) {
                for(auto j=0;j<fftsize;j++) {
                    if((i+j) >= frame_count) {
                        m_in_samples[j] = 0.0f;
                    } else {
                        m_in_samples[j] = input[(i+j)];
                    }
                }
            }
            else {
                std::copy_n(input.begin()+i, fftsize, m_in_samples.begin());
            }
            //window
            if (window_string != "square") {
                for(auto j=0;j<fftsize;j++) {
                    m_in_samples[j] *= m_window_vector[j];
                }
            }

            // Real FFT: fftsize real samples in, fftsize/2+1 complex
            // bins out (DC through Nyquist).
            m_fft->forward(m_in_samples.data(), m_spec_real.data(), m_spec_imag.data());

            // Keep bins 0 .. fftsize/2-1, matching the output of the previous
            // vDSP implementation. bqfft's forward transform already returns
            // the unscaled DFT (the old vDSP zrop result was 2x the DFT, which
            // the previous code compensated for with a 0.5 factor); the
            // Nyquist bin is discarded as before.
            std::copy_n(m_spec_real.begin(), fftsize_2, m_out_real.begin());
            std::copy_n(m_spec_imag.begin(), fftsize_2, m_out_imag.begin());
            m_out_imag[0] = 0.0f;

            if(output_polar) {
                for(size_t j=0;j<fftsize_2;j++) {
                    m_magnitudes[j] = std::sqrt(m_out_real[j]*m_out_real[j] + m_out_imag[j]*m_out_imag[j]);
                    m_phases[j] = std::atan2(m_out_imag[j], m_out_real[j]);
                }
                std::copy(m_magnitudes.begin(), m_magnitudes.end(), out_magnitudes.begin()+current_pos);
                std::copy(m_phases.begin(), m_phases.end(), out_phases.begin()+current_pos);
                current_pos += fftsize_2;

                if(full_spectrum) {
                    std::copy(m_magnitudes.rbegin(), m_magnitudes.rend(), out_magnitudes.begin()+current_pos);
                    std::copy(m_phases.rbegin(), m_phases.rend(), out_phases.begin()+current_pos);
                    current_pos += fftsize_2;
                }

            } else {
                // output rectangular data. vectors are magnitudes and phases in this case but that is misleading
                std::copy(m_out_real.begin(), m_out_real.end(), out_magnitudes.begin()+current_pos);
                std::copy(m_out_imag.begin(), m_out_imag.end(), out_phases.begin()+current_pos);
                current_pos += fftsize_2;
                if(full_spectrum) {
                    std::copy(m_out_real.rbegin(), m_out_real.rend(), out_magnitudes.begin()+current_pos);
                    std::copy(m_out_imag.rbegin(), m_out_imag.rend(), out_phases.begin()+current_pos);
                    current_pos += fftsize_2;
                }
            }
        }

    }

private:
    std::unique_ptr<breakfastquay::FFT> m_fft;
    int m_fft_size = 0;
    std::vector<float> m_in_samples;
    std::vector<float> m_spec_real;
    std::vector<float> m_spec_imag;
    std::vector<float> m_out_real;
    std::vector<float> m_out_imag;
    std::vector<float> m_window_vector;
    int m_window_fftsize = 0;
    string m_window_type;
    std::vector<float> m_magnitudes;
    std::vector<float> m_phases;
};



class ISTFT {
public:
    ISTFT() = default;
    ~ISTFT() = default;

    void process(std::vector<float>& in_real,
                 std::vector<float>& in_imag,
                 std::vector<float>& samples,
                 int fftsize,
                 int overlap,
                 const string window_string,
                 bool input_polar = true,
                 bool full_spectrum = false)
    {
        if (!mlmat::fftsize_is_valid(fftsize)) {
            std::stringstream ss;
            ss << "fftsize " << fftsize << " is not valid.";
            throw std::invalid_argument(ss.str());
        }

        if (!m_fft || m_fft_size != fftsize) {
            try {
                m_fft.reset(new breakfastquay::FFT(fftsize));
                m_fft_size = fftsize;
            } catch (breakfastquay::FFT::Exception) {
                std::stringstream ss;
                ss << "fftsize " << fftsize << " is not valid.";
                throw std::invalid_argument(ss.str());
            }
        }

        if(input_polar) {
            for(auto i=0;i<in_real.size();++i) {
                const float r = in_real[i];
                const float theta = in_imag[i];
                in_real[i] = r * std::cos(theta);
                in_imag[i] = r * std::sin(theta);
            }
        }

        size_t step = fftsize / overlap;
        size_t fftsize_2 = fftsize / 2;

        m_spec_real.resize(fftsize_2 + 1, 0.0f);
        m_spec_imag.resize(fftsize_2 + 1, 0.0f);

        std::fill(samples.begin(), samples.end(), 0.0f);

        m_signal.resize(fftsize);
        m_window_vector.resize(fftsize);

        // the window only depends on (fftsize, window type): rebuild it
        // only when either changes, not on every call
        if (m_window_fftsize != fftsize || m_window_type != window_string) {
            mlmat::make_window(m_window_vector, fftsize, window_string);
            m_window_fftsize = fftsize;
            m_window_type = window_string;
        }

        float scaling = 1.0f / fftsize / overlap;

        // iterate over complete frames only: as many as fit both in the
        // output buffer and in the provided input. (the output buffer can
        // imply more frames than the input covers; reading those would run
        // past the end of the input vectors)
        const size_t out_frames = (samples.size() + fftsize_2 - 1) / fftsize_2;
        const size_t in_frames = std::min(in_real.size(), in_imag.size()) / fftsize_2;
        const size_t frames = std::min(out_frames, in_frames);

        for (size_t frame = 0; frame < frames; ++frame) {
            const size_t i = frame * fftsize_2;
            const size_t j = frame * step;

            // Build the fftsize/2+1 bin spectrum expected by bqfft from the
            // fftsize/2 bins provided, matching the packing used by the
            // previous vDSP implementation: the imaginary part of bin 0
            // held the real part of the (discarded) Nyquist bin.
            for(size_t k=0;k<fftsize_2;k++) {
                m_spec_real[k] = in_real[i+k];
                m_spec_imag[k] = in_imag[i+k];
            }
            m_spec_imag[0] = 0.0f;
            m_spec_real[fftsize_2] = in_imag[i];
            m_spec_imag[fftsize_2] = 0.0f;

            // Inverse real FFT: fftsize/2+1 complex bins in, fftsize real
            // samples out.
            m_fft->inverse(m_spec_real.data(), m_spec_imag.data(), m_signal.data());

            for(auto k=0;k<fftsize;k++) {
                m_signal[k] *= scaling;
            }

            if (window_string != "square") {
                for(auto k=0;k<fftsize;k++) {
                    m_signal[k] *= m_window_vector[k];
                }
            }

            // Overlap-add. The caller sizes the buffer like the previous
            // implementation did, which can be a few samples short of the
            // last frame; stay in bounds.
            const size_t samples_size = samples.size();
            for(auto k=0;k<fftsize && (j+k) < samples_size;k++) {
                samples[j+k] += m_signal[k];
            }
        }
    }

private:
    std::unique_ptr<breakfastquay::FFT> m_fft;
    int m_fft_size = 0;

    std::vector<float> m_spec_real;
    std::vector<float> m_spec_imag;
    std::vector<float> m_signal;
    std::vector<float> m_window_vector;
    int m_window_fftsize = 0;
    string m_window_type;
};


/// Griffin-Lim phase reconstruction (as used by mlmat.buffer_griffinlim).
///
/// Each iteration synthesizes a time-domain signal from the current complex
/// estimate (magnitudes from the input, phases under refinement), analyzes
/// it again, and forms a new phase estimate from the consistent analysis,
/// restoring the original magnitudes. The update uses the fast Griffin-Lim
/// momentum formulation (Perraudin et al. 2014):
///
///   combo = alpha * previous estimate + (1 - alpha) * mag * e^{i c}
///   new estimate = mag * combo / |combo|
///
/// where c is the phase of the re-analysis. With momentum 0 this reduces to
/// the classic Griffin-Lim algorithm.
class GriffinLim {
public:
    GriffinLim() = default;
    ~GriffinLim() = default;

    /// magnitudes and phases provide the initial estimate; num_bins is the
    /// number of valid spectral values (the vectors may hold more slack).
    /// out_samples must be sized by the caller (num_bins + step, as the
    /// istft convention) and receives the reconstructed signal.
    void process(std::vector<float>& magnitudes,
                 std::vector<float>& phases,
                 size_t num_bins,
                 std::vector<float>& out_samples,
                 int fftsize,
                 int overlap,
                 const string window_string,
                 long iters,
                 double momentum,
                 bool full_spectrum = false)
    {
        const size_t step = size_t(fftsize) / size_t(overlap);
        const size_t fftsize_2 = size_t(fftsize) / 2;
        const long spec_dim = full_spectrum ? fftsize : long(fftsize_2);

        // rect working state from the initial (magnitude, phase) estimate
        m_rect_real.resize(num_bins, 0.0f);
        m_rect_imag.resize(num_bins, 0.0f);
        for (size_t i = 0; i < num_bins; ++i) {
            m_rect_real[i] = magnitudes[i] * std::cos(phases[i]);
            m_rect_imag[i] = magnitudes[i] * std::sin(phases[i]);
        }

        // the consistent re-analysis of out_samples covers as many frames
        // as fit in the sample buffer
        const size_t analysis_frames = (out_samples.size() + step - 1) / step;
        m_consistent_real.assign(analysis_frames * spec_dim, 0.0f);
        m_consistent_imag.assign(analysis_frames * spec_dim, 0.0f);

        for (long iter = 0; iter < iters; ++iter) {
            m_istft.process(m_rect_real, m_rect_imag, out_samples, fftsize, overlap, window_string, false, full_spectrum);
            m_stft.process(out_samples, m_consistent_real, m_consistent_imag, fftsize, overlap, window_string, true, full_spectrum);

            // restore the original magnitudes, blending the previous phase
            // estimate with the consistent analysis phase
            const double a = momentum;
            const double b = 1.0 - momentum;
            for (size_t i = 0; i < num_bins; ++i) {
                const double mag = magnitudes[i];
                const double c = m_consistent_imag[i];  // consistent phase
                const double combo_r = a * m_rect_real[i] + b * mag * std::cos(c);
                const double combo_i = a * m_rect_imag[i] + b * mag * std::sin(c);
                const double nrm = std::sqrt(combo_r * combo_r + combo_i * combo_i);
                if (nrm > 0.0) {
                    const double scale = mag / nrm;
                    m_rect_real[i] = float(combo_r * scale);
                    m_rect_imag[i] = float(combo_i * scale);
                }
                // else: keep the previous estimate (combo cancelled exactly)
            }
        }

        // final synthesis with the refined estimate
        m_istft.process(m_rect_real, m_rect_imag, out_samples, fftsize, overlap, window_string, false, full_spectrum);
    }

private:
    STFT m_stft;
    ISTFT m_istft;
    std::vector<float> m_rect_real;
    std::vector<float> m_rect_imag;
    std::vector<float> m_consistent_real;
    std::vector<float> m_consistent_imag;
};
