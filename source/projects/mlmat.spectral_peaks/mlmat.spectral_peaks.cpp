/// @file
/// mlmat.spectral_peaks: amplitude and frequency extraction from the
/// spectral matrices produced by mlmat.buffer_stft.
///
/// @license    Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "mlmat_stft.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace c74;
using namespace c74::min;

void mlmat_spectral_peaks_assist(void* x, void* b, long m, long a, char* s);
max::t_jit_err mlmat_spectral_peaks_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs);

class mlmat_spectral_peaks : public object<mlmat_spectral_peaks>, public matrix_operator<>
{
public:
    MIN_DESCRIPTION    { "Extract amplitudes and frequencies from the spectral matrices produced by mlmat.buffer_stft." };
    MIN_TAGS        { "audio, analysis" };
    MIN_AUTHOR        { "Todd Ingalls" };
    MIN_RELATED        { "mlmat.buffer_stft, sigmund~, jit.fft" };
    MIN_DISCUSSION  {"The input is the 2-plane spectral matrix output by <o>mlmat.buffer_stft</o> (plane 0 and plane 1 hold either real and imaginary parts, or magnitudes and phases, according to the <at>input_polar</at> attribute). In <at>mode</at> peaks, the object finds the spectral peaks of each frame -- bins that are local maxima above the <at>threshold</at>, refined with parabolic interpolation for the true amplitude and frequency -- and outputs the strongest <at>max_peaks</at> of them as a 3-plane matrix: plane 0 amplitudes, plane 1 frequencies in Hz, plane 2 the analysis phases in radians at each peak's bin -- the complete (amplitude, frequency, phase) partial data that <o>mlmat.buffer_mqsynth</o> synthesizes from. In <at>mode</at> reassigned, peaks are detected the same way but their frequencies are computed from the STFT phase difference between consecutive frames (instantaneous frequency estimation), which gives exact sub-bin precision for stable sinusoids. Set <at>overlap</at> to match the STFT analysis overlap for correct results. For every-bin output with amplitudes and reassigned frequencies use <o>mlmat.spectral_reassign</o> instead. By default (<at>output</at> amplitudes) plane 0 holds true signal amplitudes, corrected with the coherent gain of the analysis <at>window</at> (set <at>window</at> to match mlmat.buffer_stft's window): a sinusoid of amplitude A reads as A regardless of fftsize and window. <at>output</at> magnitudes returns the raw windowed magnitudes of the analysis instead. <at>threshold</at> always gates on those raw magnitudes, regardless of <at>output</at>. Without tracking, the columns hold the per-frame amplitude ranking: column k is the k-th loudest peak of that frame, so peaks reorder or vanish between frames. Set <at>track</at> for stable identities: peaks are matched across frames (McAulay-Quatieri matching, within <at>track_delta</at> Hz), so each column holds one partial's trajectory -- new partials are born into the next free column, and a partial that vanishes zeroes out and frees its column after <at>track_hold</at> coast frames. Noise and window side-lobes can produce tiny false peaks right next to a strong peak; set <at>spacing</at> to prune them: a peak within spacing bins of a stronger kept peak is rejected as an artifact when it is also more than <at>spacing_db</at> below it (the hanning side-lobe floor is -31 dB, the default; 0 prunes on distance alone). Main-lobe widths: square 2, hanning and hamming 4, blackman 6 bins -- so a good spacing is 3 for hanning and 5 for blackman. Frequencies are computed from <at>samplerate</at> if set, otherwise from the current MSP sample rate (which matches the buffer_stft chain, since its resampling converts its input to the MSP rate)."};

    attribute<min::symbol> mode { this, "mode", "peaks",
        description {
            "Output mode: peaks (amplitude and frequency of the spectral peaks, frequency via parabolic interpolation) or reassigned (same as peaks, but frequency via phase difference for sub-bin precision)."
        },
        range {"peaks", "reassigned"}
    };

    attribute<min::symbol> window { this, "window", "hanning",
        description {
            "The analysis window used by the STFT whose output is being corrected. Must match mlmat.buffer_stft's window: the window's coherent gain is what turns raw magnitudes into true amplitudes for the amplitudes output."
        },
        range{"square", "triangle", "hanning", "hamming", "blackman"}
    };

    attribute<min::symbol> output { this, "output", "amplitudes",
        description {
            "Plane 0 output scaling: amplitudes (the default) corrects the raw magnitudes with the coherent gain of the analysis window, so a sinusoid of amplitude A reads as A; magnitudes returns the raw windowed magnitudes of the analysis. threshold always gates on the raw magnitudes either way."
        },
        range {"magnitudes", "amplitudes"}
    };

    attribute<int, threadsafe::no, limit::clamp> spacing { this, "spacing", 1,
        description {
            "Minimum peak separation in bins, to prune window side-lobe artifacts: a peak within this many bins of a stronger kept peak is rejected when it is also below spacing_db relative to it. 1 (the default) disables pruning. Main-lobe widths: square 2, hanning and hamming 4, blackman 6 bins -- so 3 (hanning) or 5 (blackman) are good values."
        },
        range {1, 32}
    };

    attribute<double> spacing_db { this, "spacing_db", -31.0,
        description {
            "Relative amplitude gate for spacing: a near peak is pruned only if it is also more than this many dB below the stronger kept peak (the hanning side-lobe floor is -31 dB, the default). 0 prunes on distance alone regardless of level."
        },
        range {-120.0, 0.0}
    };

    attribute<bool> track { this, "track", false,
        description {
            "If set, track partials across frames (McAulay-Quatieri matching). Each output column then holds one partial's trajectory: births take the next free column, deaths zero-fill after track_hold coast frames, instead of the per-frame amplitude ranking."
        }
    };

    attribute<double> track_delta { this, "track_delta", 0.0,
        description {
            "Matching tolerance in Hz for partial tracking. 0 (the default) auto-selects one bin (samplerate / fftsize)."
        },
        range {0.0, 192000.0}
    };

    attribute<int, threadsafe::no, limit::clamp> track_hold { this, "track_hold", 0,
        description {
            "Frames an unmatched partial coasts (zero amplitude, holding its frequency so it can reacquire) before it dies and frees its column. 0 kills immediately; 1-64 bridge brief dropouts or that number of frames."
        },
        range {0, 64}
    };

    attribute<double> samplerate { this, "samplerate", 0.0,
        description {
            "Sample rate in Hz used to compute frequencies. If 0 (the default), the current MSP sample rate is used, which matches the mlmat.buffer_stft chain when its resampling is on."
        },
        range {0.0, 192000.0}
    };

    attribute<bool> input_polar { this, "input_polar", false,
        description {
            "If set, the input planes hold magnitudes and phases (as with mlmat.buffer_stft's output_polar); otherwise real and imaginary parts."
        }
    };

    attribute<bool> full_spectrum { this, "full_spectrum", false,
        description {
            "If set, the input is a full spectrum (fftsize bins) as output by mlmat.buffer_stft with full_spectrum; otherwise the half spectrum (fftsize/2 bins)."
        }
    };

    attribute<int, threadsafe::no, limit::clamp> overlap { this, "overlap", 4,
        description {
            "The STFT overlap factor used in the analysis. Must match mlmat.buffer_stft's overlap setting for correct reassigned-mode frequency estimation."
        },
        range {1, 65536}
    };

    attribute<int, threadsafe::no, limit::clamp> max_peaks { this, "max_peaks", 16,
        description {
            "Maximum number of peaks output per frame in peaks mode. Fewer peaks than this are zero-filled."
        },
        range {1, 1024}
    };

    attribute<double> threshold { this, "threshold", -60.0,
        description {
            "Minimum peak amplitude in dB in peaks mode. Bins below this are ignored."
        },
        range {-120.0, 0.0}
    };

    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        max::t_object* mop = static_cast<max::t_object*>(max::jit_object_new(max::_jit_sym_jit_mop, 1, 1));
        max::jit_mop_single_type(mop, max::_jit_sym_float64);
        max::jit_class_addadornment(c, mop);
        max::jit_class_addmethod(c, (max::method)mlmat_spectral_peaks_matrix_calc, "matrix_calc", max::A_CANT, 0);
        return {};
    }};

    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        max::max_jit_class_mop_wrap(c, this_jit_class, 0);
        max::max_jit_class_wrap_standard(c, this_jit_class, 0);
        max::class_addmethod(c, (max::method)mlmat_spectral_peaks_assist, "assist", max::A_CANT, 0);
        return {};
    }};

    max::t_jit_err matrix_calc(max::t_object* x, max::t_object* in_matrix, max::t_object* out_matrix) {
        max::t_jit_err err = max::JIT_ERR_NONE;
        max::t_jit_matrix_info in_minfo;

        auto in_savelock = max::object_method(in_matrix, max::_jit_sym_lock, (void*)1);
        auto out_savelock = max::object_method(out_matrix, max::_jit_sym_lock, (void*)1);
        max::object_method(in_matrix, max::_jit_sym_getinfo, &in_minfo);

        const bool is64 = (in_minfo.type == max::_jit_sym_float64);
        if (!is64 && in_minfo.type != max::_jit_sym_float32) {
            cerr << "mlmat.spectral_peaks: input matrix must be float32 or float64 (use mlmat.convert or mlmat.buffer_stft output)." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in_minfo.dimcount != 2) {
            cerr << "mlmat.spectral_peaks: input matrix must be 2d (frames x bins), received " << in_minfo.dimcount << "d." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in_minfo.planecount != 2) {
            cerr << "mlmat.spectral_peaks: input matrix must have 2 planes, received " << in_minfo.planecount << "." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in_minfo.dim[0] <= 0 || in_minfo.dim[1] <= 0) {
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }

        // all locals live in this block: no goto crosses an initialization
        {
            const long frames = in_minfo.dim[0];
            const size_t bins = static_cast<size_t>(in_minfo.dim[1]);
            const int fftsize = full_spectrum ? static_cast<int>(bins)
                                              : static_cast<int>(bins) * 2;
            const double sr = (samplerate > 0.0) ? static_cast<double>(samplerate)
                                                : static_cast<double>(max::sys_getsr());
            const bool reassigned_mode = (mode == min::symbol("reassigned"));
            const size_t out_cols = static_cast<size_t>(max_peaks);

            // amplitude correction: coherent gain of the analysis
            // window, cached per (fftsize, window)
            const bool amplitudes_output = (output != min::symbol("magnitudes"));
            const string window_string = window.get().c_str();
            if (m_gain_fftsize != fftsize || m_gain_window != window_string) {
                m_gain = mlmat::window_coherent_gain(fftsize, window_string);
                m_gain_fftsize = fftsize;
                m_gain_window = window_string;
            }
            // the unscaled forward DFT puts a bin-centered sinusoid of
            // amplitude A at A * fftsize * cg / 2 in the complex bins;
            // the packed real Nyquist is the exception (one real bin,
            // no factor 2)
            const double denom = double(fftsize) * m_gain;
            const double scale_complex = amplitudes_output ? 2.0 / denom : 1.0;
            const double scale_real = amplitudes_output ? 1.0 / denom : 1.0;

            // each arriving matrix is a complete, self-contained
            // analysis: start the reassigned-mode phase state and the
            // partial tracker fresh on every matrix, so re-sending a
            // matrix reproduces the same output (state never carries
            // across matrix_calc calls -- leftover end-state from a
            // previous matrix would permute the tracker's column
            // assignments and shift frame 0's reassigned frequencies)
            m_have_prev_frame = false;
            m_prev_phases.clear();
            m_tracker.reset();

            max::uchar* in_data = nullptr;
            max::object_method(in_matrix, max::_jit_sym_getdata, &in_data);
            if (in_data == nullptr) {
                err = max::JIT_ERR_INVALID_INPUT;
                goto out;
            }

            max::t_jit_matrix_info out_minfo;
            out_minfo.type = max::_jit_sym_float64;
            out_minfo.planecount = 3;
            out_minfo.dimcount = 2;
            out_minfo.dim[0] = frames;
            out_minfo.dim[1] = static_cast<long>(out_cols);
            max::object_method(out_matrix, max::_jit_sym_setinfo, &out_minfo);
            max::t_jit_matrix_info dest_minfo;
            max::object_method(out_matrix, max::_jit_sym_getinfo, &dest_minfo);
            max::uchar* dataptr = nullptr;
            max::object_method(out_matrix, max::_jit_sym_getdata, &dataptr);
            if (dataptr == nullptr) {
                err = max::JIT_ERR_INVALID_OUTPUT;
                goto out;
            }

            // per-frame scratch
            std::vector<float> mags(bins);
            std::vector<double> phases(bins);
            std::vector<mlmat::spectral_peak> peaks;
            std::vector<mlmat::spectral_peak> tracked;

            for (long f = 0; f < frames; ++f) {
                max::uchar* row = in_data + static_cast<size_t>(f) * in_minfo.dimstride[0];

                // per-bin magnitudes (and phases for reassigned mode),
                // honoring the half-spectrum packing at bin 0 (re holds
                // the real DC, im the real Nyquist)
                for (size_t k = 0; k < bins; ++k) {
                    max::uchar* ip = row + static_cast<size_t>(k) * in_minfo.dimstride[1];
                    double p0, p1;
                    if (is64) {
                        p0 = reinterpret_cast<const double*>(ip)[0];
                        p1 = reinterpret_cast<const double*>(ip)[1];
                    } else {
                        p0 = static_cast<double>(reinterpret_cast<const float*>(ip)[0]);
                        p1 = static_cast<double>(reinterpret_cast<const float*>(ip)[1]);
                    }
                    if (input_polar) {
                        mags[k] = static_cast<float>(p0);
                        phases[k] = p1;
                    } else if (k == 0 && !full_spectrum) {
                        mags[k] = static_cast<float>(std::fabs(p0));
                        phases[k] = (p0 >= 0.0) ? 0.0 : M_PI;
                    } else {
                        mags[k] = static_cast<float>(std::sqrt(p0 * p0 + p1 * p1));
                        phases[k] = std::atan2(p1, p0);
                    }
                }

                max::uchar* orow = dataptr + static_cast<size_t>(f) * dest_minfo.dimstride[0];

                peaks.clear();
                // for a full spectrum only the first half (through the
                // Nyquist bin) is meaningful; the mirrored bins would
                // duplicate every peak
                if (full_spectrum) {
                    const size_t half = static_cast<size_t>(fftsize) / 2;
                    std::vector<float> half_mags(mags.begin(), mags.begin() + half + 1);
                    mlmat::spectral_find_peaks(half_mags, sr, fftsize,
                                               static_cast<float>(threshold),
                                               static_cast<size_t>(max_peaks), peaks);
                } else {
                    mlmat::spectral_find_peaks(mags, sr, fftsize,
                                               static_cast<float>(threshold),
                                               static_cast<size_t>(max_peaks), peaks);
                    // half spectrum + rect input: the real Nyquist
                    // (packed in im[0]) is a peak candidate at sr/2
                    if (!input_polar) {
                        max::uchar* ip = row; // bin 0
                        const double p1 = is64
                            ? reinterpret_cast<const double*>(ip)[1]
                            : static_cast<double>(reinterpret_cast<const float*>(ip)[1]);
                        const float nyq = static_cast<float>(std::fabs(p1));
                        // threshold always gates on the raw magnitudes
                        if (nyq >= mlmat::db_to_lin(static_cast<float>(threshold))) {
                            mlmat::spectral_peak p;
                            p.amplitude = static_cast<float>(double(nyq) * scale_real);
                            p.frequency = static_cast<float>(sr / 2.0);
                            if (peaks.size() < static_cast<size_t>(max_peaks)) {
                                peaks.push_back(p);
                            }
                        }
                    }
                }
                // prune side-lobe artifacts next to strong peaks
                // (before the reassignment refinement so the tracker
                // and the output both see the cleaned list)
                if (spacing > 1) {
                    mlmat::spectral_prune_peaks(peaks, fftsize, sr, spacing, spacing_db);
                }

                // reassigned mode: refine each peak's frequency using
                // the phase difference from the previous frame
                if (reassigned_mode && m_have_prev_frame) {
                    for (auto& pk : peaks) {
                        // recover the peak's bin index from the
                        // parabolic-interpolated frequency
                        const long bin_k = static_cast<long>(std::lround(
                            pk.frequency * double(fftsize) / sr));
                        if (bin_k >= 0 && static_cast<size_t>(bin_k) < m_prev_phases.size()) {
                            const double rf = mlmat::reassigned_frequency(
                                bin_k, phases[bin_k], m_prev_phases[bin_k],
                                fftsize, overlap, sr);
                            if (rf >= 0.0) {
                                pk.frequency = static_cast<float>(rf);
                            }
                        }
                    }
                }

                // amplitudes output: convert the raw windowed
                // magnitudes to true signal amplitudes. the Nyquist
                // candidate (exactly sr/2, never touched by the
                // reassignment refine) was already scaled with the
                // single-real-bin factor at append time
                if (amplitudes_output) {
                    const float nyq = static_cast<float>(sr / 2.0);
                    for (auto& pk : peaks) {
                        if (pk.frequency != nyq) {
                            pk.amplitude = static_cast<float>(pk.amplitude * scale_complex);
                        }
                    }
                }

                std::sort(peaks.begin(), peaks.end(),
                          [](const mlmat::spectral_peak& a, const mlmat::spectral_peak& b) {
                              return a.amplitude > b.amplitude;
                          });

                // plane 2: the analysis phase at the peak's bin, the
                // phase anchor for MQ synthesis. dead slots give 0;
                // the packed Nyquist candidate is a real value, so
                // its phase is 0 or pi
                auto phase_of = [&](double frequency, double amplitude) -> double {
                    if (frequency == 0.0 && amplitude == 0.0)
                        return 0.0;
                    if (frequency == static_cast<double>(sr) / 2.0)
                        return (amplitude >= 0.0) ? 0.0 : M_PI;
                    long k = static_cast<long>(std::lround(frequency * double(fftsize) / sr));
                    k = std::max(0L, std::min(k, static_cast<long>(bins) - 1));
                    return phases[static_cast<size_t>(k)];
                };

                if (track) {
                    // partial tracking: stable identity columns
                    // (delta auto = one bin)
                    const double delta = (track_delta > 0.0)
                        ? static_cast<double>(track_delta)
                        : sr / double(fftsize);
                    m_tracker.process(peaks, out_cols, delta, track_hold, tracked);
                    for (size_t k = 0; k < out_cols; ++k) {
                        double* op = reinterpret_cast<double*>(orow + static_cast<size_t>(k) * dest_minfo.dimstride[1]);
                        op[0] = tracked[k].amplitude;
                        op[1] = tracked[k].frequency;
                        op[2] = phase_of(tracked[k].frequency, tracked[k].amplitude);
                    }
                } else {
                    for (size_t k = 0; k < out_cols; ++k) {
                        double* op = reinterpret_cast<double*>(orow + static_cast<size_t>(k) * dest_minfo.dimstride[1]);
                        if (k < peaks.size()) {
                            op[0] = peaks[k].amplitude;
                            op[1] = peaks[k].frequency;
                            op[2] = phase_of(peaks[k].frequency, peaks[k].amplitude);
                        } else {
                            op[0] = 0.0;
                            op[1] = 0.0;
                            op[2] = 0.0;
                        }
                    }
                }

                // store this frame's phases for the next frame's
                // reassignment
                if (reassigned_mode) {
                    m_prev_phases = phases;
                    m_have_prev_frame = true;
                }
            }
        }

    out:
        max::object_method(out_matrix, max::_jit_sym_lock, out_savelock);
        max::object_method(in_matrix, max::_jit_sym_lock, in_savelock);
        return err;
    }

    template<typename matrix_type>
    matrix_type calc_cell(matrix_type input, const c74::min::matrix_info& info, c74::min::matrix_coord& position) {
        matrix_type output;
        return output;
    }

    // reassigned mode: phases from the previous frame (per-bin), used to
    // compute instantaneous frequency via phase difference. reset when
    // the overlap or samplerate changes.
    std::vector<double> m_prev_phases;
    bool m_have_prev_frame = false;

    // track mode: the partial tracker (identity-stable columns).
    // reset alongside the phase state (the auto track_delta depends
    // on the samplerate and fftsize).
    mlmat::peak_tracker m_tracker;

    // amplitudes output: coherent gain of the analysis window,
    // cached per (fftsize, window)
    double m_gain = 1.0;
    int m_gain_fftsize = 0;
    string m_gain_window;
};

max::t_jit_err mlmat_spectral_peaks_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs) {
    max::t_jit_err err = max::JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return max::JIT_ERR_INVALID_PTR;
    auto in_matrix = (max::t_object*)max::object_method(inputs, max::_jit_sym_getindex, 0);
    auto out_matrix = (max::t_object*)max::object_method(outputs, max::_jit_sym_getindex, 0);
    if (!in_matrix || !out_matrix)
        return max::JIT_ERR_INVALID_PTR;
    minwrap<mlmat_spectral_peaks>* job = (minwrap<mlmat_spectral_peaks>*)(x);
    err = job->m_min_object.matrix_calc(x, in_matrix, out_matrix);
    return err;
}

void mlmat_spectral_peaks_assist(void* x, void* b, long io, long index, char* s) {
    switch (io) {
        case 1:
            switch (index) {
                case 0:
                    snprintf(s, 64, "(matrix) spectral frames (2 planes)");
                    break;
                default:
                    break;
            }
            break;
        case 2:
            switch (index) {
                case 0:
                    snprintf(s, 64, "(matrix) amplitudes (plane 0), frequencies in Hz (plane 1), phases in radians (plane 2)");
                    break;
                default:
                    snprintf(s, 64, "dumpout");
                    break;
            }
    }
}

MIN_EXTERNAL(mlmat_spectral_peaks);
