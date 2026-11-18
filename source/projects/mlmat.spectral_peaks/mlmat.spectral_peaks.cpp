/// @file
/// mlmat.spectral_peaks: amplitude and frequency extraction from the
/// spectral matrices produced by mlmat.buffer_stft.
///
/// @license    Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "mlmat_stft.hpp"

#include <algorithm>
#include <cmath>
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
    MIN_DISCUSSION  {"The input is the 2-plane spectral matrix output by <o>mlmat.buffer_stft</o> (plane 0 and plane 1 hold either real and imaginary parts, or magnitudes and phases, according to the <at>input_polar</at> attribute). In <at>mode</at> peaks, the object finds the spectral peaks of each frame -- bins that are local maxima above the <at>threshold</at>, refined with parabolic interpolation for the true amplitude and frequency -- and outputs the strongest <at>max_peaks</at> of them as a 2-plane matrix: plane 0 amplitudes, plane 1 frequencies in Hz. In <at>mode</at> reassigned, peaks are detected the same way but their frequencies are computed from the STFT phase difference between consecutive frames (instantaneous frequency estimation), which gives exact sub-bin precision for stable sinusoids. Set <at>overlap</at> to match the STFT analysis overlap for correct results. For every-bin output with amplitude correction and reassigned frequencies use <o>mlmat.spectral_reassign</o> instead. Frequencies are computed from <at>samplerate</at> if set, otherwise from the current MSP sample rate (which matches the buffer_stft chain, since its resampling converts its input to the MSP rate)."};

    attribute<min::symbol> mode { this, "mode", "peaks",
        description {
            "Output mode: peaks (amplitude and frequency of the spectral peaks, frequency via parabolic interpolation) or reassigned (same as peaks, but frequency via phase difference for sub-bin precision)."
        },
        range {"peaks", "reassigned"}
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

    attribute<int, threadsafe::no, limit::clamp> overlap { this, "overlap", 1,
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

            // reassigned mode: reset the phase state when the overlap
            // or samplerate changes (the expected phase advance changes)
            if (reassigned_mode) {
                if (m_prev_overlap != overlap || m_prev_samplerate != sr) {
                    m_have_prev_frame = false;
                    m_prev_phases.clear();
                    m_prev_overlap = overlap;
                    m_prev_samplerate = sr;
                }
            }

            max::uchar* in_data = nullptr;
            max::object_method(in_matrix, max::_jit_sym_getdata, &in_data);
            if (in_data == nullptr) {
                err = max::JIT_ERR_INVALID_INPUT;
                goto out;
            }

            max::t_jit_matrix_info out_minfo;
            out_minfo.type = max::_jit_sym_float64;
            out_minfo.planecount = 2;
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
                        if (reassigned_mode) {
                            phases[k] = p1;
                        }
                    } else if (k == 0 && !full_spectrum) {
                        mags[k] = static_cast<float>(std::fabs(p0));
                        if (reassigned_mode) {
                            phases[k] = (p0 >= 0.0) ? 0.0 : M_PI;
                        }
                    } else {
                        mags[k] = static_cast<float>(std::sqrt(p0 * p0 + p1 * p1));
                        if (reassigned_mode) {
                            phases[k] = std::atan2(p1, p0);
                        }
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
                        if (nyq >= mlmat::db_to_lin(static_cast<float>(threshold))) {
                            mlmat::spectral_peak p;
                            p.amplitude = nyq;
                            p.frequency = static_cast<float>(sr / 2.0);
                            if (peaks.size() < static_cast<size_t>(max_peaks)) {
                                peaks.push_back(p);
                            }
                        }
                    }
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

                std::sort(peaks.begin(), peaks.end(),
                          [](const mlmat::spectral_peak& a, const mlmat::spectral_peak& b) {
                              return a.amplitude > b.amplitude;
                          });

                for (size_t k = 0; k < out_cols; ++k) {
                    double* op = reinterpret_cast<double*>(orow + static_cast<size_t>(k) * dest_minfo.dimstride[1]);
                    if (k < peaks.size()) {
                        op[0] = peaks[k].amplitude;
                        op[1] = peaks[k].frequency;
                    } else {
                        op[0] = 0.0;
                        op[1] = 0.0;
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
    int m_prev_overlap = 0;
    double m_prev_samplerate = 0.0;
    bool m_have_prev_frame = false;
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
                    snprintf(s, 64, "(matrix) amplitudes (plane 0), frequencies in Hz (plane 1)");
                    break;
                default:
                    snprintf(s, 64, "dumpout");
                    break;
            }
    }
}

MIN_EXTERNAL(mlmat_spectral_peaks);
