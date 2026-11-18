/// @file
/// mlmat.spectral_reassign: amplitude-corrected magnitudes and
/// per-bin reassigned (instantaneous) frequencies from the spectral
/// matrices produced by mlmat.buffer_stft.
///
/// @license    Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "mlmat_stft.hpp"

#include <cmath>
#include <string>
#include <vector>

using namespace c74;
using namespace c74::min;

void mlmat_spectral_reassign_assist(void* x, void* b, long m, long a, char* s);
max::t_jit_err mlmat_spectral_reassign_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs);

class mlmat_spectral_reassign : public object<mlmat_spectral_reassign>, public matrix_operator<>
{
public:
    MIN_DESCRIPTION    { "Amplitude-corrected magnitudes and reassigned per-bin frequencies from the spectral matrices produced by mlmat.buffer_stft." };
    MIN_TAGS        { "audio, analysis" };
    MIN_AUTHOR        { "Todd Ingalls" };
    MIN_RELATED        { "mlmat.buffer_stft, mlmat.spectral_peaks, sigmund~, jit.fft" };
    MIN_DISCUSSION  {"The input is the 2-plane spectral matrix output by <o>mlmat.buffer_stft</o> (plane 0 and plane 1 hold either real and imaginary parts, or magnitudes and phases, according to the <at>input_polar</at> attribute). The output has the same dimensions, with plane 0 holding the amplitude and plane 1 the frequency in Hz of every bin. Amplitudes are corrected with the coherent gain of the <at>window</at>, so a sinusoid of amplitude A reads back as A regardless of the window or fftsize -- set <at>window</at> to match the analysis window of mlmat.buffer_stft (raw windowed magnitudes would instead show, e.g., half amplitude for a hanning). Frequencies are reassigned: computed from the STFT phase difference between consecutive frames (instantaneous frequency estimation), giving exact sub-bin precision for stable sinusoids; the first frame (and any frame after an <at>overlap</at> or <at>samplerate</at> change) falls back to the bin center frequencies. Set <at>overlap</at> to match the STFT analysis overlap. Frequencies are computed from <at>samplerate</at> if set, otherwise from the current MSP sample rate (which matches the buffer_stft chain, since its resampling converts its input to the MSP rate). For <at>full_spectrum</at> input only the first half of the bins is meaningful (mlmat.buffer_stft's full-spectrum output mirrors the half spectrum, so the mirrored bins repeat the first half's data with the wrong bin indices). Use <o>mlmat.spectral_peaks</o> instead when only the strongest peaks are wanted."};

    attribute<min::symbol> window { this, "window", "hanning",
        description {
            "The analysis window used by the STFT whose output is being corrected. Must match mlmat.buffer_stft's window: the window's coherent gain is what turns raw magnitudes into true amplitudes."
        },
        range{"square", "triangle", "hanning", "hamming", "blackman"}
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
            "The STFT overlap factor used in the analysis. Must match mlmat.buffer_stft's overlap setting for correct frequency reassignment."
        },
        range {1, 65536}
    };

    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        max::t_object* mop = static_cast<max::t_object*>(max::jit_object_new(max::_jit_sym_jit_mop, 1, 1));
        max::jit_mop_single_type(mop, max::_jit_sym_float64);
        max::jit_class_addadornment(c, mop);
        max::jit_class_addmethod(c, (max::method)mlmat_spectral_reassign_matrix_calc, "matrix_calc", max::A_CANT, 0);
        return {};
    }};

    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        max::max_jit_class_mop_wrap(c, this_jit_class, 0);
        max::max_jit_class_wrap_standard(c, this_jit_class, 0);
        max::class_addmethod(c, (max::method)mlmat_spectral_reassign_assist, "assist", max::A_CANT, 0);
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
            cerr << "mlmat.spectral_reassign: input matrix must be float32 or float64 (use mlmat.convert or mlmat.buffer_stft output)." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in_minfo.dimcount != 2) {
            cerr << "mlmat.spectral_reassign: input matrix must be 2d (frames x bins), received " << in_minfo.dimcount << "d." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in_minfo.planecount != 2) {
            cerr << "mlmat.spectral_reassign: input matrix must have 2 planes, received " << in_minfo.planecount << "." << endl;
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

            // coherent gain of the analysis window, cached per
            // (fftsize, window): the amplitude correction factor
            const string window_string = window.get().c_str();
            if (m_gain_fftsize != fftsize || m_gain_window != window_string) {
                m_gain = mlmat::window_coherent_gain(fftsize, window_string);
                m_gain_fftsize = fftsize;
                m_gain_window = window_string;
            }

            // reset the phase state when the overlap or samplerate
            // changes (the expected phase advance changes)
            if (m_prev_overlap != overlap || m_prev_samplerate != sr) {
                m_have_prev_frame = false;
                m_prev_phases.clear();
                m_prev_overlap = overlap;
                m_prev_samplerate = sr;
            }

            max::uchar* in_data = nullptr;
            max::object_method(in_matrix, max::_jit_sym_getdata, &in_data);
            if (in_data == nullptr) {
                err = max::JIT_ERR_INVALID_INPUT;
                goto out;
            }

            // the output mirrors the input's dimensions, one bin per row
            max::t_jit_matrix_info out_minfo;
            out_minfo.type = max::_jit_sym_float64;
            out_minfo.planecount = 2;
            out_minfo.dimcount = 2;
            out_minfo.dim[0] = frames;
            out_minfo.dim[1] = static_cast<long>(bins);
            max::object_method(out_matrix, max::_jit_sym_setinfo, &out_minfo);
            max::t_jit_matrix_info dest_minfo;
            max::object_method(out_matrix, max::_jit_sym_getinfo, &dest_minfo);
            max::uchar* dataptr = nullptr;
            max::object_method(out_matrix, max::_jit_sym_getdata, &dataptr);
            if (dataptr == nullptr) {
                err = max::JIT_ERR_INVALID_OUTPUT;
                goto out;
            }

            // amplitude correction: the unscaled forward DFT puts a
            // bin-centered sinusoid of amplitude A at A * fftsize * cg / 2
            // in the complex bins; the real DC bin (half spectrum, bin 0:
            // re holds the real DC) is the exception -- all its energy is
            // in one real bin, so there is no factor 2
            const double denom = double(fftsize) * m_gain;
            const double scale_complex = 2.0 / denom;
            const double scale_real = 1.0 / denom;

            // per-frame scratch
            std::vector<double> phases(bins);

            for (long f = 0; f < frames; ++f) {
                max::uchar* row = in_data + static_cast<size_t>(f) * in_minfo.dimstride[0];
                max::uchar* orow = dataptr + static_cast<size_t>(f) * dest_minfo.dimstride[0];

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

                    double mag;
                    if (input_polar) {
                        mag = p0;
                        phases[k] = p1;
                    } else if (k == 0 && !full_spectrum) {
                        // half spectrum + rect input: bin 0 packs the
                        // real DC in re (im holds the real Nyquist)
                        mag = std::fabs(p0);
                        phases[k] = (p0 >= 0.0) ? 0.0 : M_PI;
                    } else {
                        mag = std::sqrt(p0 * p0 + p1 * p1);
                        phases[k] = std::atan2(p1, p0);
                    }

                    const bool real_only = (k == 0 && !full_spectrum);
                    const double amp = real_only ? mag * scale_real
                                                 : mag * scale_complex;

                    // reassigned frequency: the phase difference from
                    // the previous frame; bin centers on the first
                    double freq = mlmat::bin_frequency(static_cast<double>(k), sr, fftsize);
                    if (m_have_prev_frame && k < m_prev_phases.size()) {
                        const double rf = mlmat::reassigned_frequency(
                            static_cast<long>(k), phases[k], m_prev_phases[k],
                            fftsize, overlap, sr);
                        if (rf >= 0.0) {
                            freq = rf;
                        }
                    }

                    double* op = reinterpret_cast<double*>(orow + static_cast<size_t>(k) * dest_minfo.dimstride[1]);
                    op[0] = amp;
                    op[1] = freq;
                }

                m_prev_phases = phases;
                m_have_prev_frame = true;
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

    // coherent gain of the analysis window, cached per (fftsize, window)
    double m_gain = 1.0;
    int m_gain_fftsize = 0;
    string m_gain_window;

    // phases from the previous frame (per-bin), used to compute
    // instantaneous frequency via phase difference. reset when the
    // overlap or samplerate changes.
    std::vector<double> m_prev_phases;
    int m_prev_overlap = 0;
    double m_prev_samplerate = 0.0;
    bool m_have_prev_frame = false;
};

max::t_jit_err mlmat_spectral_reassign_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs) {
    max::t_jit_err err = max::JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return max::JIT_ERR_INVALID_PTR;
    auto in_matrix = (max::t_object*)max::object_method(inputs, max::_jit_sym_getindex, 0);
    auto out_matrix = (max::t_object*)max::object_method(outputs, max::_jit_sym_getindex, 0);
    if (!in_matrix || !out_matrix)
        return max::JIT_ERR_INVALID_PTR;
    minwrap<mlmat_spectral_reassign>* job = (minwrap<mlmat_spectral_reassign>*)(x);
    err = job->m_min_object.matrix_calc(x, in_matrix, out_matrix);
    return err;
}

void mlmat_spectral_reassign_assist(void* x, void* b, long io, long index, char* s) {
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
                    snprintf(s, 64, "(matrix) amplitudes (plane 0), reassigned frequencies in Hz (plane 1)");
                    break;
                default:
                    snprintf(s, 64, "dumpout");
                    break;
            }
    }
}

MIN_EXTERNAL(mlmat_spectral_reassign);
