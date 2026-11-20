/// @file
/// mlmat.spectral_mask: bin-wise masking of one spectral matrix by
/// another -- magnitude/phase cross-synthesis, vocoder gating, Wiener
/// soft masking and spectral subtraction -- on the matrices produced
/// by mlmat.buffer_stft.
///
/// @license    Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "mlmat_stft.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace c74;
using namespace c74::min;

void mlmat_spectral_mask_assist(void* x, void* b, long m, long a, char* s);
max::t_jit_err mlmat_spectral_mask_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs);

class mlmat_spectral_mask : public object<mlmat_spectral_mask>, public matrix_operator<>
{
public:
    MIN_DESCRIPTION    { "Bin-wise masking of one spectral matrix by another: magnitude/phase cross-synthesis, hard gating, Wiener soft masking, or spectral subtraction, on the matrices from two mlmat.buffer_stft analyses." };
    MIN_TAGS        { "audio, analysis" };
    MIN_AUTHOR        { "Todd Ingalls" };
    MIN_RELATED        { "mlmat.buffer_stft, mlmat.buffer_istft, mlmat.spectral_crosssynth, mlmat.spectral_op" };
    MIN_DISCUSSION  {"The two inputs are the 2-plane spectral matrices output by <o>mlmat.buffer_stft</o> (plane 0 and plane 1 hold either real and imaginary parts, or magnitudes and phases, according to the <at>input_polar</at> attribute); the output uses the same format as the input and feeds <o>mlmat.buffer_istft</o> directly. Input 1 (the right inlet) is the masker. In <at>mode</at> mag_from the output takes the magnitude of input 0 with the phase of input 1 -- the classic phase-vocoder cross-synthesis: one signal's timbre under another's temporal behavior; phase_from is the reverse. In mode hard, bins of input 0 are kept where input 1's amplitude is at least <at>threshold</at> (dB, full scale -- set <at>window</at> to match the analysis) and zeroed below it -- vocoder gating and noise suppression. In mode soft each bin of input 0 is scaled by |input 1|^2 / (|input 0|^2 + |input 1|^2), the Wiener soft mask, and the same threshold gates it: bins whose masker is below the floor are zeroed instead of leaking -- smooth, artifact-free cross-synthesis. The threshold has no effect in mag_from and phase_from modes. In mode subtract input 1's magnitudes are subtracted from input 0's (floored at zero, phase from input 0) -- spectral-subtraction noise reduction when input 1 is a noise-floor estimate. Both inputs must use the same fftsize; the output frame count matches input 0, and input 1 loops if shorter, so a short masker can shape a longer signal without truncating the output."};

    attribute<min::symbol> mode { this, "mode", "mag_from",
        description {
            "Mask operation: mag_from, phase_from, hard (gated by threshold), soft (Wiener weight, threshold-gated) or subtract (floored at zero). See the discussion for the per-mode formulas."
        },
        range {"mag_from", "phase_from", "hard", "soft", "subtract"}
    };

    attribute<min::symbol> window { this, "window", "hanning",
        description {
            "The analysis window used by the STFT whose output is being gated. Must match mlmat.buffer_stft's window: the window's coherent gain converts the amplitude-domain threshold to the analysis magnitudes' scale."
        },
        range{"square", "triangle", "hanning", "hamming", "blackman"}
    };

    attribute<double> threshold { this, "threshold", -60.0,
        description {
            "Minimum masker amplitude in dB (full scale), for hard and soft modes: bins whose masker amplitude is below this are zeroed. Set window to match the analysis. In mag_from and phase_from modes the threshold has no effect."
        },
        range {-120.0, 0.0}
    };

    attribute<bool> input_polar { this, "input_polar", false,
        description {
            "If set, the input planes hold magnitudes and phases (as with mlmat.buffer_stft's output_polar); otherwise real and imaginary parts. The output uses the same format as the input."
        }
    };

    attribute<bool> full_spectrum { this, "full_spectrum", false,
        description {
            "If set, the inputs are full spectra (fftsize bins) as output by mlmat.buffer_stft with full_spectrum; otherwise half spectra (fftsize/2 bins)."
        }
    };

    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        max::t_object* mop = static_cast<max::t_object*>(max::jit_object_new(max::_jit_sym_jit_mop, 2, 1));
        max::jit_mop_single_type(mop, max::_jit_sym_float64);
        max::jit_class_addadornment(c, mop);
        max::jit_class_addmethod(c, (max::method)mlmat_spectral_mask_matrix_calc, "matrix_calc", max::A_CANT, 0);
        return {};
    }};

    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        max::max_jit_class_mop_wrap(c, this_jit_class, 0);
        max::max_jit_class_wrap_standard(c, this_jit_class, 0);
        max::class_addmethod(c, (max::method)mlmat_spectral_mask_assist, "assist", max::A_CANT, 0);
        return {};
    }};

    max::t_jit_err matrix_calc(max::t_object* x, max::t_object* in0_matrix, max::t_object* in1_matrix, max::t_object* out_matrix) {
        max::t_jit_err err = max::JIT_ERR_NONE;
        max::t_jit_matrix_info in0_minfo, in1_minfo;

        auto in0_savelock = max::object_method(in0_matrix, max::_jit_sym_lock, (void*)1);
        auto in1_savelock = max::object_method(in1_matrix, max::_jit_sym_lock, (void*)1);
        auto out_savelock = max::object_method(out_matrix, max::_jit_sym_lock, (void*)1);
        max::object_method(in0_matrix, max::_jit_sym_getinfo, &in0_minfo);
        max::object_method(in1_matrix, max::_jit_sym_getinfo, &in1_minfo);

        const bool is0_64 = (in0_minfo.type == max::_jit_sym_float64);
        const bool is1_64 = (in1_minfo.type == max::_jit_sym_float64);
        if ((!is0_64 && in0_minfo.type != max::_jit_sym_float32)
            || (!is1_64 && in1_minfo.type != max::_jit_sym_float32)) {
            cerr << "mlmat.spectral_mask: input matrices must be float32 or float64." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in0_minfo.dimcount != 2 || in1_minfo.dimcount != 2) {
            cerr << "mlmat.spectral_mask: input matrices must be 2d (frames x bins)." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in0_minfo.planecount != 2 || in1_minfo.planecount != 2) {
            cerr << "mlmat.spectral_mask: input matrices must have 2 planes." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in0_minfo.dim[1] != in1_minfo.dim[1]) {
            cerr << "mlmat.spectral_mask: input bin counts differ (" << in0_minfo.dim[1]
                 << " and " << in1_minfo.dim[1] << "); both analyses must use the same fftsize."
                 << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in0_minfo.dim[0] <= 0 || in0_minfo.dim[1] <= 0) {
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }

        // all locals live in this block: no goto crosses an initialization
        {
            // output matches input 0 (the source); input 1 (the masker)
            // loops if shorter, so a short masker applied to a longer
            // signal doesn't truncate the output
            const long frames = in0_minfo.dim[0];
            const long frames_b = in1_minfo.dim[0];
            const size_t bins = static_cast<size_t>(in0_minfo.dim[1]);

            // the packed real Nyquist only exists in rectangular
            // half-spectrum input (polar output of buffer_stft drops it)
            const bool packed_nyquist = !static_cast<bool>(input_polar) && !static_cast<bool>(full_spectrum);

            // infer the analysis fftsize from the matrix width, as
            // the other spectral consumers do
            const int fftsize = full_spectrum ? static_cast<int>(bins)
                                              : static_cast<int>(bins) * 2;

            // the threshold is amplitude-domain (full scale): convert
            // to the analysis magnitudes' raw scale with the window's
            // coherent gain (a bin-centered sinusoid of amplitude A
            // has raw magnitude A * fftsize * cg / 2). the packed real
            // Nyquist is a single real bin, so its gate is twice that
            const string window_string = window.get().c_str();
            if (m_gain_fftsize != fftsize || m_gain_window != window_string) {
                m_gain = mlmat::window_coherent_gain(fftsize, window_string);
                m_gain_fftsize = fftsize;
                m_gain_window = window_string;
            }
            const double amp_thr = static_cast<double>(mlmat::db_to_lin(static_cast<float>(threshold)));
            const double thr_raw = amp_thr * double(fftsize) * m_gain / 2.0;
            const double thr_nyq = amp_thr * double(fftsize) * m_gain;

            max::uchar* in0_data = nullptr;
            max::uchar* in1_data = nullptr;
            max::object_method(in0_matrix, max::_jit_sym_getdata, &in0_data);
            max::object_method(in1_matrix, max::_jit_sym_getdata, &in1_data);
            if (in0_data == nullptr || in1_data == nullptr) {
                err = max::JIT_ERR_INVALID_INPUT;
                goto out;
            }

            // polar-input heuristic (one warning per object instance):
            // phases are bounded by pi by definition, so a
            // rect-configured object receiving a 2-plane matrix whose
            // plane-1 values all fit in [-pi, pi] while plane 0
            // carries meaningful magnitudes is almost certainly being
            // fed a magnitude/phase matrix -- the unpack then destroys
            // every phase
            if (!m_polar_warned && !static_cast<bool>(input_polar)) {
                double p0max = 0.0, p1max = 0.0, q0max = 0.0, q1max = 0.0;
                for (size_t k = 0; k < bins; ++k) {
                    const max::uchar* ip = in0_data + static_cast<size_t>(k) * in0_minfo.dimstride[1];
                    double a, b;
                    if (is0_64) {
                        a = reinterpret_cast<const double*>(ip)[0];
                        b = reinterpret_cast<const double*>(ip)[1];
                    } else {
                        a = static_cast<double>(reinterpret_cast<const float*>(ip)[0]);
                        b = static_cast<double>(reinterpret_cast<const float*>(ip)[1]);
                    }
                    p0max = std::max(p0max, std::fabs(a));
                    p1max = std::max(p1max, std::fabs(b));
                    const max::uchar* iq = in1_data + static_cast<size_t>(k) * in1_minfo.dimstride[1];
                    if (is1_64) {
                        a = reinterpret_cast<const double*>(iq)[0];
                        b = reinterpret_cast<const double*>(iq)[1];
                    } else {
                        a = static_cast<double>(reinterpret_cast<const float*>(iq)[0]);
                        b = static_cast<double>(reinterpret_cast<const float*>(iq)[1]);
                    }
                    q0max = std::max(q0max, std::fabs(a));
                    q1max = std::max(q1max, std::fabs(b));
                }
                if (p1max <= 3.15 && p0max > 1.0) {
                    m_polar_warned = true;
                    cerr << "mlmat.spectral_mask: inlet 0 looks like a POLAR (magnitude/phase) matrix but the object expects rectangular -- every phase would be misread. Set @input_polar 1, or make the analysis output rectangular." << endl;
                }
                if (q1max <= 3.15 && q0max > 1.0) {
                    m_polar_warned = true;
                    cerr << "mlmat.spectral_mask: inlet 1 looks like a POLAR (magnitude/phase) matrix but the object expects rectangular -- every phase would be misread. Set @input_polar 1, or make the analysis output rectangular." << endl;
                }
            }
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

            // resolve the mode once per call
            mlmat::mask_mode mm = mlmat::mask_mode::mag_from;
            if (mode == min::symbol("phase_from"))
                mm = mlmat::mask_mode::phase_from;
            else if (mode == min::symbol("hard"))
                mm = mlmat::mask_mode::hard;
            else if (mode == min::symbol("soft"))
                mm = mlmat::mask_mode::soft;
            else if (mode == min::symbol("subtract"))
                mm = mlmat::mask_mode::subtract;

            // per-frame scratch
            std::vector<float> mag_a(bins), phase_a(bins), mag_b(bins), phase_b(bins);
            std::vector<float> mag_o(bins), phase_o(bins);

            // unpack one frame into magnitude/phase, honoring the
            // half-spectrum packing at bin 0 (re holds the real DC)
            auto unpack = [&](max::uchar* row, const max::t_jit_matrix_info& mi, bool is64src,
                              float* mag, float* phase) {
                for (size_t k = 0; k < bins; ++k) {
                    max::uchar* ip = row + static_cast<size_t>(k) * mi.dimstride[1];
                    double p0, p1;
                    if (is64src) {
                        p0 = reinterpret_cast<const double*>(ip)[0];
                        p1 = reinterpret_cast<const double*>(ip)[1];
                    } else {
                        p0 = static_cast<double>(reinterpret_cast<const float*>(ip)[0]);
                        p1 = static_cast<double>(reinterpret_cast<const float*>(ip)[1]);
                    }
                    if (input_polar) {
                        mag[k] = static_cast<float>(p0);
                        phase[k] = static_cast<float>(p1);
                    } else if (k == 0 && !full_spectrum) {
                        mag[k] = static_cast<float>(std::fabs(p0));
                        phase[k] = (p0 >= 0.0) ? 0.0f : static_cast<float>(M_PI);
                    } else {
                        mag[k] = static_cast<float>(std::sqrt(p0 * p0 + p1 * p1));
                        phase[k] = static_cast<float>(std::atan2(p1, p0));
                    }
                }
            };

            for (long f = 0; f < frames; ++f) {
                const long fb = (frames_b > 0) ? (f % frames_b) : 0;
                max::uchar* row_a = in0_data + static_cast<size_t>(f) * in0_minfo.dimstride[0];
                max::uchar* row_b = in1_data + static_cast<size_t>(fb) * in1_minfo.dimstride[0];

                unpack(row_a, in0_minfo, is0_64, mag_a.data(), phase_a.data());
                unpack(row_b, in1_minfo, is1_64, mag_b.data(), phase_b.data());


                mlmat::spectral_mask(mag_a.data(), phase_a.data(),
                                     mag_b.data(), phase_b.data(),
                                     mag_o.data(), phase_o.data(), bins,
                                     mm, static_cast<float>(thr_raw));

                // the packed real Nyquist (rect half spectrum) rides
                // through the same core as a sign-encoded real bin
                float nyq_o = 0.0f;
                if (packed_nyquist) {
                    max::uchar* ip_a = row_a; // bin 0 of input 0
                    max::uchar* ip_b = row_b; // bin 0 of input 1
                    double na = is0_64 ? reinterpret_cast<const double*>(ip_a)[1]
                                      : static_cast<double>(reinterpret_cast<const float*>(ip_a)[1]);
                    double nb = is1_64 ? reinterpret_cast<const double*>(ip_b)[1]
                                      : static_cast<double>(reinterpret_cast<const float*>(ip_b)[1]);
                    float mna = static_cast<float>(std::fabs(na));
                    float pna = (na >= 0.0) ? 0.0f : static_cast<float>(M_PI);
                    float mnb = static_cast<float>(std::fabs(nb));
                    float pnb = (nb >= 0.0) ? 0.0f : static_cast<float>(M_PI);
                    float mno = 0.0f, pno = 0.0f;
                    mlmat::spectral_mask(&mna, &pna, &mnb, &pnb, &mno, &pno, 1, mm,
                                         static_cast<float>(thr_nyq));
                    nyq_o = mno * static_cast<float>(std::cos(pno));
                }

                max::uchar* orow = dataptr + static_cast<size_t>(f) * dest_minfo.dimstride[0];
                for (size_t k = 0; k < bins; ++k) {
                    double* op = reinterpret_cast<double*>(orow + static_cast<size_t>(k) * dest_minfo.dimstride[1]);
                    if (input_polar) {
                        op[0] = mag_o[k];
                        op[1] = phase_o[k];
                    } else {
                        op[0] = static_cast<double>(mag_o[k]) * std::cos(phase_o[k]);
                        const double imv = (k == 0 && packed_nyquist)
                            ? static_cast<double>(nyq_o)
                            : static_cast<double>(mag_o[k]) * std::sin(phase_o[k]);
                        op[1] = imv;
                    }
                }
            }
        }

    out:
        max::object_method(out_matrix, max::_jit_sym_lock, out_savelock);
        max::object_method(in1_matrix, max::_jit_sym_lock, in1_savelock);
        max::object_method(in0_matrix, max::_jit_sym_lock, in0_savelock);
        return err;
    }

    template<typename matrix_type>
    matrix_type calc_cell(matrix_type input, const c74::min::matrix_info& info, c74::min::matrix_coord& position) {
        matrix_type output;
        return output;
    }

    // threshold conversion: coherent gain of the analysis window,
    // cached per (fftsize, window)
    double m_gain = 1.0;
    int m_gain_fftsize = 0;
    string m_gain_window;

    // one-shot polar-input warning
    bool m_polar_warned = false;

};

max::t_jit_err mlmat_spectral_mask_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs) {
    max::t_jit_err err = max::JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return max::JIT_ERR_INVALID_PTR;
    auto in0_matrix = (max::t_object*)max::object_method(inputs, max::_jit_sym_getindex, 0);
    auto in1_matrix = (max::t_object*)max::object_method(inputs, max::_jit_sym_getindex, 1);
    auto out_matrix = (max::t_object*)max::object_method(outputs, max::_jit_sym_getindex, 0);
    if (!in0_matrix || !in1_matrix || !out_matrix)
        return max::JIT_ERR_INVALID_PTR;
    minwrap<mlmat_spectral_mask>* job = (minwrap<mlmat_spectral_mask>*)(x);
    err = job->m_min_object.matrix_calc(x, in0_matrix, in1_matrix, out_matrix);
    return err;
}

void mlmat_spectral_mask_assist(void* x, void* b, long io, long index, char* s) {
    switch (io) {
        case 1:
            switch (index) {
                case 0:
                    snprintf(s, 64, "(matrix) source spectral analysis (2 planes)");
                    break;
                case 1:
                    snprintf(s, 64, "(matrix) masker spectral analysis (2 planes)");
                    break;
                default:
                    break;
            }
            break;
        case 2:
            switch (index) {
                case 0:
                    snprintf(s, 64, "(matrix) masked spectra (2 planes, feeds mlmat.buffer_istft)");
                    break;
                default:
                    snprintf(s, 64, "dumpout");
                    break;
            }
    }
}

MIN_EXTERNAL(mlmat_spectral_mask);
