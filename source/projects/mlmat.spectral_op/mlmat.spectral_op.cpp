/// @file
/// mlmat.spectral_op: bin-wise arithmetic (add, subtract, multiply,
/// max, min, mix) on the spectral matrices produced by two
/// mlmat.buffer_stft objects, for STFT-domain mixing and crossfades
/// with mlmat.buffer_istft.
///
/// @license    Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "mlmat_stft.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace c74;
using namespace c74::min;

void mlmat_spectral_op_assist(void* x, void* b, long m, long a, char* s);
max::t_jit_err mlmat_spectral_op_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs);

class mlmat_spectral_op : public object<mlmat_spectral_op>, public matrix_operator<>
{
public:
    MIN_DESCRIPTION    { "Bin-wise arithmetic on the spectral matrices from two mlmat.buffer_stft analyses: add, subtract, multiply, max, min, or mix, resynthesized with mlmat.buffer_istft." };
    MIN_TAGS        { "audio, analysis" };
    MIN_AUTHOR        { "Todd Ingalls" };
    MIN_RELATED        { "mlmat.buffer_stft, mlmat.buffer_istft, mlmat.spectral_crosssynth, mlmat.spectral_mask" };
    MIN_DISCUSSION  {"The two inputs are the 2-plane spectral matrices output by <o>mlmat.buffer_stft</o> (plane 0 and plane 1 hold either real and imaginary parts, or magnitudes and phases, according to the <at>input_polar</at> attribute); the output uses the same format as the input and feeds <o>mlmat.buffer_istft</o> directly. add, subtract and mix are complex-domain operations, linear in the STFT coefficients: add is exactly the analysis of the summed signals (spectral-domain mixing), and mix is a true crossfade, (1 - <at>mix</at>) * input 0 + <at>mix</at> * input 1, equivalent to a time-domain crossfade of the two signals. multiply is the frame-wise complex product -- the same operation as <o>mlmat.spectral_crosssynth</o>, which remains the dedicated object for it (see its reference page for the windowing scaling and gain compensation). max and min select, per bin, the whole content (magnitude and phase) of whichever input has the larger (smaller) magnitude. <at>gain</at> scales the output after the operation. Both inputs must use the same fftsize; the output frame count matches input 0, and input 1 loops if shorter, so a short matrix can shape a longer signal without truncating the output."};

    attribute<min::symbol> op { this, "op", "add",
        description {
            "Bin-wise operation: add (complex sum), subtract (complex difference), multiply (complex product), max (the bin of whichever input has the larger magnitude), min (smaller magnitude), or mix (complex crossfade by the mix attribute)."
        },
        range {"add", "subtract", "multiply", "max", "min", "mix"}
    };

    attribute<double> mix { this, "mix", 0.5,
        description {
            "Crossfade amount for the mix operation: 0 is input 0 alone, 1 is input 1 alone."
        },
        range {0.0, 1.0}
    };

    attribute<double> gain { this, "gain", 1.0,
        description {
            "Output gain, applied after the operation."
        }
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
        max::jit_class_addmethod(c, (max::method)mlmat_spectral_op_matrix_calc, "matrix_calc", max::A_CANT, 0);
        return {};
    }};

    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        max::max_jit_class_mop_wrap(c, this_jit_class, 0);
        max::max_jit_class_wrap_standard(c, this_jit_class, 0);
        max::class_addmethod(c, (max::method)mlmat_spectral_op_assist, "assist", max::A_CANT, 0);
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
            cerr << "mlmat.spectral_op: input matrices must be float32 or float64." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in0_minfo.dimcount != 2 || in1_minfo.dimcount != 2) {
            cerr << "mlmat.spectral_op: input matrices must be 2d (frames x bins)." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in0_minfo.planecount != 2 || in1_minfo.planecount != 2) {
            cerr << "mlmat.spectral_op: input matrices must have 2 planes." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in0_minfo.dim[1] != in1_minfo.dim[1]) {
            cerr << "mlmat.spectral_op: input bin counts differ (" << in0_minfo.dim[1]
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
            // output matches input 0; input 1 loops if shorter, so a
            // short matrix applied to a longer signal doesn't truncate
            // the output
            const long frames = in0_minfo.dim[0];
            const long frames_b = in1_minfo.dim[0];
            const size_t bins = static_cast<size_t>(in0_minfo.dim[1]);

            // the packed real Nyquist only exists in rectangular
            // half-spectrum input (polar output of buffer_stft drops it)
            const bool packed_nyquist = !static_cast<bool>(input_polar) && !static_cast<bool>(full_spectrum);

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
            // every phase and all modes output noise
            if (!m_polar_warned && !static_cast<bool>(input_polar)) {
                double p0max = 0.0, p1max = 0.0, q0max = 0.0, q1max = 0.0;
                for (size_t k = 0; k < bins; ++k) {
                    const max::uchar* ip = in0_data + static_cast<size_t>(k) * in0_minfo.dimstride[1];
                    const max::uchar* iq = in1_data + static_cast<size_t>(k) * in1_minfo.dimstride[1];
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
                    cerr << "mlmat.spectral_op: inlet 0 looks like a POLAR (magnitude/phase) matrix but the object expects rectangular -- every phase would be misread. Set @input_polar 1, or make the analysis output rectangular." << endl;
                }
                if (q1max <= 3.15 && q0max > 1.0) {
                    m_polar_warned = true;
                    cerr << "mlmat.spectral_op: inlet 1 looks like a POLAR (magnitude/phase) matrix but the object expects rectangular -- every phase would be misread. Set @input_polar 1, or make the analysis output rectangular." << endl;
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

            // resolve the operation once per call
            mlmat::op_kind ook = mlmat::op_kind::add;
            if (op == min::symbol("subtract"))
                ook = mlmat::op_kind::subtract;
            else if (op == min::symbol("multiply"))
                ook = mlmat::op_kind::multiply;
            else if (op == min::symbol("max"))
                ook = mlmat::op_kind::max;
            else if (op == min::symbol("min"))
                ook = mlmat::op_kind::min;
            else if (op == min::symbol("mix"))
                ook = mlmat::op_kind::mix;

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

                mlmat::spectral_op(mag_a.data(), phase_a.data(),
                                   mag_b.data(), phase_b.data(),
                                   mag_o.data(), phase_o.data(), bins,
                                   ook, mix, static_cast<float>(gain));

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
                    mlmat::spectral_op(&mna, &pna, &mnb, &pnb, &mno, &pno, 1, ook,
                                       mix, static_cast<float>(gain));
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

    // one-shot polar-input warning
    bool m_polar_warned = false;
};

max::t_jit_err mlmat_spectral_op_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs) {
    max::t_jit_err err = max::JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return max::JIT_ERR_INVALID_PTR;
    auto in0_matrix = (max::t_object*)max::object_method(inputs, max::_jit_sym_getindex, 0);
    auto in1_matrix = (max::t_object*)max::object_method(inputs, max::_jit_sym_getindex, 1);
    auto out_matrix = (max::t_object*)max::object_method(outputs, max::_jit_sym_getindex, 0);
    if (!in0_matrix || !in1_matrix || !out_matrix)
        return max::JIT_ERR_INVALID_PTR;
    minwrap<mlmat_spectral_op>* job = (minwrap<mlmat_spectral_op>*)(x);
    err = job->m_min_object.matrix_calc(x, in0_matrix, in1_matrix, out_matrix);
    return err;
}

void mlmat_spectral_op_assist(void* x, void* b, long io, long index, char* s) {
    switch (io) {
        case 1:
            switch (index) {
                case 0:
                    snprintf(s, 64, "(matrix) first spectral analysis (2 planes)");
                    break;
                case 1:
                    snprintf(s, 64, "(matrix) second spectral analysis (2 planes)");
                    break;
                default:
                    break;
            }
            break;
        case 2:
            switch (index) {
                case 0:
                    snprintf(s, 64, "(matrix) result spectra (2 planes, feeds mlmat.buffer_istft)");
                    break;
                default:
                    snprintf(s, 64, "dumpout");
                    break;
            }
    }
}

MIN_EXTERNAL(mlmat_spectral_op);
