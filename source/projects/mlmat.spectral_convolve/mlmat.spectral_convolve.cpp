/// @file
/// mlmat.spectral_convolve: frame-wise spectral convolution (complex
/// multiplication) of the spectral matrices produced by two
/// mlmat.buffer_stft objects, for STFT cross-synthesis with
/// mlmat.buffer_istft.
///
/// @license    Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "mlmat_stft.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace c74;
using namespace c74::min;

void mlmat_spectral_convolve_assist(void* x, void* b, long m, long a, char* s);
max::t_jit_err mlmat_spectral_convolve_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs);

class mlmat_spectral_convolve : public object<mlmat_spectral_convolve>, public matrix_operator<>
{
public:
    MIN_DESCRIPTION    { "Convolve the spectral matrices of two mlmat.buffer_stft analyses (frame-wise complex multiplication of the spectra), for STFT cross-synthesis with mlmat.buffer_istft." };
    MIN_TAGS        { "audio, analysis" };
    MIN_AUTHOR        { "Todd Ingalls" };
    MIN_RELATED        { "mlmat.buffer_stft, mlmat.buffer_istft, jit.fft" };
    MIN_DISCUSSION  {"Convolution in the STFT domain is multiplication of spectra: each output frame is the complex product of the corresponding input frames, computed with the same packing conventions as the rest of the spectral stack. The output feeds <o>mlmat.buffer_istft</o> directly. The two inputs must use the same fftsize; the number of output frames is the shorter of the two inputs. Because both inputs passed through the analysis windowing, the resynthesized result is scaled by the windowing of both; use <at>gain</at> to compensate to taste."};

    attribute<bool> input_polar { this, "input_polar", false,
        description {
            "If set, the input planes hold magnitudes and phases (as with mlmat.buffer_stft's output_polar); otherwise real and imaginary parts."
        }
    };

    attribute<bool> output_polar { this, "output_polar", false,
        description {
            "If set, output magnitudes and phases; otherwise real and imaginary parts."
        }
    };

    attribute<bool> full_spectrum { this, "full_spectrum", false,
        description {
            "If set, the inputs are full spectra (fftsize bins) as output by mlmat.buffer_stft with full_spectrum; otherwise half spectra (fftsize/2 bins)."
        }
    };

    attribute<double> gain { this, "gain", 1.0,
        description {
            "Output gain. Convolution of two windowed analyses scales the result down by both windows; adjust to taste."
        }
    };

    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        max::t_object* mop = static_cast<max::t_object*>(max::jit_object_new(max::_jit_sym_jit_mop, 2, 1));
        max::jit_mop_single_type(mop, max::_jit_sym_float64);
        max::jit_class_addadornment(c, mop);
        max::jit_class_addmethod(c, (max::method)mlmat_spectral_convolve_matrix_calc, "matrix_calc", max::A_CANT, 0);
        return {};
    }};

    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        max::max_jit_class_mop_wrap(c, this_jit_class, 0);
        max::max_jit_class_wrap_standard(c, this_jit_class, 0);
        max::class_addmethod(c, (max::method)mlmat_spectral_convolve_assist, "assist", max::A_CANT, 0);
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
            cerr << "mlmat.spectral_convolve: input matrices must be float32 or float64." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in0_minfo.dimcount != 2 || in1_minfo.dimcount != 2) {
            cerr << "mlmat.spectral_convolve: input matrices must be 2d (frames x bins)." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in0_minfo.planecount != 2 || in1_minfo.planecount != 2) {
            cerr << "mlmat.spectral_convolve: input matrices must have 2 planes." << endl;
            err = max::JIT_ERR_INVALID_INPUT;
            goto out;
        }
        if (in0_minfo.dim[1] != in1_minfo.dim[1]) {
            cerr << "mlmat.spectral_convolve: input bin counts differ (" << in0_minfo.dim[1]
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
            const long frames = std::min(in0_minfo.dim[0], in1_minfo.dim[0]);
            const size_t bins = static_cast<size_t>(in0_minfo.dim[1]);
            if (in0_minfo.dim[0] != in1_minfo.dim[0]) {
                cout << "mlmat.spectral_convolve: frame counts differ (" << in0_minfo.dim[0]
                     << " and " << in1_minfo.dim[0] << "); outputting " << frames << " frames."
                     << endl;
            }

            max::uchar* in0_data = nullptr;
            max::uchar* in1_data = nullptr;
            max::object_method(in0_matrix, max::_jit_sym_getdata, &in0_data);
            max::object_method(in1_matrix, max::_jit_sym_getdata, &in1_data);
            if (in0_data == nullptr || in1_data == nullptr) {
                err = max::JIT_ERR_INVALID_INPUT;
                goto out;
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

            // per-frame scratch
            std::vector<float> re1(bins), im1(bins), re2(bins), im2(bins);
            std::vector<float> re_out(bins), im_out(bins);

            for (long f = 0; f < frames; ++f) {
                // unpack both frames (converting polar input to rect)
                max::uchar* const rows[2] = {
                    in0_data + static_cast<size_t>(f) * in0_minfo.dimstride[0],
                    in1_data + static_cast<size_t>(f) * in1_minfo.dimstride[0]
                };
                const max::t_jit_matrix_info* const minfos[2] = { &in0_minfo, &in1_minfo };
                const bool is64[2] = { is0_64, is1_64 };
                float* const res[2] = { re1.data(), re2.data() };
                float* const ims[2] = { im1.data(), im2.data() };
                for (int s = 0; s < 2; ++s) {
                    const max::t_jit_matrix_info& mi = *minfos[s];
                    for (size_t k = 0; k < bins; ++k) {
                        max::uchar* ip = rows[s] + static_cast<size_t>(k) * mi.dimstride[1];
                        double p0, p1;
                        if (is64[s]) {
                            p0 = reinterpret_cast<const double*>(ip)[0];
                            p1 = reinterpret_cast<const double*>(ip)[1];
                        } else {
                            p0 = static_cast<double>(reinterpret_cast<const float*>(ip)[0]);
                            p1 = static_cast<double>(reinterpret_cast<const float*>(ip)[1]);
                        }
                        if (input_polar) {
                            res[s][k] = static_cast<float>(p0 * std::cos(p1));
                            ims[s][k] = static_cast<float>(p0 * std::sin(p1));
                        } else {
                            res[s][k] = static_cast<float>(p0);
                            ims[s][k] = static_cast<float>(p1);
                        }
                    }
                }

                // frame-wise spectral product
                mlmat::spectral_multiply(re1.data(), im1.data(), re2.data(), im2.data(),
                                         re_out.data(), im_out.data(), bins,
                                         full_spectrum, static_cast<float>(gain));

                max::uchar* orow = dataptr + static_cast<size_t>(f) * dest_minfo.dimstride[0];
                for (size_t k = 0; k < bins; ++k) {
                    double* op = reinterpret_cast<double*>(orow + static_cast<size_t>(k) * dest_minfo.dimstride[1]);
                    if (output_polar) {
                        op[0] = std::sqrt(static_cast<double>(re_out[k]) * re_out[k]
                                          + static_cast<double>(im_out[k]) * im_out[k]);
                        op[1] = std::atan2(static_cast<double>(im_out[k]),
                                           static_cast<double>(re_out[k]));
                    } else {
                        op[0] = re_out[k];
                        op[1] = im_out[k];
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
};

max::t_jit_err mlmat_spectral_convolve_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs) {
    max::t_jit_err err = max::JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return max::JIT_ERR_INVALID_PTR;
    auto in0_matrix = (max::t_object*)max::object_method(inputs, max::_jit_sym_getindex, 0);
    auto in1_matrix = (max::t_object*)max::object_method(inputs, max::_jit_sym_getindex, 1);
    auto out_matrix = (max::t_object*)max::object_method(outputs, max::_jit_sym_getindex, 0);
    if (!in0_matrix || !in1_matrix || !out_matrix)
        return max::JIT_ERR_INVALID_PTR;
    minwrap<mlmat_spectral_convolve>* job = (minwrap<mlmat_spectral_convolve>*)(x);
    err = job->m_min_object.matrix_calc(x, in0_matrix, in1_matrix, out_matrix);
    return err;
}

void mlmat_spectral_convolve_assist(void* x, void* b, long io, long index, char* s) {
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
                    snprintf(s, 64, "(matrix) convolved spectra (2 planes, feeds mlmat.buffer_istft)");
                    break;
                default:
                    snprintf(s, 64, "dumpout");
                    break;
            }
    }
}

MIN_EXTERNAL(mlmat_spectral_convolve);
