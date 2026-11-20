/// @file mlmat.buffer_fft.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved. Also based on examples provided with the mlpack library. Please see source/mlpack for license details
/// @license  Use of this source code is governed by the MIT License found in the License.md file.



#include "c74_min.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>
#include "mlmat_stft.hpp"
using namespace c74;
using namespace c74::min;




max::t_object* convert_to_float32(max::t_object *matrix, max::t_jit_matrix_info& minfo) {
    max::t_object *m; // destination matrix
    if(minfo.type == max::_jit_sym_float32) {
        return matrix;
    }
    max::t_jit_matrix_info dest_info = minfo;
    
    dest_info.type = max::_jit_sym_float32;
    m = static_cast<max::t_object*>(max::jit_object_new(max::_jit_sym_jit_matrix,&dest_info));
    max::object_method(m, max::_jit_sym_frommatrix,matrix,NULL);
    return m;
}
    
void mlmat_assist(void* x, void* b, long io, long index, char* s) ;
max::t_jit_err mlmat_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs);
max::t_object* maxob_from_jitob(max::t_object* job) ;
void max_mlmat_jit_matrix(max_jit_wrapper *x, max::t_symbol *s, short argc, max::t_atom *argv) ;
void mlmat_notify(max::t_object* x, max::t_symbol *s, max::t_symbol *msg, void *sender, void *data);


class mlmat_buffer_griffinlim : public object<mlmat_buffer_griffinlim>, public matrix_operator<>
{
public:
    MIN_DESCRIPTION    { "Reconstructs a time-domain signal in a buffer~ from a magnitude spectrogram, using (momentum-accelerated) Griffin-Lim phase reconstruction. The input is a 1-plane magnitude matrix or the stack's 2-plane spectral matrix." };
    MIN_TAGS        { "audio, analysis" };
    MIN_AUTHOR        { "Todd Ingalls" };
    MIN_RELATED        { "fft~, ifft~, pfft~, jit.fft" };
    MIN_DISCUSSION  {"The input is a magnitude spectrogram, reconstructed into the named buffer~ by iterated analysis and synthesis: each Griffin-Lim iteration synthesizes a candidate signal from the current phase estimate, re-analyzes it, and restores the input magnitudes (optionally blending the previous estimate via <at>momentum</at>, the fast Griffin-Lim formulation). The input may be 1-plane (magnitudes only -- the classic magnitude-only cold start with zero phases) or the stack's 2-plane matrix: with <at>input_polar</at> set, plane 0 and plane 1 hold magnitudes and an initial phase estimate; with rectangular input (the default) the magnitudes are computed as hypot(re, im) and the phase seed as atan2(im, re) -- so <o>mlmat.buffer_stft</o>'s default output connects directly, and an unmodified analysis seeds the iteration with its true phases. <at>fftsize</at> and <at>overlap</at> must match the analysis; the output length is frames * fftsize / overlap samples, written into the buffer~ named by the object's argument (or the set message)."};
    
    max::t_buffer_ref *m_buffer_reference {nullptr};

        
    attribute<int, threadsafe::no, limit::clamp> channel {this, "channel", 1,
        description {"Channel to read from the buffer~. The channel number uses 1-based counting."},
        range {1, buffer_reference::k_max_channels}
    };
    
    attribute<int, threadsafe::no, limit::clamp> fftsize {this, "fftsize", 1024,
        description {"Size of DFT ."},
        range {16, 65536} //what should max be?
    };
    
    attribute<int, threadsafe::no, limit::clamp> overlap {this, "overlap", 4,
        description {"Overlap of windows."},
        range {1, 65536} //what should max be?
    };
    
    attribute<c74::min::symbol> window { this, "window", "hanning",
        description {
            "FFT Window."
        },
        range{"square", "triangle", "hanning", "hamming", "blackman"}
    };
    
    
    attribute<bool> full_spectrum { this, "full_spectrum", false,
        description {
            "If set, output full spectrum."
    }};
    attribute<bool> input_polar { this, "input_polar", false,
        description {
            "If set, the input planes hold magnitudes and phases (the initial phase estimate for the iteration). Otherwise (the default) the input is rectangular: the magnitudes are computed as hypot(re, im) and the phase seed as atan2(im, re), so mlmat.buffer_stft's default output connects directly. A 1-plane input is always magnitudes, regardless of this attribute."
        }
    };
    
    attribute<min::symbol> initial_buffer { this, "initial_buffer", true,
        visibility(visibility::hide)
    };
    
    attribute<double, threadsafe::no, limit::clamp> momentum {this, "momentum", 0.,
        description {"Momentum for the fast Griffin-Lim phase reconstruction. 0 (the default) is the classic algorithm; nonzero values blend each phase update with the previous estimate, which can speed up convergence when using large iteration counts."},
        range {0., 1.}
    };
    
    attribute<int, threadsafe::no, limit::clamp> iters {this, "iters", 25,
        description {"Number of Griffin-Lim iterations."},
        range {1, 255}
    };
    
    // I find I need this to get valid buffer when instatiated in a loaded patch
    queue<> initial_buffer_deferrer { this,
        MIN_FUNCTION {
            set(initial_buffer.get());
            return {};
        }};
    
    queue<> update_buffer { this,
        MIN_FUNCTION {
            buffer_update();
            return {};
    }};
    
    // Using this to get first arg for buffer
    min::message<> maxob_setup {this, "maxob_setup",
        MIN_FUNCTION {
            max::t_object* mob = maxob_from_jitob(min::object_base::maxobj());
            m_dumpoutlet = max_jit_obex_dumpout_get(mob);
            if(!args.empty()) {
                initial_buffer.set(args);
                initial_buffer_deferrer.set();
            }
            return {};
    }};
    
    min::message<> dblclick {this, "dblclick",
        MIN_FUNCTION {
            max::buffer_view(max::buffer_ref_getobject(m_buffer_reference));
            return {};
    }};
    
    min::message<> set {this, "set",
        MIN_FUNCTION {
            if (!m_buffer_reference)
                m_buffer_reference = c74::max::buffer_ref_new(min::object_base::maxobj(), args[0]);
            else
                max::buffer_ref_set(m_buffer_reference, args[0]);
            m_buffer_name = args[0];
            return {};
    }};
    

    
    
    c74::max::t_jit_err matrix_calc(c74::max::t_object* x, c74::max::t_object* inputs, c74::max::t_object* outputs) {
        max::t_jit_err err = max::JIT_ERR_NONE;
        max::t_jit_matrix_info in_minfo;
        auto in_matrix = max::object_method(inputs, max::_jit_sym_getindex, 0);
        long in_savelock = (long) max::object_method(in_matrix, max::_jit_sym_lock, 1);
       
        max::object_method(in_matrix,max::_jit_sym_getinfo, &in_minfo);
        
        size_t b_frame_count = full_spectrum ? (in_minfo.dim[0] * in_minfo.dim[1]) / 2 : (in_minfo.dim[0] * in_minfo.dim[1]);
        size_t output_length = (b_frame_count*2) / overlap;

        max::t_buffer_obj *buffer = max::buffer_ref_getobject(m_buffer_reference);
        max::t_object* in_matrix32 = nullptr;

        // a missing or invalid buffer must not reach sizeinsamps or the
        // sample lock (they dereference it). return directly: nothing
        // but the input matrix is locked at this point
        if (!(buffer && m_buffer_reference)) {
            cerr << "mlmat.buffer_griffinlim: buffer \"" << m_buffer_name
                 << "\" specified is not valid" << endl;
            max::object_method(in_matrix, c74::max::_jit_sym_lock, in_savelock);
            return max::JIT_ERR_INVALID_PTR;
        }
        max::object_method(static_cast<max::t_object*>(buffer), max::gensym("sizeinsamps"), (void*)output_length, 0);

        float *tab = buffer_locksamples(buffer);

        // float32 and float64 inputs are unpacked directly; other types go
        // through a conversion matrix first
        err = (max::t_jit_err)max::object_method(in_matrix, max::_jit_sym_getinfo, &in_minfo);
        max::t_object* src_matrix = static_cast<max::t_object*>(in_matrix);
        if(in_minfo.type != max::_jit_sym_float32 && in_minfo.type != max::_jit_sym_float64) {
            in_matrix32 = convert_to_float32(static_cast<max::t_object*>(in_matrix), in_minfo);
            src_matrix = in_matrix32;
        }
        err = (max::t_jit_err)max::object_method(src_matrix, max::_jit_sym_getinfo, &in_minfo);
        const bool input_is_float64 = (in_minfo.type == max::_jit_sym_float64);

        if(in_minfo.dim[1] != (full_spectrum ? fftsize : fftsize / 2)) {
            int requested_fft = full_spectrum ? fftsize : fftsize / 2;
            cerr << "dim[1] of input matrix is " << in_minfo.dim[1] << " but should be " << requested_fft << " to match requested fftsize." << endl;
            goto out;
        }

        if(in_minfo.planecount != 1 && in_minfo.planecount != 2) {
            cerr << "input matrix has " << in_minfo.planecount << " planes; needs 1 (magnitudes) or 2 (magnitudes + phase seed, or rectangular re/im)." << endl;
            err = c74::max::JIT_ERR_INVALID_INPUT;
            goto out;
        }

        if (buffer && tab) {

            size_t step = fftsize / overlap;
            size_t num_bins = in_minfo.dim[1];

            // scratch members, so a steady-state call allocates nothing.
            // (griffin lim zeroes the sample buffer itself)
            m_samples.resize(b_frame_count+step);
            m_in_real.resize(b_frame_count+step);
            m_in_imag.resize(b_frame_count+step);
            const string window_string = window.get().c_str();

            /* write fft frames to vector: m_in_real takes the
               magnitudes, m_in_imag the initial phase seed. a 1-plane
               input is the magnitude-only cold start (zero phases);
               a polar input passes through; a rectangular input is
               converted (magnitudes = hypot, seed = atan2), so the
               default buffer_stft chain works with no configuration */
            max::uchar *dataptr = nullptr;
            max::uchar *ip = nullptr;

            err = (max::t_jit_err)max::object_method(src_matrix, max::_jit_sym_getdata, &dataptr);

            // polar-input heuristic (one warning per object instance):
            // phases are bounded by pi by definition, so a
            // rect-configured resynthesizer receiving a 2-plane matrix
            // whose plane-1 values all fit in [-pi, pi] while plane 0
            // carries meaningful magnitudes is almost certainly being
            // fed a magnitude/phase matrix -- the unpack then destroys
            // every phase and the buffer fills with noise
            if (!m_polar_warned && !input_polar) {
                double p0max = 0.0, p1max = 0.0;
                for (size_t k = 0; k < num_bins; ++k) {
                    const max::uchar* ip = dataptr + static_cast<size_t>(k) * in_minfo.dimstride[1];
                    double a, b;
                    if (input_is_float64) {
                        a = reinterpret_cast<const double*>(ip)[0];
                        b = reinterpret_cast<const double*>(ip)[1];
                    } else {
                        a = static_cast<double>(reinterpret_cast<const float*>(ip)[0]);
                        b = static_cast<double>(reinterpret_cast<const float*>(ip)[1]);
                    }
                    p0max = std::max(p0max, std::fabs(a));
                    p1max = std::max(p1max, std::fabs(b));
                }
                if (p1max <= 3.15 && p0max > 1.0) {
                    m_polar_warned = true;
                    cerr << "mlmat.buffer_griffinlim: the input looks like a POLAR (magnitude/phase) matrix but the object expects rectangular -- every phase would be misread and the output is noise. Set @input_polar 1, or make the analysis output rectangular." << endl;
                }
            }


            size_t buf_pos = 0;
            const bool one_plane = (in_minfo.planecount == 1);

            for(auto width=0;width<in_minfo.dim[0];width++) {
                ip = dataptr + (width*in_minfo.dimstride[0]);
                for(auto height=0;height<num_bins;height++) {
                    double p0, p1;
                    if(input_is_float64) {
                        double *fp = (double*)ip;
                        p0 = fp[0];
                        p1 = one_plane ? 0.0 : fp[1];
                    } else {
                        float *fp = (float*)ip;
                        p0 = fp[0];
                        p1 = one_plane ? 0.0 : static_cast<double>(fp[1]);
                    }
                    if (one_plane) {
                        m_in_real[buf_pos] = static_cast<float>(p0);
                        m_in_imag[buf_pos++] = 0.0f;
                    } else if (input_polar) {
                        m_in_real[buf_pos] = static_cast<float>(p0);
                        m_in_imag[buf_pos++] = static_cast<float>(p1);
                    } else {
                        m_in_real[buf_pos] = static_cast<float>(std::sqrt(p0 * p0 + p1 * p1));
                        m_in_imag[buf_pos++] = static_cast<float>(std::atan2(p1, p0));
                    }
                    ip = ip + in_minfo.dimstride[1];
                }
            }

            try {
                griffinlim.process(m_in_real, m_in_imag, b_frame_count, m_samples, fftsize, overlap, window_string, iters, momentum, full_spectrum);
            } catch ( const std::invalid_argument& e ) {
                cerr << e.what() << endl;
                goto out;
            }
            max::buffer_setdirty(buffer);
            update_buffer.set();
        } else {
           cerr << "buffer specified is not valid" << endl;
           goto out;
        }

    out:
        if (buffer) {
            buffer_unlocksamples(buffer);
        }
        max::object_method(in_matrix, c74::max::_jit_sym_lock, in_savelock);
        if(in_matrix != in_matrix32) { jit_object_free(in_matrix32); }
        return err;

    }

    max::t_max_err notify(max::t_object* x, max::t_symbol *s, max::t_symbol *msg, void *sender, void *data) {
        return max::buffer_ref_notify(m_buffer_reference, s, msg, sender, data);
    }
    
    template<typename matrix_type>
    matrix_type calc_cell(matrix_type input, const min::matrix_info& info, min::matrix_coord& position) {
        matrix_type output;
        return output;
    }
    
    ~mlmat_buffer_griffinlim() {

        
        if(m_buffer_reference) {
            object_free(m_buffer_reference);
        }

    }

protected:
    void* m_dumpoutlet { nullptr };
    min::symbol m_buffer_name;
    
    
private:
    
    message<> buffer_update {this, "buffer_update",
        MIN_FUNCTION {
            max::t_buffer_obj *buffer = max::buffer_ref_getobject(m_buffer_reference);
            if (!(buffer && m_buffer_reference)) {
                return {};
            }
            float *tab = buffer_locksamples(buffer);
            size_t b_frame_count = max::buffer_getframecount(buffer);
            max::t_atom_long b_channel_count = max::buffer_getchannelcount(buffer);
            
            if (buffer && tab) {
                auto chan = std::min<size_t>(channel - 1, b_channel_count);
                for(auto i = 0;i<b_frame_count;i++) {
                    tab[(i*b_channel_count) + chan] = m_samples[i];
                }
                max::buffer_setdirty(buffer);
            }
            max::buffer_unlocksamples(buffer);
            return {};
    }};
    
    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        c74::max::t_class* c = args[0];
        //auto x = c74::min::object_base::maxobj();
        max_jit_class_mop_wrap(c, this_jit_class, 0);
        max_jit_class_wrap_standard(c, this_jit_class, 0);
        class_addmethod(c, (c74::max::method)mlmat_assist, "assist", c74::max::A_CANT, 0);
        return {};
    }};
    
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        // add mop
        max::t_object* mop = static_cast<max::t_object*>(max::jit_object_new(max::_jit_sym_jit_mop, 1, 0));
        jit_mop_single_type(mop, max::_jit_sym_float32);
        jit_class_addadornment(c, mop);
        jit_class_addmethod(c, (max::method)mlmat_matrix_calc, "matrix_calc", max::A_CANT, 0);
        jit_class_addmethod(c, (max::method)mlmat_notify, "notify", max::A_CANT, 0); //cant get notify to work unless i do this
        return {};
    }};
//
    // making custom mop so max_jit_mop_matrix_args does not get called as don't need these
    message<> mop_setup {this, "mop_setup", MIN_FUNCTION {
        max::t_jit_err err = max::JIT_ERR_NONE;

        // max_jit_mop_new() appends the max wrapper pointer as the final
        // argument of this message, and inside the jit object maxobj() is
        // the jit object itself. Link the two first: this is the canonical
        // custom-mop sequence (see max.jit.mop.h), and it is also what
        // stores the "maxwrapper" key that maxob_from_jitob() relies on
        // elsewhere (e.g. in maxob_setup). Without it the wrapper and the
        // jit object are never connected: no outlets are created, no
        // attribute getting/setting is forwarded from the box, and
        // jit_error_code() during creation crashes on the null doc.
        max::t_object* x = args.empty()
            ? maxob_from_jitob(min::object_base::maxobj())
            : static_cast<max::t_object*>(args.back());
        max::t_object* o = min::object_base::maxobj();
        max::max_jit_obex_jitob_set(x, o);

        max::max_jit_obex_dumpout_set(x,outlet_new(x,NULL));
        err = max::max_jit_mop_setup(x);
        if(err != max::JIT_ERR_NONE) {
            (cerr << "mop_setup: max_jit_mop_setup error " << err << endl);
        }

        err = max::max_jit_mop_inputs(x);
        if(err != max::JIT_ERR_NONE) {
            (cerr << "mop_setup: max_jit_mop_inputs error " << err << endl);
        }

        err = max::max_jit_mop_outputs(x);
        if(err != max::JIT_ERR_NONE) {
            (cerr << "mop_setup: max_jit_mop_outputs error " << err << endl);
        }
        return {};
    }};

    // one-shot polar-input warning
    bool m_polar_warned = false;
    std::vector<float> m_samples;
    std::vector<float> m_in_real;
    std::vector<float> m_in_imag;
    GriffinLim griffinlim;
};


MIN_EXTERNAL(mlmat_buffer_griffinlim);

void mlmat_notify(c74::max::t_object* x, c74::max::t_symbol *s, c74::max::t_symbol *msg, void *sender, void *data)
{
    minwrap<mlmat_buffer_griffinlim>* job = (minwrap<mlmat_buffer_griffinlim>*)(x);
    job->m_min_object.notify(x, s, msg, sender, data);
}

c74::max::t_jit_err mlmat_matrix_calc(c74::max::t_object* x, c74::max::t_object* inputs, c74::max::t_object* outputs) {
    c74::max::t_jit_err err = c74::max::JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return c74::max::JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_buffer_griffinlim>* job = (minwrap<mlmat_buffer_griffinlim>*)(x);
        
        
        
        err = job->m_min_object.matrix_calc(x, inputs, outputs);
    }
    return err;
}


c74::max::t_object* maxob_from_jitob(c74::max::t_object* job) {
    c74::max::t_object* mwrap = NULL;
    c74::max::object_obex_lookup(job, c74::max::gensym("maxwrapper"), &mwrap);
    
    return mwrap;
}

void mlmat_assist(void* x, void* b, long io, long index, char* s) {
   switch(io) {
       case 1:
           switch(index) {
               case 0:
                   snprintf(s, 64, "messages");
                   break;
               default:
                   break;
           }
           break;
       case 2:
           switch(index) {
               case 0:
                   snprintf(s, 64, "dumpout");
                   break;

               default:
                   snprintf(s, 64, "dumpout");
                   break;
           }

   }
}

