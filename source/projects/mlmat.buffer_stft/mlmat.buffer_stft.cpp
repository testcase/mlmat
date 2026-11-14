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

/*
*/





max::t_object* maxob_from_jitob(max::t_object* job) {
    max::t_object* mwrap = NULL;
    object_obex_lookup(job, max::gensym("maxwrapper"), &mwrap);
    return mwrap;
}


void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
max::t_jit_err mlmat_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs);
void mlmat_outputmatrix(max_jit_wrapper *x);
void mlmat_notify(max::t_object* x, max::t_symbol *s, max::t_symbol *msg, void *sender, void *data);

class mlmat_buffer_stft : public object<mlmat_buffer_stft>, public matrix_operator<>
{
public:
    MIN_DESCRIPTION    { "Run STFT on a channel of a buffer~." };
    MIN_TAGS        { "audio, analysis" };
    MIN_AUTHOR        { "Todd Ingalls" };
    MIN_RELATED        { "fft~, ifft~, pfft~, jit.fft" };

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

    attribute<min::symbol> window { this, "window", "hanning",
        description {
            "FFT Window."
        },
        range{"square", "triangle", "hanning", "hamming", "blackman"}
    };
    
    
    attribute<bool> full_spectrum { this, "full_spectrum", false,
        description {
            "If set, output full spectrum."
        }
    };
    
    attribute<bool> output_polar { this, "output_polar", false,
        description {
            "If set, output polar."
        }
    };
    
    attribute<min::symbol> initial_buffer { this, "initial_buffer", true,
        visibility(visibility::hide)
    };
    
    // I find I need this to get valid buffer when instatiated in a loaded patch
    queue<> initial_buffer_deferrer { this,
        MIN_FUNCTION {
            set(initial_buffer.get());
            return {};
        }
    };
    
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
            return {};
        }};
    
    //
    max::t_jit_err matrix_calc(max::t_object* x, max::t_object* out_matrix) {
        max::t_jit_err err = max::JIT_ERR_NONE;
        max::t_jit_matrix_info out_minfo;
        auto out_savelock = max::object_method(out_matrix, max::_jit_sym_lock, (void*)1);

        max::t_buffer_obj *buffer = max::buffer_ref_getobject(m_buffer_reference);
        float *tab = buffer_locksamples(buffer);

        if (buffer && tab) {
            size_t b_frame_count = max::buffer_getframecount(buffer);
            max::t_atom_long b_channel_count = max::buffer_getchannelcount(buffer);
            size_t chan = std::min<size_t>(channel - 1, b_channel_count);

            size_t num_fft_frames = (b_frame_count * overlap + fftsize - 1) / fftsize;
            size_t spec_dim = full_spectrum ? fftsize : (fftsize / 2);

            out_minfo.type = max::_jit_sym_float64;
            out_minfo.planecount = 2;
            out_minfo.dimcount = 2;
            out_minfo.dim[0] = num_fft_frames;
            out_minfo.dim[1] = spec_dim;

            // scratch members, so a steady-state bang allocates nothing
            m_input.resize(b_frame_count);
            m_out_magnitudes.resize(num_fft_frames * spec_dim);
            m_out_phases.resize(num_fft_frames * spec_dim);

            for(size_t i=0;i<b_frame_count;i++) {
                m_input[i] = tab[i*b_channel_count + chan];
            }

            const string window_string = window.get().c_str();

            try {
                stft.process(m_input, m_out_magnitudes, m_out_phases, fftsize, overlap, window_string, output_polar, full_spectrum);
            } catch ( const std::invalid_argument& e ) {
                cerr << e.what() << endl;
                goto out;
            }

            // resize the destination and write the analysis directly into
            // it, converting float32 -> float64 in the same pass (no
            // temporary matrices)
            max::object_method(out_matrix, max::_jit_sym_setinfo, &out_minfo);
            max::t_jit_matrix_info dest_minfo;
            c74::max::uchar *dataptr = nullptr;
            max::object_method(out_matrix, max::_jit_sym_getinfo, &dest_minfo);
            max::object_method(out_matrix, max::_jit_sym_getdata, &dataptr);

            if (dataptr != nullptr) {
                size_t buf_pos = 0;
                for(auto width=0;width<dest_minfo.dim[0];width++) {
                    c74::max::uchar *ip = dataptr + (width*dest_minfo.dimstride[0]);
                    for(auto height=0;height<dest_minfo.dim[1];height++) {
                        double *fp = (double*)ip;
                        fp[0] = m_out_magnitudes[buf_pos];
                        fp[1] = m_out_phases[buf_pos++];
                        ip = ip + dest_minfo.dimstride[1];
                    }
                }
            }
        }
    out:
        buffer_unlocksamples(buffer);
        max::object_method(out_matrix, max::_jit_sym_lock, out_savelock);

        return err;

    }
    
    max::t_max_err notify(max::t_object* x, max::t_symbol *s, max::t_symbol *msg, void *sender, void *data) {
        return max::buffer_ref_notify(m_buffer_reference, s, msg, sender, data);
    }
    
    // need this for matrix_operator . Does nothing here
    template<typename matrix_type>
    matrix_type calc_cell(matrix_type input, const min::matrix_info& info, min::matrix_coord& position) {
        matrix_type output;
        return output;
    }

    
    ~mlmat_buffer_stft() {
        if(m_buffer_reference) {
            object_free(m_buffer_reference);
        }
    }

protected:
    void* m_dumpoutlet { nullptr };
    
private:
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        // add mop
        max::t_object* mop = static_cast<max::t_object*>(max::jit_object_new(max::_jit_sym_jit_mop, -1, 1));
        jit_mop_single_type(mop, max::_jit_sym_float64);
        jit_class_addadornment(c, mop);
        jit_class_addmethod(c, (max::method)mlmat_matrix_calc, "matrix_calc", max::A_CANT, 0);
        jit_class_addmethod(c, (max::method)mlmat_notify, "notify", max::A_CANT, 0); //cant get notify to work unless i do this
        return {};
    }};


    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        char outputmatrix[] = "outputmatrix"; //to avoid warning about c++11 not allowing conversion from string literal to 'char*' when passing to max_jit_class_addmethod_usurp_low
        long flags = max::MAX_JIT_MOP_FLAGS_OWN_OUTPUTMATRIX | max::MAX_JIT_MOP_FLAGS_OWN_JIT_MATRIX;
        max_jit_class_mop_wrap(c, this_jit_class, flags);
        max_jit_class_wrap_standard(c, this_jit_class, 0);
        class_addmethod(c, (max::method)mlmat_assist, "assist", max::A_CANT, 0);
       
        max_jit_class_addmethod_usurp_low(c, (max::method)mlmat_outputmatrix,outputmatrix);
        return {};
    }};
    
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
    
    STFT stft;
    std::vector<float> m_input;
    std::vector<float> m_out_magnitudes;
    std::vector<float> m_out_phases;
};


MIN_EXTERNAL(mlmat_buffer_stft);


void mlmat_notify(max::t_object* x, max::t_symbol *s, max::t_symbol *msg, void *sender, void *data)
{
    minwrap<mlmat_buffer_stft>* job = (minwrap<mlmat_buffer_stft>*)(x);
    job->m_min_object.notify(x, s, msg, sender, data);
}

max::t_jit_err mlmat_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs) {
    max::t_jit_err err = max::JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return max::JIT_ERR_INVALID_PTR;
    else {
        auto out_matrix = (max::t_object*)max::object_method(outputs, max::_jit_sym_getindex, 0);
        if(out_matrix) {
            minwrap<mlmat_buffer_stft>* job = (minwrap<mlmat_buffer_stft>*)(x);
            err = job->m_min_object.matrix_calc(x, out_matrix);
        } else {
            return max::JIT_ERR_INVALID_PTR;
        }
    }
    return err;
}

// this method is called on a 'outputmatrix' or 'bang' message
void mlmat_outputmatrix(max_jit_wrapper *x) {
    long outputmode = c74::max::max_jit_mop_getoutputmode(x);
    // get the reference to the max-wrapped jitter object
    max::t_object *mop = (max::t_object*)max_jit_obex_adornment_get(x, max::_jit_sym_jit_mop);
    max::t_jit_err err;

    if (outputmode && mop) { //always output unless output mode is none
        if (outputmode==1) {

            err = (max::t_jit_err)c74::max::object_method((c74::max::t_object*)c74::max::max_jit_obex_jitob_get(x), max::_jit_sym_matrix_calc, max::object_method(mop, max::_jit_sym_getinputlist), max::object_method(mop, max::_jit_sym_getoutputlist)
            );
            if(err) {
                max::jit_error_code(x,err);
            }
            else {
                max::max_jit_mop_outputmatrix(x);
            }
        }
        else {
            max::max_jit_mop_outputmatrix(x);
        }
    }
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
                   snprintf(s, 64, "(matrix) data");
                   break;

               default:
                   snprintf(s, 64, "dumpout");
                   break;
           }

   }
}
