/// @file mlmat.buffer_fft.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved. Also based on examples provided with the mlpack library. Please see source/mlpack for license details
/// @license  Use of this source code is governed by the MIT License found in the License.md file.


#include <Accelerate/Accelerate.h>
#include "c74_min.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>
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


class mlmat_buffer_istft : public object<mlmat_buffer_istft>, public matrix_operator<>
{
public:
    MIN_DESCRIPTION    { "Read from a buffer~." };
    MIN_TAGS        { "audio, sampling" };
    MIN_AUTHOR        { "Cycling '74" };
    MIN_RELATED        { "index~, buffer~, wave~" };
    
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
            "If set, output polar."
    }};
    
    attribute<min::symbol> initial_buffer { this, "initial_buffer", true,
        visibility(visibility::hide)
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
        
        max::object_method(static_cast<max::t_object*>(buffer), max::gensym("sizeinsamps"), (void*)output_length, 0);
    
        float *tab = buffer_locksamples(buffer);
    
        err = (max::t_jit_err)max::object_method(in_matrix, max::_jit_sym_getinfo, &in_minfo);
        
        max::t_object* in_matrix32 = convert_to_float32(static_cast<max::t_object*>(in_matrix), in_minfo);
        
        err = (max::t_jit_err)max::object_method(in_matrix32, max::_jit_sym_getinfo, &in_minfo);
        
        // what to do if error

        // need to check fftsize to matrrix dims
        
        
        if (buffer && tab) {

            max::t_atom_long b_channel_count = max::buffer_getchannelcount(buffer);
            size_t chan = std::min<size_t>(channel - 1, b_channel_count);
        
            size_t step = fftsize / overlap;
            size_t fftsize_2 = fftsize / 2;
            
            std::vector<float> in_real(b_frame_count+step);
            std::vector<float> in_imag(b_frame_count+step);
            
            std::vector<float> frame_real(fftsize_2, 0.0f);
            std::vector<float> frame_imag(fftsize_2, 0.0f);

            m_samples.resize(b_frame_count+step);
            m_samples.assign(b_frame_count+step, 0.0f);
            std::vector<float>signal(fftsize);
            std::vector<float> window_vector(fftsize);
            
            
            size_t num_bins = fftsize_2;
            
            /* window */
            
            const string window_string = window.get().c_str();
            
            if(window_string == "triangle") {
                float tristart = 0.0f;
                float tritop = 1.0f;
                float up = 1.0f/fftsize_2;
                float down = -1.0f/fftsize_2;
                vDSP_vramp(&tristart, &up,window_vector.data(), 1, fftsize_2);
                vDSP_vramp(&tritop, &down,window_vector.data()+((fftsize/2)-1), 1, fftsize_2);
            } else if (window_string == "hanning") {
                vDSP_hann_window(window_vector.data(), fftsize, vDSP_HANN_NORM );
            } else if (window_string == "hamming") {
                vDSP_hamm_window(window_vector.data(), fftsize, 0);
            } else if (window_string == "blackman") {
                vDSP_blkman_window(window_vector.data(), fftsize, 0);
            }
            
            vDSP_DFT_Setup tmp_setup = m_setup; m_setup = vDSP_DFT_zrop_CreateSetup(tmp_setup, fftsize, vDSP_DFT_INVERSE );
            if(m_setup == 0) {
                cerr << "fftsize " << fftsize << " is not valid." << endl;
                goto out;
            }
            
            if(tmp_setup != NULL) {
                vDSP_DFT_DestroySetup(tmp_setup);
            }
            
            // if full_spectrum just going to get half of matrix
            
            
            DSPSplitComplex complex_input = { in_real.data(), in_imag.data()};
            DSPSplitComplex complex_output = { frame_real.data(), frame_imag.data()};
    
            
            /* write fft frames to vector*/
            max::uchar *dataptr = nullptr;
            max::uchar *ip = nullptr;
            float *fp = nullptr;
            
            err = (max::t_jit_err)max::object_method(in_matrix32, max::_jit_sym_getdata, &dataptr);

            long buf_pos = 0;

           // input_polar
           
            
            if(input_polar) {
                float r = 0.0;
                float theta = 0.0;
                for(auto width=0;width<in_minfo.dim[0];width++) {
                    ip = dataptr + (width*in_minfo.dimstride[0]);
                    for(auto height=0;height<num_bins;height++) {
                        fp = (float*)ip;
                        r = *fp++;
                        theta = *fp++;
                        in_real[buf_pos] = r * cos(theta);
                        in_imag[buf_pos++] = r * sin(theta);
                        ip = ip + in_minfo.dimstride[1];
                    }
                }
                
            } else {
                for(auto width=0;width<in_minfo.dim[0];width++) {
                    ip = dataptr + (width*in_minfo.dimstride[0]);
                    for(auto height=0;height<num_bins;height++) {
                        fp = (float*)ip;
                        in_real[buf_pos] = *fp++;
                        in_imag[buf_pos++] = *fp++;
                        ip = ip + in_minfo.dimstride[1];
                    }
                }
            }
            float scaling = 1.0f / fftsize / overlap;
            
            for (auto i = 0, j=0; i < (b_frame_count+step); i+=fftsize_2, j+=step) {
                
                vDSP_DFT_Execute(m_setup,complex_input.realp+i, complex_input.imagp+i,complex_output.realp,complex_output.imagp);

                vDSP_ztoc(&complex_output, 1, ( DSPComplex *)signal.data(), 2,fftsize_2);
                
                vDSP_vsmul(signal.data(), 1, &scaling, signal.data(), 1, fftsize);
                
                if (window_string != "square") {
                    vDSP_vmul(signal.data(), 1, window_vector.data(), 1, signal.data(), 1, fftsize);
                }
                
                
                
                vDSP_vadd(signal.data(), 1, m_samples.data()+j, 1, m_samples.data()+j, 1, fftsize);
                
            }
            
            max::buffer_setdirty(buffer);
            update_buffer.set();
        } else {
           cerr << "buffer specified is not valid" << endl;
           goto out;
        }
        
    out:
        buffer_unlocksamples(buffer);
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
    
    ~mlmat_buffer_istft() {
        if(m_setup != NULL) {
            vDSP_DFT_DestroySetup(m_setup);
        }
        
        if(m_buffer_reference) {
            object_free(m_buffer_reference);
        }

    }

protected:
    void* m_dumpoutlet { nullptr };
    
    
private:
    
    message<> buffer_update {this, "buffer_update",
        MIN_FUNCTION {
            max::t_buffer_obj *buffer = max::buffer_ref_getobject(m_buffer_reference);
            float *tab = buffer_locksamples(buffer);
            size_t b_frame_count = max::buffer_getframecount(buffer);
            max::t_atom_long b_channel_count = max::buffer_getchannelcount(buffer);
            
            if (buffer && tab) {
                
//                if(b_frame_count != m_samples.size()) {
//                    cerr << "b_frame_count != m_samples.size()" << endl;
//                    return {};
//                }
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
        auto x = c74::min::object_base::maxobj();
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
        max::t_object* x = maxob_from_jitob(min::object_base::maxobj());

        max::max_jit_obex_dumpout_set(x,outlet_new(x,NULL));
        err = max::max_jit_mop_setup(x);
        if(err != max::JIT_ERR_NONE) {
            jit_error_code(x,err);
        }

        err = max::max_jit_mop_inputs(x);
        if(err != max::JIT_ERR_NONE) {
            jit_error_code(x,err);
        }

        err = max::max_jit_mop_outputs(x);
        if(err != max::JIT_ERR_NONE) {
            jit_error_code(x,err);
        }
        return {};
    }};

    
    vDSP_DFT_Setup m_setup = NULL;
    std::vector<float> m_samples;

};


MIN_EXTERNAL(mlmat_buffer_istft);

void mlmat_notify(c74::max::t_object* x, c74::max::t_symbol *s, c74::max::t_symbol *msg, void *sender, void *data)
{
    minwrap<mlmat_buffer_istft>* job = (minwrap<mlmat_buffer_istft>*)(x);
    job->m_min_object.notify(x, s, msg, sender, data);
}

c74::max::t_jit_err mlmat_matrix_calc(c74::max::t_object* x, c74::max::t_object* inputs, c74::max::t_object* outputs) {
    c74::max::t_jit_err err = c74::max::JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return c74::max::JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_buffer_istft>* job = (minwrap<mlmat_buffer_istft>*)(x);
        
        
        
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
                   sprintf(s, "messages");
                   break;
               default:
                   break;
           }
           break;
       case 2:
           switch(index) {
               case 0:
                   sprintf(s, "(matrix) data");
                   break;

               default:
                   sprintf(s, "dumpout");
                   break;
           }

   }
}

