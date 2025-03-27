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

/*
*/

max::t_object* maxob_from_jitob(max::t_object* job) {
    max::t_object* mwrap = NULL;
    object_obex_lookup(job, max::gensym("maxwrapper"), &mwrap);
    return mwrap;
}

max::t_object* convert_to_float64(max::t_object *matrix, max::t_jit_matrix_info& minfo) {
    max::t_object *m; // destination matrix
    if(minfo.type == max::_jit_sym_float64) {
        return matrix;
    }
    max::t_jit_matrix_info dest_info = minfo;
    
    dest_info.type = max::_jit_sym_float64;
    m = static_cast<max::t_object*>(max::jit_object_new(max::_jit_sym_jit_matrix,&dest_info));
    max::object_method(m, max::_jit_sym_frommatrix,matrix,NULL);
    return m;
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
            
            size_t step = fftsize/overlap;
            size_t out_len = (((b_frame_count * overlap) + fftsize - 1) / fftsize) * fftsize;
            size_t num_fft_frames = out_len / fftsize;
            size_t fftsize_2 = fftsize / 2;
            
            out_minfo.planecount = 2;
            out_minfo.dimcount = 2;
            out_minfo.dim[0] = num_fft_frames;
            out_minfo.type = max::_jit_sym_float32;

            if(full_spectrum) {
                out_minfo.dim[1] = fftsize ;
            } else {
                out_minfo.dim[1] = fftsize_2;
            }
        
            vDSP_DFT_Setup tmp_setup = m_setup;

            m_setup = vDSP_DFT_zrop_CreateSetup(tmp_setup, fftsize, vDSP_DFT_FORWARD );
            if(m_setup == 0) {
                cerr << "fftsize " << fftsize << " is not valid." << endl;
                goto out;
            }
            
            if(tmp_setup != NULL) {
                vDSP_DFT_DestroySetup(tmp_setup);
            }
            
            std::vector<float> in_samples(fftsize);
            std::vector<float> in_real(fftsize_2);
            std::vector<float> in_imag(fftsize_2, 0.0f);
            std::vector<float> out_real(fftsize_2);
            std::vector<float> out_imag(fftsize_2);
            std::vector<float> window_vector(fftsize);
            std::vector<float> magnitudes(fftsize_2);
            std::vector<float> phases(fftsize_2);
            std::vector<float> out_magnitudes;
            std::vector<float> out_phases;
            
            /* window */
            
            const string window_string = window.get().c_str();
            
            if(window_string == "triangle") {
                float tristart = 0.0f;
                float tritop = 1.0f;
                float up = 1.0f/fftsize_2;
                float down = -1.0f/fftsize_2;
                vDSP_vramp(&tristart, &up,window_vector.data(), 1, fftsize/2);
                vDSP_vramp(&tritop, &down,window_vector.data()+((fftsize/2)-1), 1, fftsize/2);
            } else if (window_string == "hanning") {
                vDSP_hann_window(window_vector.data(), fftsize, vDSP_HANN_NORM );
            } else if (window_string == "hamming") {
                vDSP_hamm_window(window_vector.data(), fftsize, 0);
            } else if (window_string == "blackman") {
                vDSP_blkman_window(window_vector.data(), fftsize, 0);
            }
            
            DSPSplitComplex complex_input = { in_real.data(), in_imag.data()};
            DSPSplitComplex complex_output = { out_real.data(), out_imag.data()};
            
            
            for (auto i = 0; i < b_frame_count; i+=step) {
                //get next N samples
                if(i+fftsize > b_frame_count) {
                    for(auto j=0;j<fftsize;j++) {
                        if((i+j) > b_frame_count) {
                            in_samples[j] = 0.0f;
                        } else {
                            in_samples[j] = tab[(i+j)*b_channel_count + chan];
                        }
                    }
                }
                else {
                    for(auto j=0;j<fftsize;j++) {
                        in_samples[j] = tab[(i+j)*b_channel_count + chan];
                    }
                }
                //window
                if (window_string != "square") {
                    vDSP_vmul(in_samples.data(), 1, window_vector.data(), 1, in_samples.data(), 1, fftsize);
                }
                
                //put into SplitComplex
                vDSP_ctoz((DSPComplex *)in_samples.data(), 2, &complex_input, 1, fftsize_2);

                // dft
                vDSP_DFT_Execute(m_setup,complex_input.realp, complex_input.imagp,complex_output.realp,complex_output.imagp);
            

                float scale = .5;
            
                vDSP_vsmul(complex_output.realp, 1, &scale, complex_output.realp, 1, fftsize_2);
                vDSP_vsmul(complex_output.imagp, 1, &scale, complex_output.imagp, 1, fftsize_2);
         
            
                complex_output.imagp[0] = 0.0;
                
                if(output_polar) {
                    
                    vDSP_zvabs(&complex_output, 1, magnitudes.data(), 1, fftsize_2);  // Compute magnitude
                    vDSP_zvphas(&complex_output, 1, phases.data(), 1, fftsize_2);  // Compute phase (radians)
                    std::copy(magnitudes.begin(), magnitudes.end(), std::back_inserter(out_magnitudes));
                    std::copy(phases.begin(), phases.end(), std::back_inserter(out_phases));
                    if(full_spectrum) {
                        std::copy(magnitudes.rbegin(), magnitudes.rend(), std::back_inserter(out_magnitudes));
                        std::copy(phases.rbegin(), phases.rend(), std::back_inserter(out_phases));
                    }
                
                } else {
                    std::copy(out_real.begin(), out_real.end(), std::back_inserter(out_magnitudes));
                    std::copy(out_imag.begin(), out_imag.end(), std::back_inserter(out_phases));
                    if(full_spectrum) {
                        std::copy(out_real.rbegin(), out_real.rend(), std::back_inserter(out_magnitudes));
                        std::copy(out_imag.rbegin(), out_imag.rend(), std::back_inserter(out_phases));
                    }
                }
            }
            
            max::t_object* tmp_matrix = static_cast<max::t_object*>(max::jit_object_new(max::_jit_sym_jit_matrix,&out_minfo));
            
            if(!tmp_matrix) {
                (std::cerr << "could not create matrix" << std::endl);
                return err;
            }
            
            c74::max::uchar *dataptr = nullptr;
            c74::max::uchar *ip = nullptr;
            float *fp = nullptr;
            err = (max::t_jit_err)max::object_method(tmp_matrix, max::_jit_sym_getinfo, &out_minfo);
            err = (max::t_jit_err)max::object_method(tmp_matrix, max::_jit_sym_getdata, &dataptr);
            
            long buf_pos = 0 ;

            for(auto width=0;width<out_minfo.dim[0];width++) {
                ip = dataptr + (width*out_minfo.dimstride[0]);
                for(auto height=0;height<out_minfo.dim[1];height++) {
                    fp = (float*)ip;
                    *fp++ = out_magnitudes[buf_pos];
                    *fp++ = out_phases[buf_pos++];
                    ip = ip + out_minfo.dimstride[1];
                }
            }
            // not sure. seems easier to do fft in float32 since that is what buffer samples in and then convert to float64 matrix
            max::t_object* out_matrix64 = convert_to_float64(static_cast<max::t_object*>(tmp_matrix), out_minfo);
            err = (max::t_jit_err)max::object_method(out_matrix64, max::_jit_sym_getinfo, &out_minfo);
            max::object_method(out_matrix, max::_jit_sym_setinfo,&out_minfo);
            max::object_method(out_matrix, max::_jit_sym_frommatrix,out_matrix64,nullptr);
            max::jit_object_free(tmp_matrix);
            max::jit_object_free(out_matrix64);


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
