/// @file mlmat.buffer_fft.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved. Also based on examples provided with the mlpack library. Please see source/mlpack for license details
/// @license  Use of this source code is governed by the MIT License found in the License.md file.


#include <Accelerate/Accelerate.h>
#include "c74_min.h"
//#include "c74_min_logger.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>
using namespace c74;
using namespace c74::min;

/*
*/

c74::max::t_object* maxob_from_jitob(c74::max::t_object* job) {
    c74::max::t_object* mwrap = NULL;
    object_obex_lookup(job, c74::max::gensym("maxwrapper"), &mwrap);
    return mwrap;
}

void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
c74::max::t_jit_err mlmat_matrix_calc(c74::max::t_object* x, c74::max::t_object* inputs, c74::max::t_object* outputs);
void mlmat_outputmatrix(max_jit_wrapper *x);

class mlmat_buffer_stft : public object<mlmat_buffer_stft>, public matrix_operator<>
{
public:
    MIN_DESCRIPTION    { "Read from a buffer~." };
    MIN_TAGS        { "audio, sampling" };
    MIN_AUTHOR        { "Cycling '74" };
    MIN_RELATED        { "index~, buffer~, wave~" };

    buffer_reference m_buffer { this};

    attribute<c74::min::symbol> buffer { this, "buffer", "",
        setter { MIN_FUNCTION {
            m_buffer.set(args[0]);
            return args;
        }}
    };

    attribute<int, threadsafe::no, limit::clamp> channel {this, "channel", 1,
        description {"Channel to read from the buffer~. The channel number uses 1-based counting."},
        range {1, buffer_reference::k_max_channels}
    };

    attribute<int, threadsafe::no, limit::clamp> fftsize {this, "fftsize", 1024,
        description {"fftsize."},
        range {16, 65536} //what should max be?
    };
    
    attribute<int, threadsafe::no, limit::clamp> overlap {this, "overlap", 4,
        description {"overlap."},
        range {1, 65536} //what should max be?
    };

    attribute<c74::min::symbol> window { this, "window", "hanning",
        description {
            "fft window."
        },
        range{"square", "triangle", "hanning", "hamming", "blackman"}
    };
    
    
    attribute<bool> full_spectrum { this, "full_spectrum", true,
        description {
            "If set, output full spectrum."
        }
    };
    
    attribute<bool> output_polar { this, "output_polar", true,
        description {
            "If set, output polar."
        }
    };
    
    c74::min::message<> maxob_setup {this, "maxob_setup",
        MIN_FUNCTION {
            c74::max::t_object* mob = maxob_from_jitob(c74::min::object_base::maxobj());
            m_dumpoutlet = max_jit_obex_dumpout_get(mob);
            return {};
        }};
    
    template<typename matrix_type>
    matrix_type calc_cell(matrix_type input, const c74::min::matrix_info& info, c74::min::matrix_coord& position) {
        matrix_type output;
        
        return output;
    }
    
    c74::max::t_jit_err matrix_calc(c74::max::t_object* x, c74::max::t_object* out_matrix) {
        c74::max::t_jit_err err = c74::max::JIT_ERR_NONE;
        c74::max::t_jit_matrix_info out_minfo;
        auto out_savelock = c74::max::object_method(out_matrix, c74::max::_jit_sym_lock, (void*)1);
        c74::min::buffer_lock<> b(m_buffer);
        auto  chan = std::min<size_t>(channel - 1, b.channel_count());
 
        
        if (b.valid()) {
            long step = fftsize/overlap;
            long out_len = b.frame_count() * overlap;
            
            // 16384
            
            out_minfo.planecount = 2;
            out_minfo.dimcount = 2;
            out_minfo.dim[0] = out_len / fftsize;// b.frame_count();
            
            if(full_spectrum) {
                out_minfo.dim[1] = fftsize ;
            } else {
                out_minfo.dim[1] = fftsize / 2;
            }
            
            out_minfo.type = c74::max::_jit_sym_float32;
    
            vDSP_DFT_Setup tmp_setup = m_setup;
            

            m_setup = vDSP_DFT_zop_CreateSetup(tmp_setup, fftsize, vDSP_DFT_FORWARD );
            if(m_setup == 0) {
                cerr << "fftsize " << fftsize << " is not valid." << endl;
                goto out;
            }
//            cout << "hello" << endl;
            
            if(tmp_setup != NULL) {
                vDSP_DFT_DestroySetup(tmp_setup);
            }
            
            
            std::vector<float> in_real(fftsize);
            std::vector<float> in_imag(fftsize, 0.0f);
            std::vector<float> out_real(fftsize);
            std::vector<float> out_imag(fftsize);
            std::vector<float> window_vector(fftsize);
            std::vector<float> magnitudes(fftsize);
            std::vector<float> phases(fftsize);
            std::vector<float> out_magnitudes;
            std::vector<float> out_phases;
            
            /* window */
            
            const string window_string = window.get().c_str();
            
            if(window_string == "triangle") {
                float tristart = 0.0f;
                float tritop = 1.0f;
                float up = 1.0f/(fftsize/2.0f);
                float down = -1.0f/(fftsize/2.0f);
                vDSP_vramp(&tristart, &up,window_vector.data(), 1, fftsize/2);
                vDSP_vramp(&tritop, &down,window_vector.data()+511, 1, fftsize/2);
            } else if (window_string == "hanning") {
                vDSP_hann_window(window_vector.data(), fftsize, vDSP_HANN_NORM );
            } else if (window_string == "hamming") {
                vDSP_hamm_window(window_vector.data(), fftsize, 0);
            } else if (window_string == "blackman") {
                vDSP_blkman_window(window_vector.data(), fftsize, 0);
            }
            
            DSPSplitComplex complex_input = { in_real.data(), in_imag.data()};
            DSPSplitComplex complex_output = { out_real.data(), out_imag.data()};
            
            for (auto i = 0; i < b.frame_count(); i+=step) {
            
                if(i+fftsize > b.frame_count()) {
                    for(auto j=0;j<fftsize;j++) {
                        if((i+j) > b.frame_count()) {
                            complex_input.realp[j] = 0.0f;
                        } else {
                            complex_input.realp[j] = b.lookup(i+j, chan);
                        }
                    }
                }
                else {
                    for(auto j=0;j<fftsize;j++) {
                        complex_input.realp[j] = b.lookup(i+j, chan);
                    }
                }
                //window
                if (window_string != "square") {
                    vDSP_vmul(complex_input.realp, 1, window_vector.data(), 1, complex_input.realp, 1, fftsize);
                }
                
                // dft
                vDSP_DFT_Execute(m_setup,complex_input.realp, complex_input.imagp,complex_output.realp,complex_output.imagp);
            
//                  not needed here
//                float scale = .5;
//            
//                vDSP_vsmul(complex_output.realp, 1, &scale, complex_output.realp, 1, fftsize);
//                vDSP_vsmul(complex_output.imagp, 1, &scale, complex_output.imagp, 1, fftsize);
         
            
                complex_output.imagp[0] = 0.0;
                
                if(output_polar) {
                    
                    vDSP_zvabs(&complex_output, 1, magnitudes.data(), 1, fftsize);  // Compute magnitude
                    vDSP_zvphas(&complex_output, 1, phases.data(), 1, fftsize);  // Compute phase (radians)
                    if(full_spectrum) {
                        std::copy(magnitudes.begin(), magnitudes.end(), std::back_inserter(out_magnitudes));
                        std::copy(phases.begin(), phases.end(), std::back_inserter(out_phases));
                    } else {
                        std::copy(magnitudes.begin()+(fftsize/2), magnitudes.end(), std::back_inserter(out_magnitudes));
                        std::copy(phases.begin()+(fftsize/2), phases.end(), std::back_inserter(out_phases));
                    }
                } else {
                    if(full_spectrum) {
                        std::copy(out_real.begin(), out_real.end(), std::back_inserter(out_magnitudes));
                        std::copy(out_imag.begin(), out_imag.end(), std::back_inserter(out_phases));
                    } else {
                        std::copy(out_real.begin()+(fftsize/2), out_real.end(), std::back_inserter(out_magnitudes));
                        std::copy(out_imag.begin()+(fftsize/2), out_imag.end(), std::back_inserter(out_phases));
                    }
                }
            }

        
            c74::max::t_object* tmp_matrix = static_cast<c74::max::t_object*>(c74::max::jit_object_new(c74::max::_jit_sym_jit_matrix,&out_minfo));
            
            if(!tmp_matrix) {
                (std::cerr << "could not create matrix" << std::endl);
                //c74::max::object_method(out_matrix, c74::max::_jit_sym_lock, out_savelock);
                return err;
            }
            
            
            c74::max::uchar *dataptr = nullptr;
            c74::max::uchar *ip = nullptr;
            float *fp = nullptr;
            err = (c74::max::t_jit_err)c74::max::object_method(tmp_matrix, c74::max::_jit_sym_getinfo, &out_minfo);
            err = (c74::max::t_jit_err)c74::max::object_method(tmp_matrix, c74::max::_jit_sym_getdata, &dataptr);
            
            long buf_pos = 0 ;

           // std::cout << out_minfo.dim[1] * out_minfo.dim[0] << " " << out_magnitudes.size() <<  " " << out_phases.size() << std::endl;;
            

            
            for(auto width=0;width<out_minfo.dim[0];width++) {
                ip = dataptr + (width*out_minfo.dimstride[0]);
                for(auto height=0;height<out_minfo.dim[1];height++) {
                    fp = (float*)ip;
                    *fp++ = out_magnitudes[buf_pos];
                    *fp++ = out_phases[buf_pos++];
                    ip = ip + out_minfo.dimstride[1];
                }
            }
            
            
            c74::max::object_method(out_matrix, c74::max::_jit_sym_setinfo,&out_minfo);
            c74::max::object_method(out_matrix, c74::max::_jit_sym_frommatrix,tmp_matrix,nullptr);
            c74::max::jit_object_free(tmp_matrix);
            
            // release LOCK
            c74::max::object_method(out_matrix, c74::max::_jit_sym_lock, out_savelock);
            //vDSP_DFT_DestroySetup(m_setup);
            //std::cout << setup << std::endl;
        }
    out:
        
        return err;
          
    }
    ~mlmat_buffer_stft() {
        if(m_setup != NULL) {
            vDSP_DFT_DestroySetup(m_setup);
        }
    }

protected:
    void* m_dumpoutlet { nullptr };
    
private:
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        c74::max::t_class* c = args[0];
        // add mop
        c74::max::t_object* mop = static_cast<c74::max::t_object*>(c74::max::jit_object_new(c74::max::_jit_sym_jit_mop, -1, 1));
        jit_mop_single_type(mop, c74::max::_jit_sym_float32);
        jit_class_addadornment(c, mop);
        jit_class_addmethod(c, (c74::max::method)mlmat_matrix_calc, "matrix_calc", c74::max::A_CANT, 0);

        return {};
    }};


    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        c74::max::t_class* c = args[0];
        char outputmatrix[] = "outputmatrix"; //to avoid warning about c++11 not allowing conversion from string literal to 'char*' when passing to max_jit_class_addmethod_usurp_low
        long flags = c74::max::MAX_JIT_MOP_FLAGS_OWN_OUTPUTMATRIX | c74::max::MAX_JIT_MOP_FLAGS_OWN_JIT_MATRIX;
        max_jit_class_mop_wrap(c, this_jit_class, flags);
        max_jit_class_wrap_standard(c, this_jit_class, 0);
        class_addmethod(c, (c74::max::method)mlmat_assist, "assist", c74::max::A_CANT, 0);
        max_jit_class_addmethod_usurp_low(c, (c74::max::method)mlmat_outputmatrix,outputmatrix);
        return {};
    }};
    
    vDSP_DFT_Setup m_setup = NULL;
};


MIN_EXTERNAL(mlmat_buffer_stft);

c74::max::t_jit_err mlmat_matrix_calc(c74::max::t_object* x, c74::max::t_object* inputs, c74::max::t_object* outputs) {
    c74::max::t_jit_err err = c74::max::JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return c74::max::JIT_ERR_INVALID_PTR;
    else {
        auto out_matrix = (c74::max::t_object*)c74::max::object_method(outputs, c74::max::_jit_sym_getindex, 0);
        if(out_matrix) {
            minwrap<mlmat_buffer_stft>* job = (minwrap<mlmat_buffer_stft>*)(x);
            err = job->m_min_object.matrix_calc(x, out_matrix);
        } else {
            return c74::max::JIT_ERR_INVALID_PTR;
        }
    }
    return err;
}

// this method is called on a 'outputmatrix' or 'bang' message
void mlmat_outputmatrix(max_jit_wrapper *x) {
    long outputmode = c74::max::max_jit_mop_getoutputmode(x);
    // get the reference to the max-wrapped jitter object
    c74::max::t_object *mop = (c74::max::t_object*)max_jit_obex_adornment_get(x, c74::max::_jit_sym_jit_mop);
    c74::max::t_jit_err err;

    if (outputmode && mop) { //always output unless output mode is none
        if (outputmode==1) {

            err = (c74::max::t_jit_err)c74::max::object_method((c74::max::t_object*)c74::max::max_jit_obex_jitob_get(x), c74::max::_jit_sym_matrix_calc, c74::max::object_method(mop, c74::max::_jit_sym_getinputlist), c74::max::object_method(mop, c74::max::_jit_sym_getoutputlist)
            );
            if(err) {
                c74::max::jit_error_code(x,err);
            }
            else {
                c74::max::max_jit_mop_outputmatrix(x);
            }
        }
        else {
            c74::max::max_jit_mop_outputmatrix(x);
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



//void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
//t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);
//void mlmat_outputmatrix(max_jit_wrapper *x);
//
//class mlmat_buffer_fft : public object<mlmat_buffer_fft>, public matrix_operator<>, public vector_operator<> {
//public:
//    MIN_DESCRIPTION	{"Load data from file. Can load a number of file formats including CSV , TSV, PGM, and ARFF."};
//    MIN_TAGS		{"ML"};
//    MIN_AUTHOR		{"Cycling '74"};
//    MIN_RELATED		{"jit.matrix"};
//    MIN_DISCUSSION  {"There are a number of file formats supported. These include:\n csv (comma-separated values), denoted by .csv or .txt\n tsv (tab-separated values), denoted by .tsv, .csv, or .txt\n ASCII (raw ASCII, with space-separated values), denoted by .txt\n Armadillo ASCII (Armadillo's text format with a header), denoted by .txt\n PGM, denoted by .pgm\n PPM, denoted by .ppm\n Armadillo binary, denoted by .bin\n Raw binary, denoted by .bin (note: this will be loaded as one-dimensional data, which is likely not what is desired.)\n ARFF, denoted by .arff.\n"};
//    
//        buffer_reference buffer { this,
//            MIN_FUNCTION {
//                length.touch();
//                return {};
//            }
//        };
//
//        argument<symbol> name_arg {this, "buffer-name", "Initial buffer~ from which to read.",
//            MIN_ARGUMENT_FUNCTION {
//                buffer.set(arg);
//            }
//        };
//
//        argument<int> channel_arg {this, "channel", "Initial channel to read from the buffer~.",
//            MIN_ARGUMENT_FUNCTION {
//                channel = arg;
//            }
//        };
//
//
//        attribute<int> channel {this, "channel", 1, description {"Channel to read from the buffer~."},
//            setter { MIN_FUNCTION {
//                int n = args[0];
//                if (n < 1)
//                    n = 1;
//                return {n};
//            }}
//        };
//    
////    attribute<min::symbol> file {this, "file", k_sym__empty,
////        description {
////            "File"
////        },
////        setter { MIN_FUNCTION {
////            if(args[0] != k_sym__empty) {
////                load_file(args);
////            }
////            return args;
////        }}
////    };
////    
////    attribute<bool> transpose {this, "transpose", true,
////        description { "If true, transpose the matrix after loading." }
////    };
////
////    message<> read { this, "read", "data file to read",
////        MIN_FUNCTION {
////            load_file(args);
////            return {};
////        }
////    };
////    
////    void load_file(const atoms& args) {
////        try {
////            atoms f{};
////
////            if(!args.empty()) {
////                f.push_back(args[0]);
////            }
////            path p {f, path::filetype::any};
////
////            m_dataset = std::make_unique<arma::mat>();
////            mlpack::data::Load<double>(string(p), *m_dataset, false, transpose);
////        }
////        catch (...) {
////            (cerr << "Could not read file " << endl);
////        }
////    }
////    
////    
////
////    t_jit_err matrix_calc(t_object* x, t_object* out_matrix) {
////        t_jit_err err = JIT_ERR_NONE;
////        t_jit_matrix_info out_minfo;
////        auto out_savelock = object_method(out_matrix, _jit_sym_lock, (void*)1);
////    
////        out_minfo.type = _jit_sym_float64;
////
////        switch(mode.get()) {
////           case 0:
////               out_minfo.planecount = m_dataset->n_rows;
////               out_minfo.dimcount = 1;
////               out_minfo.dim[0] =   m_dataset->n_cols;
////               out_minfo.dim[1] =   1;
////               break;
////           case 1:
////               out_minfo.planecount = 1;
////                out_minfo.dim[0] = m_dataset->n_cols;
////                out_minfo.dim[1] = m_dataset->n_rows;
////               out_minfo.dim[0] = m_dataset->n_rows;
////               out_minfo.dim[1] = m_dataset->n_cols;
////               break;
////           case 2:
////               out_minfo.planecount = 1;
////               out_minfo.dim[0] = m_dataset->n_rows;
////               out_minfo.dim[1] = m_dataset->n_cols;
////               break;
////           default:
////               out_minfo.planecount = 1;
////               break;
////                       
////        }
////
////        out_matrix = arma_to_jit(mode, *m_dataset, static_cast<t_object*>(out_matrix), out_minfo);
////
////       // release LOCK
////        object_method(out_matrix, _jit_sym_lock, out_savelock);
////        return err;
////          
////    }
////    
////private:
////    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
////        t_class* c = args[0];
////        // add mop
////        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, -1, 1));
////        jit_mop_single_type(mop, _jit_sym_float64);
////        jit_class_addadornment(c, mop);
////        jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
////        
////        return {};
////    }};
////    
////    
////    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
////        t_class* c = args[0];
////        char outputmatrix[] = "outputmatrix"; //to avoid warning about c++11 not allowing conversion from string literal to 'char*' when passing to max_jit_class_addmethod_usurp_low
////        long flags = MAX_JIT_MOP_FLAGS_OWN_OUTPUTMATRIX | MAX_JIT_MOP_FLAGS_OWN_JIT_MATRIX;
////        max_jit_class_mop_wrap(c, this_jit_class, flags);
////        max_jit_class_wrap_standard(c, this_jit_class, 0);
////        class_addmethod(c, (method)mlmat_assist, "assist", A_CANT, 0);
////        max_jit_class_addmethod_usurp_low(c, (method)mlmat_outputmatrix,outputmatrix);
////        return {};
////    }};
////    std::unique_ptr<arma::mat> m_dataset {nullptr};
//};
//                
//MIN_EXTERNAL(mlmat_buffer_fft);
//
//
//t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
//    t_jit_err err = JIT_ERR_NONE;
//    if (!x || !inputs || !outputs)
//        return JIT_ERR_INVALID_PTR;
//    else {
//        auto out_matrix = (t_object*)object_method(outputs, _jit_sym_getindex, 0);
//        if(out_matrix) {
//            minwrap<mlmat_load>* job = (minwrap<mlmat_load>*)(x);
//            err = job->m_min_object.matrix_calc(x, out_matrix);
//        } else {
//            return JIT_ERR_INVALID_PTR;
//        }
//    }
//    return err;
//}
//
//// this method is called on a 'outputmatrix' or 'bang' message
//void mlmat_outputmatrix(max_jit_wrapper *x) {
//    long outputmode = max_jit_mop_getoutputmode(x);
//    // get the reference to the max-wrapped jitter object
//    t_object *mop = (t_object*)max_jit_obex_adornment_get(x, _jit_sym_jit_mop);
//    t_jit_err err;
//
//    if (outputmode && mop) { //always output unless output mode is none
//        if (outputmode==1) {
//        
//            err = (t_jit_err)object_method((t_object*)max_jit_obex_jitob_get(x), _jit_sym_matrix_calc, object_method(mop, _jit_sym_getinputlist), object_method(mop, _jit_sym_getoutputlist)
//            );
//            if(err) {
//                jit_error_code(x,err);
//            }
//            else {
//                max_jit_mop_outputmatrix(x);
//            }
//        }
//        else {
//            max_jit_mop_outputmatrix(x);
//        }
//    }
//}
//
//
//void mlmat_assist(void* x, void* b, long io, long index, char* s) {
//   switch(io) {
//       case 1:
//           switch(index) {
//               case 0:
//                   sprintf(s, "messages");
//                   break;
//               default:
//                   break;
//           }
//           break;
//       case 2:
//           switch(index) {
//               case 0:
//                   sprintf(s, "(matrix) data");
//                   break;
//
//               default:
//                   sprintf(s, "dumpout");
//                   break;
//           }
//
//   }
//}










//#include <stdio.h>
//#include <Accelerate/Accelerate.h>
//
//void fft(DSPComplex buf[], int n) {
//  float inputMemory[2*n];
//  float outputMemory[2*n];
//  // half for real and half for complex
//  DSPSplitComplex inputSplit = {inputMemory, inputMemory + n};
//  DSPSplitComplex outputSplit = {outputMemory, outputMemory + n};
//  
//  vDSP_ctoz(buf, 2, &inputSplit, 1, n);
//  
//  vDSP_DFT_Setup setup = vDSP_DFT_zop_CreateSetup(NULL, n, vDSP_DFT_FORWARD);
//  
//  vDSP_DFT_Execute(setup,
//                   inputSplit.realp, inputSplit.imagp,
//                   outputSplit.realp, outputSplit.imagp);
//  
//  vDSP_ztoc(&outputSplit, 1, buf, 2, n);
//}
//
//
//void show(const char *s, DSPComplex buf[], int n) {
//  printf("%s", s);
//  for (int i = 0; i < n; i++)
//    if (!buf[i].imag)
//      printf("%g ", buf[i].real);
//    else
//      printf("(%g, %g) ", buf[i].real, buf[i].imag);
//  printf("\n");
//}
//
//int main() {
//  DSPComplex buf[] = {{1,0}, {1,0}, {1,0}, {1,0}, {0,0}, {0,0}, {0,0}, {0,0}};
//  
//  show("Data: ", buf, 8);
//  fft(buf, 8);
//  show("FFT : ", buf, 8);
//  
//  return 0;
//}


//            // does rows
//            for(auto height=0;height<out_minfo.dim[1];height++) { //height
//                p = (float*)(dataptr + (height*out_minfo.dimstride[1]));
//                for(auto width=0;width<out_minfo.dim[0];width++) { //width
//                    *p++ = (float)buf_pos;
//                    *p++ = 0.0;
//                    buf_pos++;
//                }
//            }


//std::vector<float> in_real(fftsize);
//std::vector<float> in_imag(fftsize, 0.0f);
//std::vector<float> out_real(fftsize / 2);
//std::vector<float> out_imag(fftsize / 2);
//std::vector<float> window(fftsize);
//std::vector<float> magnitudes(fftsize);
//std::vector<float> phases(fftsize);
//std::vector<float> out_magnitudes;
//std::vector<float> out_phases;
//
//
//vDSP_hann_window(window.data(), fftsize, vDSP_HANN_NORM );
//std::vector<DSPComplex> polar_out;
//
//for (auto i = 0; i < b.frame_count(); i+=step) {
//    
//    for(auto j=0;j<fftsize;j++) {
//        in_real[j] = b.lookup(i+j, chan);
//    }
//    //window
//    vDSP_vmul(in_real.data(), 1, window.data(), 1, in_real.data(), 1, fftsize);
//    // dft
//    vDSP_DFT_Execute(setup,in_real.data(), in_imag.data(),out_real.data(),out_imag.data());
//    
//    
//    float scale = .5  / (fftsize/4);
//    
//    //float scale = 1.0f/(float)(4.0f*fftsize);
//     
//    vDSP_vsmul(out_real.data(), 1, &scale, out_real.data(), 1, out_real.size());
//    vDSP_vsmul(out_imag.data(), 1, &scale, out_imag.data(), 1, out_imag.size());
//    
////                for(auto i=0;i<fftsize;i++) {
////                    std::cout << i << ": " << out_real[i] << " ";
////                }
////                std::cout << std::endl;
//    
//    DSPSplitComplex dspsc;
//    dspsc.realp = out_real.data();
//    dspsc.imagp = out_imag.data();
//
//    
//    vDSP_zvabs(&dspsc, 1, magnitudes.data(), 1, fftsize / 2);  // Compute magnitude
//    vDSP_zvphas(&dspsc, 1, phases.data(), 1, fftsize / 2);  // Compute phase (radians)
//
//    
//    //scale = 1.0f/(float)(4.0f*fftSize)
//    //vDSP_vsmul(out_real, 1, &scale, out_real, 1, fftSize);
//    
//    std::copy(magnitudes.begin(), magnitudes.end(), std::back_inserter(out_magnitudes));
//    std::copy(phases.begin(), phases.end(), std::back_inserter(out_phases));
//}


//std::vector<float> input(fftsize);
//std::vector<float> in_real(fftsize / 2);
//std::vector<float> in_imag(fftsize / 2);
//std::vector<float> out_real(fftsize / 2);
//std::vector<float> out_imag(fftsize / 2);
//std::vector<float> window(fftsize);
//std::vector<float> magnitudes(fftsize / 2);
//std::vector<float> phases(fftsize / 2);
//std::vector<float> out_magnitudes;
//std::vector<float> out_phases;
//
//
//vDSP_hann_window(window.data(), fftsize, vDSP_HANN_NORM );
//
//DSPSplitComplex complex_input = { in_real.data(), in_imag.data()};
//DSPSplitComplex complex_output = { out_real.data(), out_imag.data()};
//
//for (auto i = 0; i < b.frame_count(); i+=step) {
//    
//    for(auto j=0;j<fftsize;j++) {
//        input[j] = b.lookup(i+j, chan);
//    }
//  
//    //window
//    vDSP_vmul(input.data(), 1, window.data(), 1, input.data(), 1, fftsize);
//    //convert to complex
//    vDSP_ctoz((DSPComplex *) input.data(), 2, &complex_input, 1, fftsize/2);
////                for(auto i=0;i<fftsize;i++) {
////                    std::cout << i << ": " << complex_input.realp[i] << " " << complex_input.imagp[i];
////                }
////                std::cout << std::endl;
//    
//    // dft
//    vDSP_DFT_Execute(setup,complex_input.realp, complex_input.imagp,complex_output.realp,complex_output.imagp);
//    
//    
//    float scale = .5;
//    
//    //float scale = 1.0f/(float)(4.0f*fftsize);
////                for(auto i=0;i<20;i++) {
////                    std::cout << i << ": " << complex_output.realp[i] << " ";
////                }
////                std::cout << std::endl;
//     
//   // vDSP_vsmul(complex_output.realp, 1, &scale, complex_output.realp, 1, fftsize / 2);
//    
////                for(auto i=0;i<20;i++) {
////                    std::cout << i << ": " << complex_output.realp[i] << " ";
////                }
////                std::cout << std::endl;
//    
//    //vDSP_vsmul(complex_output.imagp, 1, &scale, complex_output.imagp, 1, fftsize / 2);
//    
////                for(auto i=0;i<fftsize;i++) {
////                    std::cout << i << ": " << out_real[i] << " ";
////                }
////                std::cout << std::endl;
//    
////                DSPSplitComplex dspsc;
////                dspsc.realp = out_real.data();
////                dspsc.imagp = out_imag.data();
//
//    
//    vDSP_zvabs(&complex_output, 1, magnitudes.data(), 1, fftsize / 2);  // Compute magnitude
//    vDSP_zvphas(&complex_output, 1, phases.data(), 1, fftsize / 2);  // Compute phase (radians)
//
//    
//    //scale = 1.0f/(float)(4.0f*fftSize)
//    //vDSP_vsmul(out_real, 1, &scale, out_real, 1, fftSize);
//    
//    std::copy(magnitudes.begin(), magnitudes.end(), std::back_inserter(out_magnitudes));
//    std::copy(phases.begin(), phases.end(), std::back_inserter(out_phases));
//}


//vDSP_DFT_Setup setup = vDSP_DFT_zrop_CreateSetup(0, fftsize, vDSP_DFT_FORWARD );
//
//std::vector<float> input(fftsize);
//std::vector<float> in_real(fftsize / 2);
//std::vector<float> in_imag(fftsize / 2);
//std::vector<float> out_real(fftsize / 2);
//std::vector<float> out_imag(fftsize / 2);
//std::vector<float> window(fftsize);
//std::vector<float> magnitudes(fftsize / 2);
//std::vector<float> phases(fftsize / 2);
//std::vector<float> out_magnitudes;
//std::vector<float> out_phases;
//
//
//vDSP_hann_window(window.data(), fftsize, vDSP_HANN_NORM );
//
//DSPSplitComplex complex_input = { in_real.data(), in_imag.data()};
//DSPSplitComplex complex_output = { out_real.data(), out_imag.data()};
//
//for (auto i = 0; i < b.frame_count(); i+=step) {
//
//    for(auto j=0;j<fftsize;j++) {
//        input[j] = b.lookup(i+j, chan);
//    }
//
//    //window
//    vDSP_vmul(input.data(), 1, window.data(), 1, input.data(), 1, fftsize);
//    
//    
//    //convert to complex
//    vDSP_ctoz((DSPComplex *) input.data(), 2, &complex_input, 1, fftsize / 2);
//
//    // dft
//    vDSP_DFT_Execute(setup,complex_input.realp, complex_input.imagp,complex_output.realp,complex_output.imagp);
//
//
//    float scale = .5;
//
//    //float scale = 1.0f/(float)(4.0f*fftsize);
////                for(auto i=0;i<20;i++) {
////                    std::cout << i << ": " << complex_output.realp[i] << " ";
////                }
////                std::cout << std::endl;
//
//   // vDSP_vsmul(complex_output.realp, 1, &scale, complex_output.realp, 1, fftsize / 2);
//
////                for(auto i=0;i<20;i++) {
////                    std::cout << i << ": " << complex_output.realp[i] << " ";
////                }
////                std::cout << std::endl;
//
//    //vDSP_vsmul(complex_output.imagp, 1, &scale, complex_output.imagp, 1, fftsize / 2);
//
////                for(auto i=0;i<fftsize;i++) {
////                    std::cout << i << ": " << out_real[i] << " ";
////                }
////                std::cout << std::endl;
//
////                DSPSplitComplex dspsc;
////                dspsc.realp = out_real.data();
////                dspsc.imagp = out_imag.data();
//
//
//    vDSP_zvabs(&complex_output, 1, magnitudes.data(), 1, fftsize / 2);  // Compute magnitude
//    vDSP_zvphas(&complex_output, 1, phases.data(), 1, fftsize / 2);  // Compute phase (radians)
//
//
//    //scale = 1.0f/(float)(4.0f*fftSize)
//    //vDSP_vsmul(out_real, 1, &scale, out_real, 1, fftSize);
//
//    std::copy(magnitudes.begin(), magnitudes.end(), std::back_inserter(out_magnitudes));
//    std::copy(phases.begin(), phases.end(), std::back_inserter(out_phases));
//}
