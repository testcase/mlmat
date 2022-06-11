/// @file mlmat.load.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.


#include "mlmat.hpp"
#include <mlpack/core.hpp>

using namespace c74;
using namespace c74::min;
using namespace c74::max;


void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);
void mlmat_outputmatrix(max_jit_wrapper *x);

class mlmat_load : public mlmat_object<mlmat_load> {
public:
    MIN_DESCRIPTION	{"Load data from file. Can load a number of file formats including CSV , TSV, PGM, and ARFF."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Cycling '74"};
    MIN_RELATED		{"jit.matrix"};
    MIN_DISCUSSION  {"There are a number of file formats supported. These include:\n csv (comma-separated values), denoted by .csv or .txt\n tsv (tab-separated values), denoted by .tsv, .csv, or .txt\n ASCII (raw ASCII, with space-separated values), denoted by .txt\n Armadillo ASCII (Armadillo's text format with a header), denoted by .txt\n PGM, denoted by .pgm\n PPM, denoted by .ppm\n Armadillo binary, denoted by .bin\n Raw binary, denoted by .bin (note: this will be loaded as one-dimensional data, which is likely not what is desired.)\n ARFF, denoted by .arff.\n"};
    
    attribute<min::symbol> file {this, "file", k_sym__empty,
        description {
            "File"
        },
        setter { MIN_FUNCTION {
            if(args[0] != k_sym__empty) {
                load_file(args);
            }
            return args;
        }}
    };
    
    attribute<bool> transpose {this, "transpose", true,
        description { "If true, transpose the matrix after loading." }
    };

    message<> read { this, "read", "data file to read",
        MIN_FUNCTION {
            load_file(args);
            return {};
        }
    };
    
    void load_file(const atoms& args) {
        try {
            atoms f{};

            if(!args.empty()) {
                f.push_back(args[0]);
            }
            path p {f, path::filetype::any};

            m_dataset = std::make_unique<arma::mat>();
            mlpack::data::Load<double>(string(p), *m_dataset, false, transpose);
        }
        catch (...) {
            (cerr << "Could not read file " << endl);
        }
    }
    
    

    t_jit_err matrix_calc(t_object* x, t_object* out_matrix) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info out_minfo;
        auto out_savelock = object_method(out_matrix, _jit_sym_lock, (void*)1);
    
        out_minfo.type = _jit_sym_float64;

        switch(mode.get()) {
           case 0:
               out_minfo.planecount = m_dataset->n_rows;
               out_minfo.dimcount = 1;
               out_minfo.dim[0] =   m_dataset->n_cols;
               out_minfo.dim[1] =   1;
               break;
           case 1:
               out_minfo.planecount = 1;
                out_minfo.dim[0] = m_dataset->n_cols;
                out_minfo.dim[1] = m_dataset->n_rows;
               out_minfo.dim[0] = m_dataset->n_rows;
               out_minfo.dim[1] = m_dataset->n_cols;
               break;
           case 2:
               out_minfo.planecount = 1;
               out_minfo.dim[0] = m_dataset->n_rows;
               out_minfo.dim[1] = m_dataset->n_cols;
               break;
           default:
               out_minfo.planecount = 1;
               break;
                       
        }

        out_matrix = arma_to_jit(mode, *m_dataset, static_cast<t_object*>(out_matrix), out_minfo);

       // release LOCK
        object_method(out_matrix, _jit_sym_lock, out_savelock);
        return err;
          
    }
    
private:
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        // add mop
        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, -1, 1));
        jit_mop_single_type(mop, _jit_sym_float64);
        jit_class_addadornment(c, mop);
        jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
        
        return {};
    }};
    
    
    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        char outputmatrix[] = "outputmatrix"; //to avoid warning about c++11 not allowing conversion from string literal to 'char*' when passing to max_jit_class_addmethod_usurp_low
        long flags = MAX_JIT_MOP_FLAGS_OWN_OUTPUTMATRIX | MAX_JIT_MOP_FLAGS_OWN_JIT_MATRIX;
        max_jit_class_mop_wrap(c, this_jit_class, flags);
        max_jit_class_wrap_standard(c, this_jit_class, 0);
        class_addmethod(c, (method)mlmat_assist, "assist", A_CANT, 0);
        max_jit_class_addmethod_usurp_low(c, (method)mlmat_outputmatrix,outputmatrix);
        return {};
    }};
    std::unique_ptr<arma::mat> m_dataset {nullptr};
};
                
MIN_EXTERNAL(mlmat_load);


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        auto out_matrix = (t_object*)object_method(outputs, _jit_sym_getindex, 0);
        if(out_matrix) {
            minwrap<mlmat_load>* job = (minwrap<mlmat_load>*)(x);
            err = job->m_min_object.matrix_calc(x, out_matrix);
        } else {
            return JIT_ERR_INVALID_PTR;
        }
    }
    return err;
}

// this method is called on a 'outputmatrix' or 'bang' message
void mlmat_outputmatrix(max_jit_wrapper *x) {
    long outputmode = max_jit_mop_getoutputmode(x);
    // get the reference to the max-wrapped jitter object
    t_object *mop = (t_object*)max_jit_obex_adornment_get(x, _jit_sym_jit_mop);
    t_jit_err err;

    if (outputmode && mop) { //always output unless output mode is none
        if (outputmode==1) {
        
            err = (t_jit_err)object_method((t_object*)max_jit_obex_jitob_get(x), _jit_sym_matrix_calc, object_method(mop, _jit_sym_getinputlist), object_method(mop, _jit_sym_getoutputlist)
            );
            if(err) {
                jit_error_code(x,err);
            }
            else {
                max_jit_mop_outputmatrix(x);
            }
        }
        else {
            max_jit_mop_outputmatrix(x);
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










