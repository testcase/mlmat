/// @file mlmat.nmf.cpp
/// @ingroup mlmat
/// @copyright Copyright 2024 Todd Ingalls. All rights reserved. Also based on examples provided with the mlpack library. Please see source/mlpack for license details
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// TODO: add seed as some strategies use random initialization??
#include <string>

#include "mlmat.hpp"
#include <mlpack/methods/amf.hpp>



using namespace c74;
using namespace c74::min;
using namespace c74::max;
using namespace mlpack;
using namespace mlpack::util;

void mlmat_nmf_assist(void* x, void* b, long m, long a, char* s) ;
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);

class mlmat_nmf : public mlmat_object<mlmat_nmf> {
public:
    MIN_DESCRIPTION	{"Non-Negative Matrix factorization. Decomposes a matrix V in the form V ~ WH where W is called the basis matrix and H is called the encoding matrix. V is taken to be of size n x m and the obtained W is n x r and H is r x m. The size r is called the rank of the factorization. Factorization is done by alternately calculating W and H respectively while holding the other matrix constant."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.nmf"};


    attribute<int> rank { this, "rank", 2,
        description { "The rank of the factorization."}
    };

    attribute<int> seed { this, "seed", 0,
        description {
            "Random seed if random basis being used. 0 indicates no seed."
        }
    };
    
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        // add mop
        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 1, 2 ));
        
        // force type
        jit_mop_single_type(mop, _jit_sym_float64);
        
        auto output1 = object_method(mop,_jit_sym_getoutput,1);
        jit_attr_setlong(output1,_jit_sym_dimlink,0);
        auto output2 = object_method(mop,_jit_sym_getoutput,1);
        jit_attr_setlong(output2,_jit_sym_dimlink,0);

        jit_class_addadornment(c, mop);

        // add our custom matrix_calc method
        jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
        return {};
    }};

    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        
        max_jit_class_mop_wrap(c, this_jit_class, 0);
        max_jit_class_wrap_standard(c, this_jit_class, 0);
        class_addmethod(c, (method)mlmat_nmf_assist, "assist", A_CANT, 0);

        return {};
    }};
    
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_query_info, out_info, W_info, H_info;
        arma::mat query;

        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto W_matrix = object_method(outputs, _jit_sym_getindex, 0);
        auto H_matrix = object_method(outputs, _jit_sym_getindex, 1);

    
        auto in_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto W_savelock = object_method(W_matrix, _jit_sym_lock, 1);
        auto H_savelock = object_method(H_matrix, _jit_sym_lock, 1);

        object_method(in_matrix, _jit_sym_getinfo, &in_query_info);
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_query_info);
        
        NMFALSFactorizer nmf;
        arma::mat W, H;
        size_t r = rank;
        double residue = 0.;
        t_atom a[1];
        
        try {
            check_mode(in_query_info, mode, "nmf");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        query = jit_to_arma(mode, static_cast<t_object*>(in_matrix64), query);

        if (seed != 0)
          mlpack::RandomSeed((size_t) seed);
        else
          mlpack::RandomSeed((size_t) std::time(NULL));

        residue = nmf.Apply(query, r, W, H);
    
        std::cout << residue << std::endl;
        
        
        switch(mode.get()) {
            case 0:
                W_info.type = _jit_sym_float64;
                W_info.flags = 0;
                W_info.dimcount = 1;
                W_info.planecount = W.n_cols;
                W_info.dim[0] = W.n_rows;
                
                H_info.type = _jit_sym_float64;
                H_info.flags = 0;
                H_info.dimcount = 1;
                H_info.planecount = W.n_rows;
                H_info.dim[0] = W.n_cols;
                break;
            case 1:
                W_info.type = _jit_sym_float64;
                W_info.flags = 0;
                W_info.dimcount = 2;
                W_info.planecount = 1;
                W_info.dim[0] = W.n_rows;
                W_info.dim[1] = W.n_cols;
                
                H_info.type = _jit_sym_float64;
                H_info.flags = 0;
                H_info.dimcount = 2;
                H_info.planecount = 1;
                H_info.dim[0] = H.n_rows;
                H_info.dim[1] = H.n_cols;
                break;
            case 2:
                W_info.type = _jit_sym_float64;
                W_info.flags = 0;
                W_info.dimcount = 2;
                W_info.planecount = 1;
                W_info.dim[0] = W.n_cols;
                W_info.dim[1] = W.n_rows;
                
                H_info.type = _jit_sym_float64;
                H_info.flags = 0;
                H_info.dimcount = 2;
                H_info.planecount = 1;
                H_info.dim[0] = H.n_cols;
                H_info.dim[1] = H.n_rows;
                break;
            default:
                out_info.planecount = 1;
                break;
                
        }
            
        W_matrix = arma_to_jit(mode, W, static_cast<t_object*>(W_matrix), W_info );
        H_matrix = arma_to_jit(mode, H, static_cast<t_object*>(H_matrix), H_info );

    out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        
        object_method(in_matrix,_jit_sym_lock,in_savelock);
        object_method(W_matrix,_jit_sym_lock,W_savelock);
        object_method(H_matrix,_jit_sym_lock,H_savelock);
        atom_setfloat(a,residue);
        outlet_anything(m_dumpoutlet, gensym("residue"), 1, a);
        return err;
    }

};


MIN_EXTERNAL(mlmat_nmf);


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_nmf>* job = (minwrap<mlmat_nmf>*)(x);
        err = job->m_min_object.matrix_calc(x, inputs, outputs);
    }

    return err;
}


void mlmat_nmf_assist(void* x, void* b, long io, long index, char* s) {
    switch(io) {
        case 1:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) dataset");
                    break;

                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) W");
                    break;
                case 1:
                    sprintf(s, "(matrix) H");
                    break;
                default:
                    sprintf(s, "dumpout");
                    break;
            }

    }
}
