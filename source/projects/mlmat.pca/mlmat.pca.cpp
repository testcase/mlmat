/// @file mlmat.pca.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// TODO: add seed as some strategies use random initialization??

#include "c74_min.h"
#include "pca_ext.hpp"
#include "pca_ext_impl.hpp"
#include <mlpack/methods/pca/decomposition_policies/exact_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/quic_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/randomized_svd_method.hpp>
#include <mlpack/methods/pca/decomposition_policies/randomized_block_krylov_method.hpp>
#include "mlmat_operator.hpp"
#include <string>
using namespace c74;
using namespace c74::min;
using namespace c74::max;
using namespace mlpack;
using namespace mlpack::pca;
using namespace mlpack::util;

void mlmat_pca_assist(void* x, void* b, long m, long a, char* s) ;
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);

class mlmat_pca : public mlmat_operator<mlmat_pca> {
public:
    MIN_DESCRIPTION	{"Principal Components Analysis. An implementation of several strategies for principal components analysis (PCA), a common preprocessing step.  Given a dataset and a desired new dimensionality, this can reduce the dimensionality of the data using the linear transformation determined by PCA."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.linear_regression"};


    attribute<int> new_dimensionality { this, "new_dimensionality", 2,
        description { "Desired dimensionality obf output dataset. If 0, no dimensionality reduction is performed."}
    };
    
    attribute<bool> scale { this, "scale", false,
        description {
            "If set, the data will be scaled before running PCA, such that the variance of each feature is 1."
        }
    };
    
    attribute<int> seed { this, "seed", 0,
        description {
            "Random seed if random basis being used. 0 indicates no seed."
        }
    };
    
    attribute<min::symbol> decomposition_method { this, "decomposition_method", "exact",
         description {
             "Method used for the principal components analysis."
         },

         range {"exact", "randomized", "randomized-block-krylov", "quic"}
     };

    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        // add mop
        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 1, 3));
        
        // force type
        jit_mop_single_type(mop, _jit_sym_float64);
        
        auto output1 = object_method(mop,_jit_sym_getoutput,1);
        jit_attr_setlong(output1,_jit_sym_dimlink,0);
        auto output2 = object_method(mop,_jit_sym_getoutput,1);
        jit_attr_setlong(output2,_jit_sym_dimlink,0);
        auto output3 = object_method(mop,_jit_sym_getoutput,1);
        jit_attr_setlong(output3,_jit_sym_dimlink,0);

        jit_class_addadornment(c, mop);

        // add our custom matrix_calc method
        jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
        return {};
    }};

    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        
        max_jit_class_mop_wrap(c, this_jit_class, 0);
        max_jit_class_wrap_standard(c, this_jit_class, 0);
        class_addmethod(c, (method)mlmat_pca_assist, "assist", A_CANT, 0);

        return {};
    }};
    
    template<typename DecompositionPolicy>
    void RunPCA(arma::mat& dataset, arma::vec& eigVal, arma::mat& eigvec, const size_t newDimension, const bool scale) {
        t_atom a[1];
        PCA_EXT<DecompositionPolicy> p(scale);
        double variance = 0.;

        variance = p.Apply(dataset, eigVal, eigvec, new_dimensionality);
        atom_setfloat(a,variance);
        outlet_anything(m_dumpoutlet, gensym("variance"), 1, a);
    }
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_query_info, out_info, eigVal_info, eigvec_info;
        arma::mat query;
        arma::vec eigVal;
        arma::mat eigvec;
        size_t input_dimensionality = 0;

        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto out_matrix = object_method(outputs, _jit_sym_getindex, 0);
        auto eigval_matrix = object_method(outputs, _jit_sym_getindex, 1);
        auto eigvec_matrix = object_method(outputs, _jit_sym_getindex, 2);

    
        auto in_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_savelock = object_method(out_matrix, _jit_sym_lock, 1);
        auto eigval_savelock = object_method(out_matrix, _jit_sym_lock, 1);
        auto eigvec_savelock = object_method(out_matrix, _jit_sym_lock, 1);
        
        const string decomp = decomposition_method.get().c_str() ;
        
        object_method(in_matrix, _jit_sym_getinfo, &in_query_info);
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_query_info);
        
        
        try {
            check_mode(in_query_info, mode, "pca");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        
    
        switch (mode) {
            case 0:
                input_dimensionality = in_query_info.planecount;
                break;
            case 1:
                input_dimensionality = in_query_info.dim[0];
                break;
            case 2:
                input_dimensionality = in_query_info.dim[1];
                break;
            default:
                break;
            
        }
        
        if(input_dimensionality < new_dimensionality) {
            cerr << "Input dimensionality of " << input_dimensionality
            << " is less than requested dimensionality of " << new_dimensionality << endl;
            goto out;
        }
        
        query = jit_to_arma(mode, static_cast<t_object*>(in_matrix64), query);
        
        if (seed != 0)
          mlpack::math::RandomSeed((size_t) seed);
        else
          mlpack::math::RandomSeed((size_t) std::time(NULL));
    
        // Perform PCA.
        if (decomp == "exact")
        {
          RunPCA<ExactSVDPolicy>(query, eigVal, eigvec, new_dimensionality, scale);
        }
        else if (decomp == "randomized")
        {
          RunPCA<RandomizedSVDPolicy>(query, eigVal, eigvec, new_dimensionality, scale);
        }
        else if (decomp == "randomized-block-krylov")
        {
          RunPCA<RandomizedBlockKrylovSVDPolicy>(query, eigVal, eigvec, new_dimensionality, scale);
        }
        else if (decomp == "quic")
        {
          RunPCA<QUICSVDPolicy>(query, eigVal, eigvec, new_dimensionality, scale);
        }

        out_info = in_query_info;
        out_info.planecount = query.n_rows;
        
        eigVal_info.type = _jit_sym_float64;
        eigVal_info.flags = 0;
        eigVal_info.dimcount = 1;
        eigVal_info.dim[0] = 1;
        eigVal_info.planecount = eigVal.n_rows;
        
        eigvec_info.type = _jit_sym_float64;
        eigvec_info.flags = 0;
        eigvec_info.dimcount = 1;
        eigvec_info.dim[0] = eigvec.n_cols;
        eigvec_info.planecount = eigvec.n_rows;
        
            
        out_matrix = arma_to_jit(mode, query, static_cast<t_object*>(out_matrix), out_info );
        eigval_matrix = arma_to_jit(mode, eigVal, static_cast<t_object*>(eigval_matrix), eigVal_info );
        eigvec_matrix = arma_to_jit(mode, eigvec, static_cast<t_object*>(eigvec_matrix), eigvec_info );

    out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        
        object_method(in_matrix,_jit_sym_lock,in_savelock);
        object_method(out_matrix,_jit_sym_lock,out_savelock);
        object_method(eigval_matrix,_jit_sym_lock,eigval_savelock);
        object_method(eigvec_matrix,_jit_sym_lock,eigvec_savelock);

        return err;
    }

};


MIN_EXTERNAL(mlmat_pca);


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_pca>* job = (minwrap<mlmat_pca>*)(x);
        err = job->m_min_object.matrix_calc(x, inputs, outputs);
    }

    return err;
}


void mlmat_pca_assist(void* x, void* b, long io, long index, char* s) {
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
                    sprintf(s, "(matrix) transformed data");
                    break;
                case 1:
                    sprintf(s, "(matrix) eigenvalues");
                    break;
                case 2:
                    sprintf(s, "(matrix) eigenvectors");
                    break;
                default:
                    sprintf(s, "dumpout");
                    break;
            }

    }
}
