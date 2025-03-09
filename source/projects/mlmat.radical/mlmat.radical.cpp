/// @file mlmat.nmf.cpp
/// @ingroup mlmat
/// @copyright Copyright 2024 Todd Ingalls. All rights reserved. Also based on examples provided with the mlpack library. Please see source/mlpack for license details
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// TODO: add seed as some strategies use random initialization??
#include <string>

#include "mlmat.hpp"
#include <mlpack/methods/radical.hpp>



using namespace c74;
using namespace c74::min;
using namespace c74::max;
using namespace mlpack;
using namespace mlpack::util;

void mlmat_radical_assist(void* x, void* b, long m, long a, char* s) ;
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);

class mlmat_radical : public mlmat_object<mlmat_radical> {
public:
    MIN_DESCRIPTION	{"RADICAL, an algorithm for independent component analysis (ICA). Let X be a matrix where each column is a point and each row a dimension. The goal is to find a square unmixing matrix W such that Y = W X and the rows of Y are independent components."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.radical"};


    attribute<double> std_dev { this, "std_dev", .175,
        description { "Standard deviation of the Gaussian noise added to the replicates of the data points during Radical2D."}
    };
    
    attribute<int> replicates { this, "replicates", 30,
        description { "Number of angles to consider in brute-force search during Radical2D."}
    };
    
    attribute<int> angles { this, "angles", 150,
        description { "Number of angles to consider in brute-force search during Radical2D."}
    };
    
    
    attribute<int> sweeps { this, "sweeps", 0,
        description { "Number of sweeps.  Each sweep calls Radical2D once for each pair of dimensions."}
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
        class_addmethod(c, (method)mlmat_radical_assist, "assist", A_CANT, 0);

        return {};
    }};
    
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_query_info, out_info, Y_info, W_info;
        arma::mat query;

        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto Y_matrix = object_method(outputs, _jit_sym_getindex, 0);
        auto W_matrix = object_method(outputs, _jit_sym_getindex, 1);

    
        auto in_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto Y_savelock = object_method(Y_matrix, _jit_sym_lock, 1);
        auto W_savelock = object_method(W_matrix, _jit_sym_lock, 1);

        object_method(in_matrix, _jit_sym_getinfo, &in_query_info);
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_query_info);
        
        
        try {
            check_mode(in_query_info, mode, "radical");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            ///
            ///clean up stuff
            if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
            
            object_method(in_matrix,_jit_sym_lock,in_savelock);
            object_method(Y_matrix,_jit_sym_lock,Y_savelock);
            object_method(W_matrix,_jit_sym_lock,W_savelock);
            return err;
            //goto out;
        }
        
        query = jit_to_arma(mode, static_cast<t_object*>(in_matrix64), query);
        
        //std::cout << query << std::endl;
        
        //size_t mvar = (m == 0 ? query.n_rows - 1 : m);
        int nSweeps = 0;
        if(sweeps == 0) {
            nSweeps = query.n_rows - 1;
        }
        
        Radical rad(std_dev, replicates, angles, nSweeps);
        arma::mat matY, matW;
        arma::mat matYT;
        double objective = 0.;
        t_atom a[1];
    
        
        //arma::mat queryT = trans(query);

        if (seed != 0)
          mlpack::RandomSeed((size_t) seed);
        else
          mlpack::RandomSeed((size_t) std::time(NULL));
        
        try {
            rad.DoRadical(query, matY, matW);
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        matYT = trans(matY);
        for (size_t i = 0; i < matYT.n_cols; ++i)
        {
            arma::vec y = arma::vec(matYT.col(i));
            objective += rad.Vasicek(y);
        }
        
        //std::cout << matY.n_cols << " " <<  matY.n_rows << " " << matW.n_cols << " " << matW.n_rows << std::endl;
        
        
        switch(mode.get()) {
            case 0:
                Y_info.type = _jit_sym_float64;
                Y_info.flags = 0;
                Y_info.dimcount = 1;
                Y_info.planecount = matY.n_cols;
                Y_info.dim[0] = matY.n_rows;
                
                W_info.type = _jit_sym_float64;
                W_info.flags = 0;
                W_info.dimcount = 1;
                W_info.planecount = matW.n_rows;
                W_info.dim[0] = matW.n_cols;
                break;
            case 1:
                Y_info.type = _jit_sym_float64;
                Y_info.flags = 0;
                Y_info.dimcount = 2;
                Y_info.planecount = 1;
                Y_info.dim[0] = matY.n_rows;
                Y_info.dim[1] = matY.n_cols;
                
                W_info.type = _jit_sym_float64;
                W_info.flags = 0;
                W_info.dimcount = 2;
                W_info.planecount = 1;
                W_info.dim[0] = matW.n_rows;
                W_info.dim[1] = matW.n_cols;
                break;
            case 2:
                Y_info.type = _jit_sym_float64;
                Y_info.flags = 0;
                Y_info.dimcount = 2;
                Y_info.planecount = 1;
                Y_info.dim[0] = matY.n_cols;
                Y_info.dim[1] = matY.n_rows;
                
                W_info.type = _jit_sym_float64;
                W_info.flags = 0;
                W_info.dimcount = 2;
                W_info.planecount = 1;
                W_info.dim[0] = matW.n_cols;
                W_info.dim[1] = matW.n_rows;
                break;
            default:
                out_info.planecount = 1;
                break;
                
        }
            
        Y_matrix = arma_to_jit(mode, matY, static_cast<t_object*>(Y_matrix), Y_info );
        W_matrix = arma_to_jit(mode, matW, static_cast<t_object*>(W_matrix), W_info );
        
    out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        
        object_method(in_matrix,_jit_sym_lock,in_savelock);
        object_method(Y_matrix,_jit_sym_lock,Y_savelock);
        object_method(W_matrix,_jit_sym_lock,W_savelock);
        atom_setfloat(a,objective);
        outlet_anything(m_dumpoutlet, gensym("objective"), 1, a);
        return err;
    }

};


MIN_EXTERNAL(mlmat_radical);


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_radical>* job = (minwrap<mlmat_radical>*)(x);
        err = job->m_min_object.matrix_calc(x, inputs, outputs);
    }

    return err;
}


void mlmat_radical_assist(void* x, void* b, long io, long index, char* s) {
    switch(io) {
        case 1:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) dataset X");
                    break;

                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) Y");
                    break;
                case 1:
                    sprintf(s, "(matrix) W");
                    break;
                default:
                    sprintf(s, "dumpout");
                    break;
            }

    }
}
