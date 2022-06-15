/// @file mlmat.mean_shift.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved. Also based on examples provided with the mlpack library. Please see source/mlpack for license details
/// @license  Use of this source code is governed by the MIT License found in the License.md file.


#include "mlmat.hpp"
#include <mlpack/methods/mean_shift/mean_shift.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
using namespace c74::min;
using namespace c74::max;
using namespace mlpack;
using namespace mlpack::meanshift;
using namespace mlpack::kernel;
using namespace mlpack::util;


// C function declarations
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);
void mlmat_assist(void* x, void* b, long io, long index, char* s);

class mlmat_mean_shift : public mlmat_object<mlmat_mean_shift> {
public:
    MIN_DESCRIPTION	{"Mean-shift clustering. A fast implementation of mean-shift clustering using dual-tree range search. Given a dataset, this uses the mean shift algorithm to produce and return a clustering of the data."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.kmeans, mlmat.knn"};


    attribute<int> max_iterations { this, "max_iterations", 1000,
        description {"Maximum number of iterations before k-means terminates."}
    };
    
    attribute<double> radius { this, "radius", 0.0,
        description {"If the distance between two centroids is less than the given radius, one will be removed.  A radius of 0 or less means an estimate will be calculated and used for the radius."}
    };
    
    attribute<bool> force_convergence {this, "force_convergence", false,
        description {
            "If specified, the mean shift algorithm will continue running regardless of max_iterations until the clusters converge."
        }
    };

    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        arma::Mat<double> dat;
        arma::Row<size_t> assignments;
        arma::Mat<double> centroids;
        MeanShift<> mean_shift(radius, max_iterations);
        t_jit_matrix_info in_query_info, out_assignments_info, out_centroids_info;
    

        auto in_matrix = (t_object*)object_method(inputs, _jit_sym_getindex, 0);
        auto out_assignments = (t_object*)object_method(outputs, _jit_sym_getindex, 0);
        auto out_centroids = (t_object*)object_method(outputs, _jit_sym_getindex, 1);
        
        auto query_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_assignments_savelock = object_method(out_assignments, _jit_sym_lock, 1);
        auto out_centroids_savelock = object_method(out_centroids, _jit_sym_lock, 1);
        
        object_method(in_matrix, _jit_sym_getinfo, &in_query_info);
        
        t_object *in_matrix64 = convert_to_float64(in_matrix, in_query_info);
        
        try {
            check_mode(in_query_info, mode, "mean shift");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
    
        dat = jit_to_arma(mode, in_matrix64, dat);
 
        mean_shift.Cluster(dat, assignments, centroids, force_convergence);
        
        out_assignments_info = in_query_info;
        out_assignments_info.type = _jit_sym_long;
        out_assignments_info.planecount = 1;

        out_assignments = arma_to_jit(mode, assignments, out_assignments, out_assignments_info);
        
        out_centroids_info.type = _jit_sym_float64;
        out_centroids_info.dimcount = 1;
        out_centroids_info.dim[0] = centroids.n_cols;
        out_centroids_info.planecount = centroids.n_rows;
        out_centroids = arma_to_jit(mode, centroids, out_centroids,out_centroids_info);

    out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        
        object_method(in_matrix,_jit_sym_lock,query_savelock);
        object_method(out_assignments,_jit_sym_lock,out_assignments_savelock);
        object_method(out_centroids,_jit_sym_lock,out_centroids_savelock);

        return err;
    }


private:
    

    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        // add mop
        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 1, 2));
        
        // force type
        jit_mop_single_type(mop, _jit_sym_float64);
        
        //unlink dimesions between left and right i/o
        //keep planecounts same for now.
        auto output1 = object_method(mop,_jit_sym_getoutput,1);
        jit_attr_setlong(output1,_jit_sym_dimlink,0);
        auto output2 = object_method(mop,_jit_sym_getoutput,2);
        jit_attr_setlong(output2,_jit_sym_dimlink,0);
        
        //always adapt
        jit_class_addadornment(c, mop);

        // add our custom matrix_calc method
        jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
        return {};
    }};

    
    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        t_class* c = args[0];

        max_jit_class_mop_wrap(c, this_jit_class, 0);
        max_jit_class_wrap_standard(c, this_jit_class, 0);
        class_addmethod(c, (method)mlmat_assist, "assist", A_CANT, 0);


        return {};
    }};
};


MIN_EXTERNAL(mlmat_mean_shift);


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_mean_shift>* job = (minwrap<mlmat_mean_shift>*)(x);
        // call our custom matrix_calc function defined inside the C++ class
        err = job->m_min_object.matrix_calc(x, inputs, outputs);
    }
    return err;
}



void mlmat_assist(void* x, void* b, long io, long index, char* s) {
    switch(io) {
        case 1:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) query points.");
                    break;

                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) cluster assignments");
                    break;
                case 1:
                    sprintf(s, "(matrix) cluster centroids");
                    break;
                default:
                    sprintf(s, "dumpout");
                    break;
            }

    }
}





