/// @file mlmat.knn.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// TODO: Other distance Metrics?

#include "c74_min.h"
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/neighbor_search/unmap.hpp>
#include <mlpack/core/util/timers.hpp>
#include <mlpack/methods/neighbor_search/ns_model.hpp>
#include <mlpack/methods/preprocess/scaling_model.hpp>
#include "mlmat_operator.hpp"
#include <string>

using namespace c74;
using namespace c74::min;
using namespace c74::max;
using namespace mlpack::neighbor;
using namespace mlpack::tree;
using namespace mlpack::metric;
using namespace mlpack::util;
using namespace mlpack::data;

typedef NSModel<NearestNeighborSort> KNNModel;
// C function declarations
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);
void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);


class mlmat_knn : public mlmat_operator_autoscale<mlmat_knn, KNNModel>
{
public:
    MIN_DESCRIPTION	    {"K-nearest neighbor search. An implementation of k-nearest-neighbor search using single-tree and dual-tree algorithms. Given a set of reference points and query points, this can find the k nearest neighbors in the reference set of each query point using trees; trees that are built can be saved for future use."};
    MIN_TAGS		    {"ML"};
    MIN_AUTHOR		    {"Todd Ingalls"};
    MIN_RELATED		    {"mlmat.kfn, mlmat.svm"};

    attribute<int> neighbors { this, "neighbors", 1,
        description {
            "Number of neighbors to query. "
        },
        setter { MIN_FUNCTION {
            double value = args[0];
            
            if (value < 1) {
                value = 1;
            }
            return {value};
        }}
    };

    attribute<double> epsilon { this, "epsilon", 0.,
        description {
            "If specified, will search with given relative error."
        },
        title {
            "Relative Error."
        },
        setter { MIN_FUNCTION {
            double value = args[0];
            
            if (value > 1) {
                value = 1;
            }
            return {value};
        }}
    };
    
    attribute<min::symbol> tree_type { this, "tree_type", "kd",
        description {
            "Type of tree to use"
        },
        range {"kd", "vp", "rp", "max-rp", "ub", "cover", "r", "r-star", "x", "ball", "hilbert-r", "r-plus", "r-plus-plus", "spill", "oct"}
    };
    
    attribute<min::symbol> algorithm { this, "algorithm", "dual_tree",
        description {
            "Type of neighbor search"
        },
        range {"naive", "single_tree", "dual_tree", "greedy"}
    };
    
    attribute<int> leaf_size { this, "leaf_size", 20,
        description {
            "Leaf size for tree building must be positive. Used for kd-trees, vp trees, random projection trees, UB trees, R trees, R* trees, X trees, Hilbert R trees, R+ trees, R++ trees, spill trees, and octrees."
        },
        setter { MIN_FUNCTION {
            double value = args[0];
            
            if (value < 1) {
                value = 1;
            }
            return {value};
        }}
    };
    
    attribute<double> tau { this, "tau", 0,
        description {
            "Overlapping size (only valid for spill trees). Must be positive."
        },
        setter { MIN_FUNCTION {
            double value = args[0];
            
            if (value < 0.) {
                value = 0.;
            }
            return {value};
        }}
    };
    
    attribute<double> rho { this, "rho", 0.7,
        description {
            "Balance threshold (only valid for spill trees). Range 0 to 1."
        },
        range {0., 1.}
    };
    
    attribute<bool> random_basis { this, "random_basis", false,
        description {
            "Before tree-building, project the data onto a random orthogonal basis."
        }
    };
    
    attribute<int> seed { this, "seed", 0,
        description {
            "Random seed if random basis being used. 0 indicates no seed."
        }
    };
    
    message<> clear { this, "clear", "clear model.",
        MIN_FUNCTION {
            m_model.model.reset();
            return {};
        }
        
    };
    

    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_query_info, out_neighbors_info, out_distances_info;
        arma::mat query;
        arma::Mat<size_t> resulting_neighbors;
        arma::mat resulting_distances;
        
        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto out_neighbors = object_method(outputs, _jit_sym_getindex, 0);
        auto out_distances = object_method(outputs, _jit_sym_getindex, 1);
    
        auto query_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_neighbors_savelock = object_method(out_neighbors, _jit_sym_lock, 1);
        auto out_distances_savelock = object_method(out_distances, _jit_sym_lock, 1);
        
        object_method(in_matrix, _jit_sym_getinfo, &in_query_info);
        
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_query_info);

        if(m_model.model == nullptr || m_model.model->Dataset().is_empty()) {
            (cerr << "no reference set exists" << endl);
            goto out;
        }
       
        if(m_mode_changed) {
            (cerr << "mode has changed must resubmit reference set" << endl);
            goto out;
        }
        
        object_method(in_matrix64, _jit_sym_getinfo, &in_query_info);
        
        try {
            check_mode(in_query_info, mode, "knn");
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        query = jit_to_arma(mode, static_cast<t_object*>(in_matrix64), query);
        
        try {
            CheckSameDimensionality(query, m_model.model->Dataset(), "knn");
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        if(neighbors > m_model.model->Dataset().n_cols) {
             (cerr << "number of neighbors requested(" << neighbors << ") exceeds entries in reference set (" << m_model.model->Dataset().n_cols << ")" << endl);
             goto out;
        }
        
        try {
            Timers u = Timers();
            arma::mat scaled_query;
            scaled_query = scaler_transform(m_model, query, scaled_query);
            m_model.model->Search(u, std::move(scaled_query), neighbors, resulting_neighbors, resulting_distances);
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        out_neighbors_info.planecount = m_model.dimcount;
        out_distances_info.planecount = 1;
        out_neighbors_info.type = _jit_sym_long;
        out_distances_info.type = _jit_sym_float64;

        if(mode == 0) {
            out_neighbors_info.dimcount = in_query_info.dimcount + (neighbors > 1 ? 1 : 0);
            out_distances_info.dimcount = out_neighbors_info.dimcount;
            switch(out_neighbors_info.dimcount) {
                case 1:
                    out_neighbors_info.dim[0] = in_query_info.dim[0];;
                    out_neighbors_info.dim[1] = 1;
                    out_distances_info.dim[0] = in_query_info.dim[0];
                    out_distances_info.dim[1] = 1;
                    break;
                case 2:
                    out_neighbors_info.dim[0] = in_query_info.dim[0];
                    out_neighbors_info.dim[1] = neighbors;
                    out_distances_info.dim[0] = in_query_info.dim[0];
                    out_distances_info.dim[1] = neighbors;
                    break;
                case 3:
                    out_neighbors_info.dim[0] = in_query_info.dim[0];
                    out_neighbors_info.dim[1] = in_query_info.dim[1];
                    out_neighbors_info.dim[2] = neighbors;
                    out_distances_info.dim[0] = in_query_info.dim[0];
                    out_distances_info.dim[1] = in_query_info.dim[1];
                    out_distances_info.dim[2] = neighbors;
                    break;
                default:
                    break;
            }
        }
        
        out_neighbors = arma_to_jit(mode, resulting_neighbors, static_cast<t_object*>(out_neighbors), out_neighbors_info, true, m_model.dim0);
        out_distances = arma_to_jit(mode, resulting_distances, static_cast<t_object*>(out_distances), out_distances_info);

    out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        
        object_method(in_matrix,_jit_sym_lock,query_savelock);
        object_method(out_neighbors,_jit_sym_lock,out_neighbors_savelock);
        object_method(out_distances,_jit_sym_lock,out_distances_savelock);

        return err;
    }

    t_jit_err process_reference_set_matrix(t_object *matrix) {
        t_jit_matrix_info minfo;
        t_jit_err err = JIT_ERR_NONE;
        arma::mat dat;
        arma::mat out_data;
        Timers u = Timers();
        NeighborSearchMode search_mode = DUAL_TREE_MODE;
        KNNModel::TreeTypes tree = KNNModel::KD_TREE;
        const string algorithm_string = algorithm.get().c_str();
        const string tree_type_string = tree_type.get().c_str();
        
        long savelock = (long) object_method(matrix, _jit_sym_lock, 1);
        object_method(matrix, _jit_sym_getinfo, &minfo);
        
        if(minfo.dimcount > 2) {
            cerr << "expecting 1d or 2d matrix, received " << minfo.dimcount << "d matrix" << endl;
            err = JIT_ERR_INVALID_INPUT;
            goto out;
        }
        
        m_model.dimcount = minfo.dimcount;
        
        if(minfo.dimcount == 2) {
            m_model.dim0 = minfo.dim[0];
        }

        try {
            check_mode(minfo, mode, "knn");
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
                
        dat = jit_to_arma(mode, matrix, dat);
        
        m_model.model = std::make_unique<KNNModel>();
        
        if (algorithm_string == "naive")
          search_mode = NAIVE_MODE;
        else if (algorithm_string == "single_tree")
          search_mode = SINGLE_TREE_MODE;
        else if (algorithm_string == "dual_tree")
          search_mode = DUAL_TREE_MODE;
        else if (algorithm_string == "greedy")
          search_mode = GREEDY_SINGLE_TREE_MODE;
        
        if (tree_type_string == "kd")
          tree = KNNModel::KD_TREE;
        else if (tree_type_string == "cover")
          tree = KNNModel::COVER_TREE;
        else if (tree_type_string == "r")
          tree = KNNModel::R_TREE;
        else if (tree_type_string == "r-star")
          tree = KNNModel::R_STAR_TREE;
        else if (tree_type_string == "ball")
          tree = KNNModel::BALL_TREE;
        else if (tree_type_string == "x")
          tree = KNNModel::X_TREE;
        else if (tree_type_string == "hilbert-r")
          tree = KNNModel::HILBERT_R_TREE;
        else if (tree_type_string == "r-plus")
          tree = KNNModel::R_PLUS_TREE;
        else if (tree_type_string == "r-plus-plus")
          tree = KNNModel::R_PLUS_PLUS_TREE;
        else if (tree_type_string == "spill")
          tree = KNNModel::SPILL_TREE;
        else if (tree_type_string == "vp")
          tree = KNNModel::VP_TREE;
        else if (tree_type_string == "rp")
          tree = KNNModel::RP_TREE;
        else if (tree_type_string == "max-rp")
          tree = KNNModel::MAX_RP_TREE;
        else if (tree_type_string == "ub")
          tree = KNNModel::UB_TREE;
        else if (tree_type_string == "oct")
          tree = KNNModel::OCTREE;
        
        m_model.model->TreeType() = tree;
        m_model.model->RandomBasis() = random_basis;
        m_model.model->LeafSize() = leaf_size;
        m_model.model->Tau() = tau;
        m_model.model->Rho() = rho;

        if (seed != 0) {
          mlpack::math::RandomSeed((size_t) seed);
        } else {
          mlpack::math::RandomSeed((size_t) std::time(NULL));
        }
        
        scaler_fit(m_model, dat);
        out_data = scaler_transform(m_model, dat, out_data);
        
        m_model.model->BuildModel(u, std::move(out_data), search_mode, epsilon);
        m_mode_changed = false;
    out:
        
        object_method(matrix, _jit_sym_lock, savelock);
        return err;
    }
    
    
private:

    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        // add mop
        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 2, 2));
        
        // force type
        jit_mop_single_type(mop, _jit_sym_float64);
        

        auto ref_input = jit_object_method(mop,_jit_sym_getinput,2);
        //always adapt
        jit_object_method(ref_input,_jit_sym_ioproc,jit_mop_ioproc_copy_adapt);

        jit_class_addadornment(c, mop);
        
        // add our custom matrix_calc method
        jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
        return {};
    }};
    

    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        
        max_jit_class_mop_wrap(c, this_jit_class, 0);
        max_jit_class_wrap_standard(c, this_jit_class, 0);
        class_addmethod(c, (method)max_mlmat_jit_matrix, "jit_matrix", A_GIMME, 0);
        class_addmethod(c, (method)mlmat_assist, "assist", A_CANT, 0);

        return {};
    }};

};


MIN_EXTERNAL(mlmat_knn);

void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv)
{
    //found need to call this first or the
    // info for the incoming matrix is incorrect
    max_jit_mop_jit_matrix(x,s,argc,argv);
    //only want second inlet
    if (max_jit_obex_inletnumber_get(x) == 1)
    {
        t_jit_err err = JIT_ERR_NONE;
        void *j = nullptr;
        
        void *p, *m, *mop;
        
        if (!(mop=max_jit_obex_adornment_get(x,_jit_sym_jit_mop)))
            jit_error_code(x,err);

        p = object_method(mop,_jit_sym_getinput,2);
        m = object_method(p,_jit_sym_getmatrix);
        j = max_jit_obex_jitob_get(x);
    
        minwrap<mlmat_knn>* job = (minwrap<mlmat_knn>*)(j);
        err = job->m_min_object.process_reference_set_matrix(static_cast<t_object*>(m));

        jit_error_code(x,err);
    }
}

t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_knn>* job = (minwrap<mlmat_knn>*)(x);
        err = job->m_min_object.matrix_calc(x, inputs, outputs);
    }

    return err;
}


void mlmat_assist(void* x, void* b, long io, long index, char* s) {
    switch(io) {
        case 1:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) query points");
                    break;

                case 1:
                    sprintf(s, "(matrix) reference dataset");
                    break;

                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) neighbors");
                    break;

                case 1:
                    sprintf(s, "(matrix) distances");
                    break;

                default:
                    sprintf(s, "dumpout");
                    break;
            }
    }
}




