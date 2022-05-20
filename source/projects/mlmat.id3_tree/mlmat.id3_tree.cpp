/// @file mlmat.id3_tree.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
//
// TODO: use mop to filter more that 2d
//
//
//

#include "c74_min.h"
#include "mlmat_object.hpp"
#include <mlpack/methods/decision_tree/decision_tree.hpp>

using namespace c74::min;
using namespace c74::max;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

//using this to be able to serialized model
class DecisionTreeModel {
public:
    // The tree itself, left public for direct access by this program.
    DecisionTree<> tree;
    DatasetInfo info;
    
    // Create the model.
    DecisionTreeModel() { /* Nothing to do. */ }
    
    // Serialize the model.
    template<typename Archive>
    void serialize(Archive& ar, const unsigned int /* version */) {
        ar(CEREAL_NVP(tree));
        ar(CEREAL_NVP(info));
        
    }
};

void mlmat_assist(void* x, void* b, long io, long index, char* s);
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);



class mlmat_id3_tree : public mlmat_object_writable<mlmat_id3_tree, DecisionTreeModel> {
public:
    MIN_DESCRIPTION	{"ID3 Decision Tree Classifier."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR      {"Todd Ingalls"};
    MIN_RELATED		{"mlmat.hoeffding_tree"};
    
    attribute<int> minimum_leaf_size { this, "minimum_leaf_size", 10,
        description {
            "Minimum leaf size for tree. "
        },
        setter { MIN_FUNCTION {
            double value = args[0];
            if(value < 1) {
                value = 1;
            }
            return {value};
        }}
    };
    
    // TODO: This value will not be displayed in max UI.
    attribute<double> minimum_gain_split { this, "minimum_gain_split", 0.000001,
        description {
            "Minimum gain for node splitting."
        },
        setter { MIN_FUNCTION {
            double value = args[0];
            if(value < 1e-10) {
                value = 1e-10; //just guessing here
            }
            return {value};
        }}
    };
    
    attribute<int> maximum_depth { this, "maximum_depth", 0,
        description {
            "Maximum depth of the tree.(0 means no limit). "
        },
        title {
            "Maximum Depth"
        },
        setter { MIN_FUNCTION {
            double value = args[0];
            if(value < 1e-10) {
                value = 1e-10; //just guessing here
            }
            return {value};
        }}
    };
    
    
    message<> train {this, "train", "Train model.",
        MIN_FUNCTION {
            size_t numClasses = 0;
            arma::Row<size_t> predictions;
            arma::mat probabilities;
            arma::mat out_data;
            
            size_t correct = 0;
            t_atom a[1];
            
            if(!m_labels) {
                (cerr << "no labels have been input" << endl);
                goto out;
            }
            
            if(!m_data) {
                (cerr << "no data for training" << endl);
                goto out;
            }
            
            try {
                CheckSameSizes(*m_data, *m_labels, "id3_tree");
            } catch (std::invalid_argument& s) {
                (cerr << s.what() << endl);
                goto out;
            }
            
            numClasses = arma::max(arma::max(*m_labels)) + 1;
            
            m_model.model = std::make_unique<DecisionTreeModel>();
            
            scaler_fit(m_model, *m_data);
            out_data = scaler_transform(m_model, *m_data, out_data);
            
            m_model.model->tree = DecisionTree<>(out_data, *m_labels, numClasses, minimum_leaf_size, minimum_gain_split, maximum_depth);
            
            m_model.model->tree.Classify(out_data, predictions, probabilities);
            
            for (size_t i = 0; i < m_data->n_cols; ++i) {
                if (predictions[i] == (*m_labels)[i]) {
                    ++correct;
                }
            }
           
            
            atom_setfloat(a,double(correct) / double(m_data->n_cols));
            outlet_anything(m_dumpoutlet, gensym("accuracy"), 1, a);
        out:
            return {};
        },
    };
    
    message<> clear { this, "clear", "clear model.",
        MIN_FUNCTION {
            m_model.model.reset();
            m_data.reset();
            m_labels.reset();
            return {};
        }
        
    };
    
    // post to max window == but only when the class is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            t_class* c = args[0];
            max_jit_class_mop_wrap(c, this_jit_class, 0);
            max_jit_class_wrap_standard(c, this_jit_class, 0);
            class_addmethod(c, (method)max_mlmat_jit_matrix, "jit_matrix", A_GIMME, 0);
            class_addmethod(c, (method)mlmat_assist, "assist", A_CANT, 0);
            return {};
        }
    };
    
    message<> jitclass_setup {this, "jitclass_setup",
        MIN_FUNCTION {
            t_class* c = args[0];
            t_atom long_type[1];
            
            atom_setsym(long_type, _jit_sym_long);
            
            // add mop
            t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 3, 2));
            
            // force type
            jit_mop_single_type(mop, _jit_sym_float64);
            
            auto in2 = object_method(mop,_jit_sym_getinput,2);
            auto in3 = object_method(mop,_jit_sym_getinput,3);
            
            jit_attr_setlong(in2,_jit_sym_dimlink,0);
            jit_attr_setlong(in2,_jit_sym_typelink,0);
            jit_attr_setlong(in3,_jit_sym_dimlink,0);
            jit_attr_setlong(in3,_jit_sym_typelink,0);
            
            object_method_typed(in3, _jit_sym_types, 1, long_type, NULL);
            
            //always adapt
            object_method(in2,_jit_sym_ioproc,jit_mop_ioproc_copy_adapt);
            object_method(in3,_jit_sym_ioproc,jit_mop_ioproc_copy_adapt);
            
            jit_class_addadornment(c, mop);
            
            // add our custom matrix_calc method
            jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
            return {};
        }
    };
    
    t_jit_err set_labels(t_object *matrix) {
        t_jit_err err = JIT_ERR_NONE;
        m_labels = std::make_unique<arma::Row<size_t>>();
        *m_labels = jit_to_arma(mode, matrix, *m_labels);
        return err;
    }
    
    t_jit_err set_data(t_object *matrix) {
        t_jit_err err = JIT_ERR_NONE;
        m_data = std::make_unique<arma::Mat<double>>();
        *m_data = jit_to_arma(mode, matrix, *m_data);
        m_mode_changed = false;

        return err;
    }
    
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_matrix_info, out_predictions_info, out_probabilities_info;
        arma::mat query;
        arma::Row<size_t> predictions;
        arma::mat probabilities;
        
        auto in_matrix = (t_object*)object_method(inputs, _jit_sym_getindex, 0);
        auto in_data = (t_object*)object_method(inputs, _jit_sym_getindex, 1);
        auto in_labels = (t_object*)object_method(inputs, _jit_sym_getindex, 2);
        
        auto out_predictions = (t_object*)object_method(outputs, _jit_sym_getindex, 0);
        auto out_probabilities = (t_object*)object_method(outputs, _jit_sym_getindex, 1);
        
        auto in_matrix_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto in_data_savelock = object_method(in_data, _jit_sym_lock, 1);
        auto in_labels_savelock = object_method(in_labels, _jit_sym_lock, 1);
        auto out_predictions_savelock = object_method(out_predictions, _jit_sym_lock, 1);
        auto out_probabilities_savelock = object_method(out_probabilities, _jit_sym_lock, 1);
        
        object_method(in_matrix, _jit_sym_getinfo, &in_matrix_info);
        
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_matrix_info);
        
        if(!m_model.model) {
            (cerr << "no model trained." << endl);
            goto out;
        }
        
        try {
            check_mode(in_matrix_info, mode, "id3 tree classifier");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        query = jit_to_arma(mode, in_matrix64, query);
        
        
        if(!m_data) {
            (cerr << "no reference set exists" << endl);
            goto out;
        }
        
        if(m_mode_changed) {
            (cerr << "mode has changed must resubmit training data and retrain" << endl);
            goto out;
        }
        
        try {
            arma::mat scaled_query;
            scaled_query = scaler_transform(m_model, query, scaled_query);
            m_model.model->tree.Classify(std::move(scaled_query), predictions, probabilities);
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        out_predictions_info.planecount = 1;//in_data_info.dimcount;
        out_predictions_info.type = _jit_sym_long;
        out_probabilities_info.planecount = 1;
        out_probabilities_info.type = _jit_sym_float64;
        
        
        switch(mode.get()) {
            case 0:
                out_predictions_info.dimcount = in_matrix_info.dimcount;
                out_predictions_info.dim[0] = in_matrix_info.dim[0];
                out_predictions_info.dim[1] = (out_predictions_info.dimcount == 1 ? 1 : in_matrix_info.dim[2]);
                out_predictions_info.planecount = predictions.n_rows;
                out_probabilities_info.dimcount = in_matrix_info.dimcount;
                out_probabilities_info.dim[0] = in_matrix_info.dim[0];
                out_probabilities_info.dim[1] = (out_probabilities_info.dimcount == 1 ? 1 : in_matrix_info.dim[2]);
                out_probabilities_info.planecount = probabilities.n_rows;
                break;
            case 1:
                out_predictions_info.planecount = 1;
                out_predictions_info.dim[0] = predictions.n_rows;
                out_predictions_info.dim[1] = predictions.n_cols;
                out_probabilities_info.planecount = 1;
                out_probabilities_info.dim[0] = probabilities.n_rows;
                out_probabilities_info.dim[1] = probabilities.n_cols;
                break;
            case 2:
                out_predictions_info.planecount = 1;
                out_predictions_info.dim[0] = predictions.n_cols;
                out_predictions_info.dim[1] = predictions.n_rows;
                out_probabilities_info.planecount = 1;
                out_probabilities_info.dim[0] = probabilities.n_cols;
                out_probabilities_info.dim[1] = probabilities.n_rows;
                break;
            default:
                break;
                
        }
        
        out_predictions = arma_to_jit(mode, predictions, out_predictions, out_predictions_info);
        out_probabilities = arma_to_jit(mode, probabilities, out_probabilities, out_probabilities_info);
        
    out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        object_method(in_matrix,_jit_sym_lock,in_matrix_savelock);
        object_method(in_data,_jit_sym_lock,in_data_savelock);
        object_method(in_labels,_jit_sym_lock,in_labels_savelock);
        object_method(out_predictions,_jit_sym_lock,out_predictions_savelock);
        object_method(out_probabilities,_jit_sym_lock,out_probabilities_savelock);
        
        return err;
    }
    
private:

    std::unique_ptr<arma::Mat<double>> m_data { nullptr };
    std::unique_ptr<arma::Row<size_t>> m_labels { nullptr };
};

MIN_EXTERNAL(mlmat_id3_tree);

void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv) {
    //found need to call this first or the
    // info for the incoming matrix is incorrect
    max_jit_mop_jit_matrix(x,s,argc,argv);
    //only want second inlet
    if (max_jit_obex_inletnumber_get(x) == 1)
    {
        t_jit_err err = JIT_ERR_NONE;
        void *p, *m, *mop;
        t_jit_matrix_info minfo;
        void *j = nullptr;
        
        if (!(mop=max_jit_obex_adornment_get(x,_jit_sym_jit_mop)))
            jit_error_code(x,err);
        
        p = object_method((t_object*)mop,_jit_sym_getinput,2);
        m = object_method((t_object*)p,_jit_sym_getmatrix);
        object_method((t_object*)m, _jit_sym_getinfo, &minfo);
        j = max_jit_obex_jitob_get(x);
        
        minwrap<mlmat_id3_tree>* job = (minwrap<mlmat_id3_tree>*)(j);
        err = job->m_min_object.set_data((t_object*)m);
        
        if(err)
            jit_error_code(x,err);
        
    } else if(max_jit_obex_inletnumber_get(x) == 2) {
        t_jit_err err = JIT_ERR_NONE;
        void *p, *m, *mop;
        t_jit_matrix_info minfo;
        void *j = nullptr;
        
        if (!(mop=max_jit_obex_adornment_get(x,_jit_sym_jit_mop)))
            jit_error_code(x,err);
        
        p = object_method((t_object*)mop,_jit_sym_getinput,3);
        m = object_method((t_object*)p,_jit_sym_getmatrix);
        object_method((t_object*)m, _jit_sym_getinfo, &minfo);
        
        j = max_jit_obex_jitob_get(x);
        
        minwrap<mlmat_id3_tree>* job = (minwrap<mlmat_id3_tree>*)(j);
        err = job->m_min_object.set_labels((t_object*)m);
        
        if(err)
            jit_error_code(x,err);
    }
}


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_id3_tree>* job = (minwrap<mlmat_id3_tree>*)(x);
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
                    sprintf(s, "(matrix) testing points.");
                    break;
                    
                case 1:
                    sprintf(s, "(matrix) training dataset.");
                    break;
                    
                case 2:
                    sprintf(s, "(matrix) training labels.");
                    break;
                    
                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) predictions");
                    break;
                    
                case 1:
                    sprintf(s, "(matrix) probabilities");
                    break;
                    
                default:
                    sprintf(s, "dumpout");
                    break;
            }
        default:
            break;
    }
}
