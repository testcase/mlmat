/// @file mlmat.hoeffding_tree.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// TODO: is capable of using categorical data - this is not implemented here

#include "c74_min.h"
#include "mlmat_object.hpp"
#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>
#include <mlpack/methods/hoeffding_trees/binary_numeric_split.hpp>
#include <mlpack/methods/hoeffding_trees/information_gain.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree_model.hpp>
#include <mlpack/methods/preprocess/scaling_model.hpp>
#include <queue>

using namespace c74;
using namespace c74::min;
using namespace c74::max;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::data;
using namespace mlpack::util;

typedef std::tuple<DatasetInfo, arma::mat> TupleType;

void mlmat_assist(void* x, void* b, long io, long index, char* s);
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);


class mlmat_hoeffding_tree : public mlmat_object_writable<mlmat_hoeffding_tree, HoeffdingTreeModel> {
public:
    MIN_DESCRIPTION	{"Hoeffding Decision Tree. An implementation of Hoeffding trees, a form of streaming decision tree for classification. Given labeled data, a Hoeffding tree can be trained and saved for later use, or a pre-trained Hoeffding tree can be used for predicting the classifications of new points."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.id3_tree"};
    MIN_DISCUSSION  {"The Hoeffding tree is well suited for incremental learning meaning that rather than provide all the data at once it will refine the decision tree as it receives more data. However, it can be used as a traditional decision tree if the <at>batch_mode</at> attribute is set to 1."};
    
    // define an optional argument for setting the message
    attribute<bool> batch_mode { this, "batch_mode", false,
        description {
            "The training may be performed in batch mode (like a typical decision tree algorithm) by specifying the option, but this may not be the best option for large datasets."
        }
    };
    
    attribute<double, threadsafe::no, limit::clamp> confidence { this, "confidence", .95,
        description {
            "Confidence before splitting (between 0 and 1)."
        },
        range {0., 1.}
    };
    
    attribute<int> max_samples { this, "max_samples", 5000,
        description {
            "Maximum number of samples before splitting."
        }
    };
    
    attribute<int> check_interval { this, "check_interval", 20,
        description {
            "Number of samples required before each split check."
        }
    };
    
    attribute<int> min_samples { this, "min_samples", 20,
        description {
            "Minimum number of samples before splitting."
        }
    };
    
    attribute<min::symbol> numeric_split_strategy { this, "split_strategy", "binary",
        description {
            "The splitting strategy to use for numeric features."
        },
        range {"domingos", "binary"}
    };

    attribute<bool> info_gain { this, "info_gain", false,
        description {
            "If set, information gain is used instead of Gini impurity for calculating Hoeffding bounds."
        }
    };
    
    attribute<int> passes { this, "passes", 1,
        description {
            "Number of passes to take over the dataset."
        }
    };
    
    attribute<int> bins { this, "bins", 10,
        description {
           "If the 'domingos' split strategy is used, this specifies the number of bins for each numeric split."
        }
    };
    
    attribute<int> observations_before_binning { this, "observations_before_binning", 100,
        description {
           "If the 'domingos' split strategy is used, this specifies the number of samples observed before binning is performed."
        }
    };
    
    attribute<bool> reset_tree { this, "reset_tree", false,
        description {
            "If true, reset the tree to an empty tree before training."
            
        }
    };
    
    message<> clear { this, "clear", "clear model.",
        MIN_FUNCTION {
            m_model.model.reset();
            m_data.reset();
            m_labels.reset();
            return {};
        }
        
    };
    
    message<> train {this, "train", "Train model.",
        MIN_FUNCTION {
            size_t numClasses = 0;
            arma::Row<size_t> predictions;
            arma::mat probabilities;
            size_t correct = 0;
            t_atom a[1];
            DatasetInfo dataset_info;
            int num_passes = passes;
            arma::mat out_data;
            
            const string strategy = numeric_split_strategy.get().c_str();
            
            if(!m_labels) {
                (cerr << "no labels have been input" << endl);
                goto out;
            }
                
            if(!m_data) {
                (cerr << "no data for training" << endl);
                goto out;
            }
            try {
                CheckSameSizes(*m_data, *m_labels, "hoeffding_tree");
            } catch (std::invalid_argument& s) {
                (cerr << s.what() << endl);
                goto out;
            }
            
            //1. if model is null or type is different must remake
            
            scaler_fit(m_model, *m_data);
            out_data = scaler_transform(m_model, *m_data, out_data);
            

            if(batch_mode || (m_model.model == nullptr)) {  //need to check underlying data in same format.
            
                if(!info_gain && (strategy == "domingos")) {
                    m_model.model = std::make_unique<HoeffdingTreeModel>(HoeffdingTreeModel::GINI_HOEFFDING);
                }
                else if (info_gain && (strategy == "binary")) {
                     m_model.model = std::make_unique<HoeffdingTreeModel>(HoeffdingTreeModel::GINI_BINARY);
                }
                else if (info_gain && (strategy == "domingos")) {
                    m_model.model = std::make_unique<HoeffdingTreeModel>(HoeffdingTreeModel::INFO_HOEFFDING);
                }
                else {
                    m_model.model = std::make_unique<HoeffdingTreeModel>(HoeffdingTreeModel::INFO_BINARY);
                }
            
                numClasses = arma::max(arma::max(*m_labels)) + 1;
                
                dataset_info.SetDimensionality(m_data->n_rows);
                
                for (auto i = 0; i < dataset_info.Dimensionality(); i++) {
                    dataset_info.Type(i) = data::Datatype::numeric;
                }
                // this will reset the model and build a new one. if batch_mode is true will also train using 1 pass
                m_model.model->BuildModel(out_data, dataset_info, *m_labels, numClasses, batch_mode, confidence, max_samples, check_interval, min_samples, bins, observations_before_binning);
                num_passes--;
                
                //if batch_mode is not true need to do initial training N number of passes
                if(!batch_mode) {
                    for(auto p = 0;p < num_passes;p++) {
                        m_model.model->Train(out_data, *m_labels, false);
                    }
                }
            } else {
                
                for(auto p = 0;p < num_passes;p++) {
                    m_model.model->Train(out_data, *m_labels, false);
                }
                
            }
            
            m_model.model->Classify(out_data, predictions);
    
            for (size_t i = 0; i < m_data->n_cols; ++i) {
                if (predictions[i] == (*m_labels)[i]) {
                  ++correct;
                }
            }

            atom_setfloat(a,double(correct) / double(m_labels->n_elem));
            outlet_anything(m_dumpoutlet, gensym("accuracy"), 1, a);
        out:
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
            object_method(in2,gensym("ioproc"),jit_mop_ioproc_copy_adapt);
            object_method(in3,gensym("ioproc"),jit_mop_ioproc_copy_adapt);
            
            jit_class_addadornment(c, mop);
            
            // add our custom matrix_calc method
            jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
            return {};
        }
    };
    
    t_jit_err set_labels(t_object *matrix) {
        t_jit_err err = JIT_ERR_NONE;
        m_labels = std::make_unique<arma::Row<size_t>>();
        *m_labels = jit_to_arma(mode,matrix, *m_labels);
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
        arma::rowvec probabilities;

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

        
        query = jit_to_arma(mode, in_matrix64, query);
    

        try {
            check_mode(in_matrix_info, mode, "hoeffding tree");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        if(m_mode_changed) {
            cerr << "mode has changed must resubmit reference set" << endl;
            goto out;
        }
        
        try {
            arma::mat scaled_query;
            scaled_query = scaler_transform(m_model, query, scaled_query);
            m_model.model->Classify(scaled_query, predictions, probabilities);
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        //TODO: Fix probs when dim > 1
        out_predictions_info = in_matrix_info;
        out_probabilities_info = in_matrix_info;
        out_probabilities_info.dim[0] = probabilities.n_elem;
        out_probabilities_info.dim[1] = 1;
                
        out_predictions_info.planecount = 1;
        out_predictions_info.type = _jit_sym_long;
        out_probabilities_info.planecount = 1;
        
        if(mode == 0) {
            if(out_predictions_info.dimcount == 1) {
                out_predictions_info.dim[1] = 1;//for loops
            }
        }

        out_predictions = arma_to_jit(mode, predictions, out_predictions, out_predictions_info, true);
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
    
    void load_model_file(const atoms& args) {
        atoms f{};

        if(!args.empty()) {
            f.push_back(args[0]);
        }
        
        path p {f, path::filetype::any};

        if(p) {
            try {
                mlpack::data::Load(string(p), "hoeffding_tree", m_model, true);
            } catch (const std::runtime_error& s) {
                std::throw_with_nested(std::runtime_error("Error reading model file to disk."));
            }
        }
    }

    std::unique_ptr<arma::Mat<double>> m_data { nullptr };
    std::unique_ptr<arma::Row<size_t>> m_labels { nullptr };
};


MIN_EXTERNAL(mlmat_hoeffding_tree);

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
        
        minwrap<mlmat_hoeffding_tree>* job = (minwrap<mlmat_hoeffding_tree>*)(j);
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
        
        minwrap<mlmat_hoeffding_tree>* job = (minwrap<mlmat_hoeffding_tree>*)(j);
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
        minwrap<mlmat_hoeffding_tree>* job = (minwrap<mlmat_hoeffding_tree>*)(x);
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
