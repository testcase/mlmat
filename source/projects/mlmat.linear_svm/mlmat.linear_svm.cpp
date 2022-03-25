/// @file mlmat.linear_svm.cpp
/// @ingroup mlmat
/// @copyright Copyright 2018 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"

#include <mlpack/core.hpp>
#include <mlpack/methods/linear_svm/linear_svm.hpp>
#include <ensmallen.hpp>
#include "mlmat_operator.hpp"

#include <string>

using namespace c74::min;
using namespace c74::max;
using namespace mlpack;
using namespace mlpack::svm;
using namespace mlpack::util;
using namespace ens;


// C function declarations
void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);

class LinearSVMModel
{
 public:
  arma::Col<size_t> mappings;
  LinearSVM<> svm;

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(mappings));
    ar(CEREAL_NVP(svm));
  }
};


class mlmat_linear_svm : public mlmat_operator_autoscale<mlmat_linear_svm>
{
public:
    MIN_DESCRIPTION	{"Linear SVM. An implementation of linear SVM for multiclass classification. Given labeled data, a model can be trained and saved for future use; or, a pre-trained model can be used to classify new points."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.linear_regression, mlmat.mlp_classifier"};

    inlet<>  input1 {this, "(matrix) Testing dataset", "matrix"};
    inlet<>  input2 {this, "(matrix) Training dataset.", "matrix"};
    inlet<>  input3 {this, "(matrix) Training labels.", "matrix"};
    outlet<> output1 {this, "(matrix) Class probabilities for each test point.", "matrix"};
    outlet<> output2 {this, "(matrix) Class scores for each test point.", "matrix"};

    attribute<double> lambda { this, "lambda", .0001,
        description { "L2-regularization parameter for training." }
    };
    
    attribute<double> delta { this, "delta", 1.,
        description { "Margin of difference between correct class and other classes." }
    };
    
    attribute<int> num_classes { this, "num_classes", 0,
        description { "Margin of difference between correct class and other classes." }
    };
    
    attribute<bool> no_intercept { this, "no_intercept", false,
        description { "Do not add the intercept term to the model." }
    };
    
    attribute<min::symbol> optimizer { this, "optimizer", "lbfgs",
        description{ "Optimizer to use for training."},
        range { "lbfgs", "psgd" }
    };

    attribute<double> tolerance { this, "tolerance", 1e-10,
        description { "Convergence tolerance for optimizer." }
    };
    
    attribute<int> max_iterations { this, "max_iterations", 10000,
        description { "Maximum iterations for optimizer (0 indicates no limit)." }
    };
    
    attribute<double> step_size { this, "step_size", 0.01,
        description { "Step size for parallel SGD optimizer." }
    };
    
    attribute<bool> shuffle { this, "shuffle", false,
        description { "Shuffle the order in which data points are visited for parallel SGD." }
    };
    
    attribute<int> epochs { this, "epochs", 50,
        description { "Maximum number of full epochs over dataset for psgd." }
    };
    
    attribute<int> seed { this, "seed", 0,
        description {
            "Random seed if random basis being used. 0 indicates no seed."
        }
    };
    
    attribute<min::symbol> file {this, "file", k_sym__empty,
        description {
            "File"
        },
        title {
            "File"
        },
        setter { MIN_FUNCTION {
            if(args[0] != k_sym__empty) {
                load_model_file(args);
            }
            return args;
        }}
    };
    
    
    message<> train {this, "train", "Train model",
        MIN_FUNCTION {
            size_t numClasses = 0;
            arma::Row<size_t> labels;
            arma::mat out_data;
            
            if(!m_labels) {
                (cerr << "no labels have been input" << endl);
                goto out;
            }
                
            if(!m_data) {
                (cerr << "no data for training" << endl);
                goto out;
            }
            
            if(m_labels->n_elem != m_data->n_cols) {
                (cerr << "mismatch between number of labels (" << m_labels->n_elem << ") and data (" << m_data->n_cols << ")." << endl);
                goto out;
            }
            
            if (seed == 0) {
              mlpack::math::RandomSeed(time(NULL));
            } else {
              mlpack::math::RandomSeed((size_t) seed);
            }
            
            m_model.model = std::make_unique<LinearSVMModel>();
            
            data::NormalizeLabels(*m_labels, labels, m_model.model->mappings);
            
            numClasses = (num_classes == 0) ? m_model.model->mappings.n_elem : num_classes;
            m_model.model->svm.Lambda() = lambda;
            m_model.model->svm.Delta() = delta;
            m_model.model->svm.NumClasses() = numClasses;
            m_model.model->svm.FitIntercept() = no_intercept;
            
            scaler_fit(m_model, *m_data);
            out_data = scaler_transform(m_model, *m_data, out_data);
            
            if(optimizer.get() == "lbfgs") {
                ens::L_BFGS lbfgsOpt;
                lbfgsOpt.MaxIterations() = max_iterations;
                lbfgsOpt.MinGradientNorm() = tolerance;
                //(cout << "Training model with L-BFGS optimizer." << endl);
                // This will train the model.
                m_model.model->svm.Train(std::move(out_data), labels, numClasses, lbfgsOpt);
            } else if (optimizer.get() == "psgd") {
                const size_t maxIt = epochs * m_labels->n_cols;
                ens::ConstantStep decayPolicy(step_size);
                // can OPENMP cause problems with MAX?
                //(cout << "Training model with P-SGD optimizer." << endl);
                ens::ParallelSGD<ens::ConstantStep> psgdOpt(maxIt, std::ceil((float) m_data->n_cols), tolerance, shuffle, decayPolicy);
                // This will train the model.
                m_model.model->svm.Train(std::move(out_data), labels, numClasses, psgdOpt);
            }
            
        out:
            return {};
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
    
    
    message<> write {this, "write",
        MIN_FUNCTION {
           try {
               m_model.autoscale = autoscale;
               save_model_file(args, m_model, "linear_svm");
           } catch (const std::runtime_error& s) {
               (cerr << s.what() << endl);
           }
           return {};
        }
    };
    
    message<> read {this, "read",
        MIN_FUNCTION {
            load_model_file(args);
            autoscale = m_model.autoscale;
            m_mode_changed = false;
            return {};
        }
    };
    
    void load_model_file(const atoms& args) {
        atoms f{};

        if(!args.empty()) {
            f.push_back(args[0]);
        }
     
        path p {f, path::filetype::any};

        if(p) {
            try {
                mlpack::data::Load(string(p), "linear_svm", m_model, true);
            } catch (const std::runtime_error& s) {
                std::throw_with_nested(std::runtime_error("Error reading model file to disk."));
            }
        }
    }
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_query_info, out_predictions_info, out_scores_info;
        arma::mat query;
        arma::Row<size_t> predictions;
        arma::Mat<double> scores;
        arma::Row<size_t> predicted_labels;

        size_t training_dimensionality;
        size_t numClasses;
    
        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto out_predictions = object_method(outputs, _jit_sym_getindex, 0);
        auto out_scores = object_method(outputs, _jit_sym_getindex, 1);
        
        auto in_matrix_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_predictions_savelock = object_method(out_predictions, _jit_sym_lock, 1);
        auto out_scores_savelock = object_method(out_scores, _jit_sym_lock, 1);
        
        object_method(in_matrix, _jit_sym_getinfo, &in_query_info);
        
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_query_info);
        
        if(!m_model.model) {
            (cerr << "no model trained." << endl);
            goto out;
        }
        
        try {
            check_mode(in_query_info, mode, "linear svm");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        numClasses = m_model.model->svm.NumClasses();
        
        // Set the dimensionality according to fitz intercept.
        if (no_intercept) {
            training_dimensionality = m_model.model->svm.Parameters().n_rows - 1;
        } else {
            training_dimensionality = m_model.model->svm.Parameters().n_rows;
        }

        query = jit_to_arma(mode, in_matrix64, query);
        
        try {
            CheckSameDimensionality(query, training_dimensionality, "linear svm");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        try {
            arma::mat scaled_query;
            scaled_query = scaler_transform(m_model, query, scaled_query);
            m_model.model->svm.Classify(std::move(scaled_query), predicted_labels, scores);
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        
        
        //map normalized labels back to original labels.
        data::RevertLabels(predicted_labels, m_model.model->mappings, predictions);
        out_predictions_info = in_query_info;
        out_predictions_info.planecount = 1;
        
        out_scores_info = in_query_info;
        out_scores_info.planecount = 1;
     
        out_predictions = arma_to_jit(mode, predictions, static_cast<t_object*>(out_predictions), out_predictions_info);
        out_scores = arma_to_jit(mode, scores, static_cast<t_object*>(out_scores), out_scores_info);
    
    out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        object_method(in_matrix,_jit_sym_lock,in_matrix_savelock);
        object_method(out_predictions,_jit_sym_lock,out_predictions_savelock);
        object_method(out_scores,_jit_sym_lock,out_scores_savelock);

        return err;
    }
    
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
    private:
    // override jitclass_setup so we can have our own matrix_calc. jitclass_setup is called first (and only once when the object is loaded for the first time) during the intitialization of the object.
    message<> jitclass_setup {this, "jitclass_setup",
        MIN_FUNCTION {
            t_class* c = args[0];
            t_atom long_type[1];
            
            atom_setsym(long_type, _jit_sym_long);
        
            // add mop
            t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 3, 2));

            // force type
            jit_mop_single_type(mop, _jit_sym_float64);
        
            void* in2 = object_method(mop,_jit_sym_getinput,2);
            void* in3 = object_method(mop,_jit_sym_getinput,3);
            
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
    
    message<> maxob_setup {this, "maxob_setup",
        MIN_FUNCTION {
            t_object* mob = maxob_from_jitob(maxobj());
            m_dumpoutlet = max_jit_obex_dumpout_get(mob);
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
    
    mlmat_serializable_model<LinearSVMModel> m_model;
    std::unique_ptr<arma::Mat<double>> m_data { nullptr };
    std::unique_ptr<arma::Row<size_t>> m_labels { nullptr };
};

MIN_EXTERNAL(mlmat_linear_svm);

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
        
        minwrap<mlmat_linear_svm>* job = (minwrap<mlmat_linear_svm>*)(j);
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
        
        minwrap<mlmat_linear_svm>* job = (minwrap<mlmat_linear_svm>*)(j);
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
        minwrap<mlmat_linear_svm>* job = (minwrap<mlmat_linear_svm>*)(x);
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
                    sprintf(s, "(matrix) points");
                    break;

                case 1:
                    sprintf(s, "(matrix) training dataset");
                    break;
                    
                case 2:
                    sprintf(s, "(matrix) training labels");
                    break;

                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) predicted labels");
                    break;

                case 1:
                    sprintf(s, "(matrix) label scores");
                    break;

                default:
                    sprintf(s, "dumpout");
                    break;
            }

    }
}


