/// @file mlmat.linear_regression.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include "mlmat_operator.hpp"

using namespace c74::min;
using namespace c74::max;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::util;

void mlmat_assist(void* x, void* b, long io, long index, char* s);
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);

class mlmat_linear_regression : public mlmat_operator_autoscale<mlmat_linear_regression, LinearRegression> {
public:
    MIN_DESCRIPTION {"A Linear Regression Model. An implementation of simple linear regression and ridge regression using ordinary least squares.  Given a dataset and responses, a model can be trained and saved for later use, or a pre-trained model can be used to output regression predictions for a test set."};
    MIN_TAGS        {"ML"};
    MIN_AUTHOR      {"Todd Ingalls"};
    MIN_RELATED     {"mlmat.gmm"};

    attribute<double> lambda { this, "lambda", 0.0,
        description { "Tikhonov regularization for ridge regression. If 0, the method reduces to linear regression."},
    };
    
    message<> train {this, "train", "Train model.",
        MIN_FUNCTION {
            arma::Row<size_t> predictions;
            arma::mat probabilities;
            t_atom a[1];
            arma::vec params;
            if(!m_responses) {
                (cerr << "no responses have been input" << endl);
                goto out;
            }
                
            if(!m_regressors) {
                (cerr << "no regressors for training" << endl);
                goto out;
            }
                        
           try {
                CheckSameSizes(*m_responses, *m_regressors, "linear regression", "responses");
            } catch (std::invalid_argument& s) {
                (cerr << s.what() << endl);
                goto out;
            }
            
            if(m_responses->n_elem != m_regressors->n_cols) {
                (cerr << "mismatch between number of responses (" << m_responses->n_elem << ") and regressors (" << m_regressors->n_cols << ")." << endl);
                goto out;
            }
    
            m_model.model = std::make_unique<LinearRegression>(std::move(*m_regressors), std::move(*m_responses), lambda);
            
 
            //ComputeError
            atom_setfloat(a,m_model.model->ComputeError(*m_regressors, *m_responses));
            outlet_anything(m_dumpoutlet, gensym("error"), 1, a);
        out:
            return {};
        },
    };
    
    message<> getparameters {this, "getparameters", "Outputs the parameters (the b vector) via dump outlet.",
        MIN_FUNCTION {
            if(!m_model.model) {
                (cerr << "No model has been trained." << endl);
            } else {
               try {
                   
                   const arma::vec params = m_model.model->Parameters();

                   t_atom a[params.n_elem];
                   for(auto i=0;i<params.n_elem;i++) {
                       atom_setfloat(a+i,params(i));
                   }
                   outlet_anything(m_dumpoutlet, gensym("parameters"), params.n_elem, a);
               } catch (const std::runtime_error& s) {
                   (cerr << s.what() << endl);
               }
                
            }
           return {};
        }
        
    };
    
    
    message<> clear { this, "clear", "clear model.",
        MIN_FUNCTION {
            m_model.model.reset();
            m_responses.reset();
            m_regressors.reset();
            return {};
        }
        
    };

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

        
            // add mop
            t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 3, 1));

            // force type
            jit_mop_single_type(mop, _jit_sym_float64);
        
           
            auto in2 = object_method(mop,_jit_sym_getinput,2);
            auto in3 = object_method(mop,_jit_sym_getinput,3);
            
            jit_attr_setlong(in2,_jit_sym_dimlink,0);
            jit_attr_setlong(in2,_jit_sym_typelink,0);
            jit_attr_setlong(in3,_jit_sym_dimlink,0);
            jit_attr_setlong(in3,_jit_sym_typelink,0);
            
            //always adapt
            object_method(in2,gensym("ioproc"),jit_mop_ioproc_copy_adapt);
            object_method(in3,gensym("ioproc"),jit_mop_ioproc_copy_adapt);
            
            jit_class_addadornment(c, mop);
            
            // add our custom matrix_calc method
            jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
            return {};
        }
    };
                

    t_jit_err set_responses(t_object *matrix) {
        t_jit_err err = JIT_ERR_NONE;
        m_responses = std::make_unique<arma::Row<double>>();
        *m_responses = jit_to_arma(mode, matrix, *m_responses);
        return err;
    }
    
    t_jit_err set_regressors(t_object *matrix) {
        t_jit_err err = JIT_ERR_NONE;
        m_regressors = std::make_unique<arma::Mat<double>>();
        *m_regressors = jit_to_arma(mode, matrix, *m_regressors);
        m_mode_changed = false;
        return err;
    }
                
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_matrix_info, out_predictions_info;
        arma::mat query;
        arma::Row<double> predictions;

        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto out_predictions = object_method(outputs, _jit_sym_getindex, 0);
        auto in_matrix_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_predictions_savelock = object_method(out_predictions, _jit_sym_lock, 1);


        object_method(in_matrix, _jit_sym_getinfo, &in_matrix_info);
        
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_matrix_info);

        

        if(!m_model.model) {
            (cerr << "no model trained." << endl);
            goto out;
        }
        
        try {
            check_mode(in_matrix_info, mode, "hoeffding tree");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        query = jit_to_arma(mode, in_matrix64, query);

        if(m_mode_changed) {
            cerr << "mode has changed must resubmit reference set" << endl;
            goto out;
        }
        
        if(!m_model.model) {
            (cerr << "no linear regression model has been trained" << endl);
            goto out;
        }

        
        try {
            CheckSameDimensionality(query, *m_regressors, "linear regression", "query");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }


        try {
            m_model.model->Predict(query, predictions);
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        out_predictions_info = in_matrix_info;

        out_predictions_info.planecount = 1;//in_data_info.dimcount;
        out_predictions_info.type = _jit_sym_float64;

        if(mode == 0) {
            if(out_predictions_info.dimcount == 1) {
                out_predictions_info.dim[1] = 1;//for loops
            }
        }

        out_predictions = arma_to_jit(mode, predictions, static_cast<t_object*>(out_predictions), out_predictions_info);

    out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        object_method(in_matrix,_jit_sym_lock,in_matrix_savelock);
        object_method(out_predictions,_jit_sym_lock,out_predictions_savelock);
        return err;
    }
                
private:
    std::unique_ptr<arma::Mat<double>> m_regressors { nullptr };
    std::unique_ptr<arma::Row<double>> m_responses { nullptr };
};


MIN_EXTERNAL(mlmat_linear_regression);


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
        
        minwrap<mlmat_linear_regression>* job = (minwrap<mlmat_linear_regression>*)(j);
        err = job->m_min_object.set_regressors((t_object*)m);

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
        
        minwrap<mlmat_linear_regression>* job = (minwrap<mlmat_linear_regression>*)(j);
        err = job->m_min_object.set_responses((t_object*)m);

        if(err)
            jit_error_code(x,err);
    }
}
                
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_linear_regression>* job = (minwrap<mlmat_linear_regression>*)(x);
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
                    sprintf(s, "(matrix) predictors");
                    break;

                case 2:
                    sprintf(s, "(matrix)) responses");
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

                default:
                    sprintf(s, "dumpout");
                    break;
            }
           default:
            break;
    }
}
