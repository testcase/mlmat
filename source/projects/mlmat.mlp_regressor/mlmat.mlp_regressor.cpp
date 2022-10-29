/// @file mlmat.mlp_regressor.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved. Also based on examples provided with the mlpack library. Please see source/mlpack for license details
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// TODO: Validate optimizer settings.



#include "mlmat.hpp"
#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>
#include <string>


using namespace c74::min;
using namespace c74::max;
using namespace mlpack;


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) ;
void mlmat_assist(void* x, void* b, long io, long index, char* s);
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);


class mlmat_mlp_regressor : public mlmat_object_writable<mlmat_mlp_regressor, FFN<MeanSquaredError, RandomInitialization>> {
public:
    MIN_DESCRIPTION     {"Multi layer perceptron. This mlp can be used as a regressor."};
    MIN_TAGS            {"ML"};
    MIN_AUTHOR          {"Todd Ingalls"};
    MIN_RELATED          {"mlmat.linear_regression"};

    attribute<int> hidden_layers { this, "hidden_layers", 1,
        description {
            "Number of hidden layers."
        }
    };
    
    attribute<int> hidden_neurons { this, "hidden_neurons", 1,
        description {
            "Number of neurons for each hidden layer."
        }
    };
    
    attribute<c74::min::symbol> activation { this, "activation", "relu",
        range { "sigmoid", "gaussian",  "relu", "tanh", "soft_plus",  "identity" }
    };
    
    //will add this functionality later
    attribute<c74::min::symbol> optimizer { this, "optimizer", "rmsprop",
        range { "rmsprop", "sgd", "lbfgs", "adam"}
    };
    
    attribute<double> step_size { this, "step_size", 0.01,
        description { "Step size for parallel SGD optimizer." }
    };
    
    attribute<double> batch_size { this, "batch_size", 50,
        description { "Batch size for mini-batch SGD." }
    };
    
    
    attribute<int> max_iterations { this, "max_iterations", 10000,
          description { "Maximum iterations for optimizer (0 indicates no limit)." }
    };
    
    attribute<double> tolerance { this, "tolerance", 1e-7,
        description { "Convergence tolerance for optimizer." }
    };

    message<> clear { this, "clear", "clear data and model",
        MIN_FUNCTION {
            m_target.reset();
            m_training.reset();
            m_model.model.reset();
            return {};
        }
    };
    

    message<> train { this, "train", "train model.",
        MIN_FUNCTION {
            // range for random initialization
            m_model.model = std::make_unique<FFN<MeanSquaredError,RandomInitialization>>();
            arma::mat out_data;
        
            m_model.model->Add<Linear>(hidden_neurons.get());
            
            for(auto i = 0;i<hidden_layers-1;i++) {
                add_layer(activation.get());
                m_model.model->Add<Linear>(m_training->n_rows);
            }

            m_model.model->Add<Linear>( hidden_neurons.get());
            m_model.model->Add<Identity>();
            
            scaler_fit(m_model, *m_training);
            out_data = scaler_transform(m_model, *m_training, out_data);
            
            if(optimizer.get() == "rmsprop") {
                // this is default
                try {
                    m_model.model->Train(out_data, *m_target);
                } catch (std::exception& s)  {
                    cerr << s.what() << endl;
                    goto out;
                }
            } else if(optimizer.get() == "sgd") {
                ens::StandardSGD opt(step_size, batch_size, max_iterations, tolerance);
                try {
                    m_model.model->Train(out_data, *m_target, opt);
                } catch (std::exception& s)  {
                    cerr << s.what() << endl;
                    goto out;
                }
            } else if(optimizer.get() == "lbfgs") {
                ens::L_BFGS opt;
                opt.MaxIterations() = max_iterations;
                opt.MinGradientNorm() = tolerance;
                try {
                    m_model.model->Train(out_data, *m_target, opt);
                } catch (std::exception& s)  {
                    cerr << s.what() << endl;
                    goto out;
                }
            } else if(optimizer.get() == "adam") {
                ens::Adam opt(step_size, batch_size, 0.9, 0.999, 1e-8, max_iterations,
                tolerance);
                try {
                    m_model.model->Train(out_data, *m_target, opt);
                } catch (std::exception& s)  {
                    cerr << s.what() << endl;
                    goto out;
                }
            } else {
                ///ERROR?
            }
        out:
            return {};
        }
    };


    void add_layer(const c74::min::symbol& layer_type) {
        const string layer_string = layer_type.c_str();
        
        if(layer_string == "sigmoid") {
            m_model.model->Add<SigmoidType<>>();

        } else if(layer_string == "gaussian") {
            m_model.model->Add<GaussianType<>>();

        } else if(layer_string == "relu") {
            m_model.model->Add<ReLUType<>>();

        } else if(layer_string == "tanh") {
            m_model.model->Add<TanHType<>>();

        } else if(layer_string == "soft_plus") {
            m_model.model->Add<SoftPlusType<>>();

        } else if(layer_string == "identity") {
            m_model.model->Add<IdentityType<>>();
        }

    }
        
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        // ignore last two inputs as they have already been processed
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_matrix_info, out_info;
        arma::mat query;
        arma::mat predictions;

        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto out_results_matrix = object_method(outputs, _jit_sym_getindex, 0);
        auto in_matrix_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_results_savelock = object_method(out_results_matrix, _jit_sym_lock, 1);

        object_method(in_matrix, _jit_sym_getinfo, &in_matrix_info);
        
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_matrix_info);

            
        try {
            check_mode(in_matrix_info, mode, "mlp regressor");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        
        if(!m_model.model ) {
            (cerr << "no mlp model has been trained" << endl);
            goto out;
        }
        

        query = jit_to_arma(mode, static_cast<t_object*>(in_matrix64), query);
        
        try {
            size_t p = m_model.model->InputDimensions()[0];
            cout << p << endl;
            mlpack::util::CheckSameDimensionality(query, p, "mlp regressor", "query");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        if(query.n_rows != m_training->n_rows) {
            (cerr << "mismatched "<< endl);
                goto out;
        }
        
        try {
            arma::mat scaled_query;
            scaled_query = scaler_transform(m_model, query, scaled_query);
            m_model.model->Predict(std::move(scaled_query), predictions);
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }

        out_info = in_matrix_info;
        out_info.planecount = m_target->n_rows;
        
        out_results_matrix = arma_to_jit(mode, predictions, static_cast<t_object*>(out_results_matrix), out_info);
        
   out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        object_method(in_matrix,_jit_sym_lock,in_matrix_savelock);
        object_method(out_results_matrix,_jit_sym_lock,out_results_savelock);

        return err;
    }
        
    t_jit_err process_training_matrix(t_object *matrix) {
        t_jit_matrix_info minfo;
        t_jit_err err = JIT_ERR_NONE;
        arma::mat dat;
        
        long savelock = (long) object_method((t_object*)matrix, _jit_sym_lock, 1);
        object_method((t_object*)matrix, _jit_sym_getinfo, &minfo);
        
        if(minfo.dimcount > 2) {
            (cout << "expecting 1d or 2d matrix, received " << minfo.dimcount << "d matrix" << endl);
            err = JIT_ERR_INVALID_INPUT;
            goto out;
        }
        
        try {
            check_mode(minfo, mode, "mlp regressor");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        dat = jit_to_arma(mode, matrix, dat);
        
        m_training = std::make_unique<arma::Mat<double>>(dat);
        
    out:
        object_method((t_object*)matrix, _jit_sym_lock, (void*)savelock);
        return err;
    }
        
    t_jit_err process_target_matrix(t_object *matrix) {
        t_jit_matrix_info minfo;
        t_jit_err err = JIT_ERR_NONE;
        arma::mat dat;
        
        long savelock = (long) object_method((t_object*)matrix, _jit_sym_lock, 1);
        object_method((t_object*)matrix, _jit_sym_getinfo, &minfo);
        
        if(minfo.dimcount > 2) {
            (cout << "expecting 1d or 2d matrix, received " << minfo.dimcount << "d matrix" << endl);
            err = JIT_ERR_INVALID_INPUT;
            goto out;
        }
        
        try {
            check_mode(minfo, mode, "mlp regressor");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        dat = jit_to_arma(mode, matrix, dat);
        
        m_target = std::make_unique<arma::Mat<double>>(dat);

    out:
        object_method((t_object*)matrix, _jit_sym_lock, (void*)savelock);
        
        return err;
    }
        
private:
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        // add mop
        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 3, 1));
        
        // force type
        jit_mop_single_type(mop, _jit_sym_float64);
        
        void* in2 = object_method(mop,_jit_sym_getinput,2);
        void* in3 = object_method(mop,_jit_sym_getinput,3);
        
        jit_attr_setlong(in2,_jit_sym_dimlink,0);
        jit_attr_setlong(in2,_jit_sym_typelink,0);
        jit_attr_setlong(in3,_jit_sym_dimlink,0);
        jit_attr_setlong(in3,_jit_sym_typelink,0);
        
        //always adapt
        object_method(in2,gensym("ioproc"),jit_mop_ioproc_copy_adapt);
        object_method(in3,gensym("ioproc"),jit_mop_ioproc_copy_adapt);
        
        jit_class_addadornment(c, mop);
        
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
    
    std::unique_ptr<arma::Mat<double>> m_training;
    std::unique_ptr<arma::Mat<double>> m_target;
};

MIN_EXTERNAL(mlmat_mlp_regressor);



void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv) {
    //found need to call this first or the
    // info for the incoming matrix is incorrect
    // this does not seem right
    max_jit_mop_jit_matrix(x,s,argc,argv);
    //only want second inlet
    if (max_jit_obex_inletnumber_get(x) == 1)
    {
        t_jit_err err = JIT_ERR_NONE;
        void *j = nullptr;
        
         void *p, *m, *mop;
        t_jit_matrix_info minfo;
        
        if (!(mop=max_jit_obex_adornment_get(x,_jit_sym_jit_mop)))
            jit_error_code(x,err);
        
        p = object_method((t_object*)mop,_jit_sym_getinput,2);
        m = object_method((t_object*)p,_jit_sym_getmatrix);
        j = max_jit_obex_jitob_get(x);
        object_method((t_object*)m, _jit_sym_getinfo, &minfo);
        
        minwrap<mlmat_mlp_regressor>* job = (minwrap<mlmat_mlp_regressor>*)(j);
        err = job->m_min_object.process_training_matrix((t_object*)m);
        
        jit_error_code(x,err);
    } else if (max_jit_obex_inletnumber_get(x) == 2)
    {
        t_jit_err err = JIT_ERR_NONE;
        void *j = nullptr;
        
         void *p, *m, *mop;
        t_jit_matrix_info minfo;
        
        if (!(mop=max_jit_obex_adornment_get(x,_jit_sym_jit_mop)))
            jit_error_code(x,err);
        
        p = object_method((t_object*)mop,_jit_sym_getinput,3);
        m = object_method((t_object*)p,_jit_sym_getmatrix);
        j = max_jit_obex_jitob_get(x);
        object_method((t_object*)m, _jit_sym_getinfo, &minfo);
        minwrap<mlmat_mlp_regressor>* job = (minwrap<mlmat_mlp_regressor>*)(j);
        err = job->m_min_object.process_target_matrix((t_object*)m);
        
        jit_error_code(x,err);
    }
    
}


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_mlp_regressor>* job = (minwrap<mlmat_mlp_regressor>*)(x);
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
                    sprintf(s, "(matrix) training targets.");
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
