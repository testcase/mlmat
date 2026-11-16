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

    attribute<bool> background_thread { this, "background_thread", true,
        description {
            "If set (the default), training runs on a background thread: the train message returns immediately, the object stays responsive, and predictions keep using the previous model until the new one is ready. A done message is sent out the dumpout outlet when training finishes (error, if it failed). If 0, training blocks the main thread as before."
        }
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
            if(!m_training || !m_target) {
                (cerr << "no training data or target received." << endl);
                return {};
            }

            if (background_in_progress()) {
                return {};
            }

            if (background_thread) {
                BgParams p;
                p.training = *m_training;
                p.target = *m_target;
                p.hidden_layers = hidden_layers;
                p.hidden_neurons = hidden_neurons;
                p.activation = activation.get().c_str();
                p.optimizer = optimizer.get().c_str();
                p.step_size = step_size;
                p.batch_size = batch_size;
                p.max_iterations = max_iterations;
                p.tolerance = tolerance;
                p.autoscale = autoscale;
                p.scaler_type = scaler.get().c_str();
                p.scaler_min = scaler_min;
                p.scaler_max = scaler_max;
                p.scaler_epsilon = scaler_epsilon;

                background_launch([p = std::move(p)]() mutable -> BgResult {
                    BgResult r;
                    r.model = std::make_unique<FFN<MeanSquaredError,RandomInitialization>>();
                    r.model->Add<Linear>(p.training.n_rows);
                    for (int i = 0; i < p.hidden_layers; ++i) {
                        add_layer(*r.model, p.activation);
                        r.model->Add<Linear>(p.hidden_neurons);
                    }
                    r.model->Add<Identity>();
                    r.model->Add<Linear>(p.target.n_rows);
                    r.model->Add<Identity>();

                    arma::mat out_data = p.training;
                    if (p.autoscale) {
                        r.scaler = mlmat::make_scaler(p.scaler_type, p.scaler_min, p.scaler_max, p.scaler_epsilon);
                        r.scaler->Fit(p.training);
                        r.scaler->Transform(p.training, out_data);
                    }

                    if (p.optimizer == "rmsprop") {
                        r.model->Train(out_data, p.target);
                    } else if (p.optimizer == "sgd") {
                        ens::StandardSGD opt(p.step_size, p.batch_size, p.max_iterations, p.tolerance);
                        r.model->Train(out_data, p.target, opt);
                    } else if (p.optimizer == "lbfgs") {
                        ens::L_BFGS opt;
                        opt.MaxIterations() = p.max_iterations;
                        opt.MinGradientNorm() = p.tolerance;
                        r.model->Train(out_data, p.target, opt);
                    } else if (p.optimizer == "adam") {
                        ens::Adam opt(p.step_size, p.batch_size, 0.9, 0.999, 1e-8, p.max_iterations, p.tolerance);
                        r.model->Train(out_data, p.target, opt);
                    }
                    return r;
                });
                return {};
            }

            // range for random initialization
            m_model.model = std::make_unique<FFN<MeanSquaredError,RandomInitialization>>();
            arma::mat out_data;
        
            m_model.model->Add<Linear>(m_training->n_rows);
            
            // normalized with mlmat.mlp_classifier: hidden_layers is the
            // number of hidden layers (previously this loop ran one fewer)
            for(auto i = 0;i<hidden_layers;i++) {
                add_layer(*m_model.model, activation.get().c_str());
                m_model.model->Add<Linear>(hidden_neurons.get());
            }
            m_model.model->Add<Identity>();
            m_model.model->Add<Linear>( m_target->n_rows);
            m_model.model->Add<Identity>();
            
            //m_model.model->InputDimensions() = {m_training->n_rows};
            
            //std::cout << m_model.model->InputDimensions() << std::endl;
            
            scaler_fit(m_model, *m_training);
            out_data = scaler_transform(m_model, *m_training, out_data);
            
//            auto ls =m_model.model->Network();
//            
//            for(auto i=0;i<ls.size();i++) {
//                std::cout << ls[i]->InputDimensions() << std::endl;
//            }
//            
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


    static void add_layer(FFN<MeanSquaredError,RandomInitialization>& model, const std::string& layer_string) {
        if(layer_string == "sigmoid") {
            model.Add<Sigmoid<>>();

        } else if(layer_string == "gaussian") {
            model.Add<Gaussian<>>();

        } else if(layer_string == "relu") {
            model.Add<ReLU<>>();

        } else if(layer_string == "tanh") {
            model.Add<TanH<>>();

        } else if(layer_string == "soft_plus") {
            model.Add<SoftPlus<>>();

        } else if(layer_string == "identity") {
            model.Add<Identity<>>();
        }

    }

    // background training
    struct BgResult {
        std::unique_ptr<FFN<MeanSquaredError,RandomInitialization>> model;
        std::unique_ptr<mlpack::ScalingModel> scaler;
    };

    struct BgParams {
        arma::Mat<double> training;
        arma::Mat<double> target;
        int hidden_layers;
        int hidden_neurons;
        std::string activation;
        std::string optimizer;
        double step_size;
        double batch_size;
        int max_iterations;
        double tolerance;
        bool autoscale;
        std::string scaler_type;
        double scaler_min;
        double scaler_max;
        double scaler_epsilon;
    };

    void bg_commit(BgResult&& r) {
        m_model.model = std::move(r.model);
        m_model.scaler = std::move(r.scaler);
    }
        
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        // ignore last two inputs as they have already been processed
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_matrix_info, out_info;


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
            mlpack::util::CheckSameDimensionality(query, m_model.model->InputDimensions()[0], "mlp regressor", "query");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        try {
            scaled_query = scaler_transform(m_model, query, scaled_query);
            m_model.model->Predict(std::move(scaled_query), predictions);
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }

        out_info = in_matrix_info;
        out_info.planecount = predictions.n_rows;
        
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
        
    // scratch buffers reused across calls
    arma::mat query;
    arma::mat scaled_query;
    arma::mat predictions;
    
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
                    snprintf(s, 64, "(matrix) testing points.");
                    break;

                case 1:
                    snprintf(s, 64, "(matrix) training dataset.");
                    break;

                case 2:
                    snprintf(s, 64, "(matrix) training targets.");
                    break;

                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    snprintf(s, 64, "(matrix) predictions");
                    break;

                default:
                    snprintf(s, 64, "dumpout");
                    break;
            }
           default:
            break;
    }
}
