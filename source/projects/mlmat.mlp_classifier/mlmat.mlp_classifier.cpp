/// @file mlmat.mlp_classifier.cpp
/// @ingroup mlmat
// @copyright Copyright 2021 Todd Ingalls. All rights reserved. Also based on examples provided with the mlpack library. Please see source/mlpack for license details
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// TODO: Validate optimizer settings. Don't seem to be working well. Could be because not scaling?


#include "mlmat.hpp"
#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>
#include <string>


using namespace c74::min;
using namespace c74::max;
using namespace mlpack;

t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) ;
void mlmat_mlp_assist(void* x, void* b, long io, long index, char* s);
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);

class mlmat_mlp_classifier : public mlmat_object_writable<mlmat_mlp_classifier, FFN<>> {
public:
    MIN_DESCRIPTION     {"Multi layer perceptron. This mlp can be used as a classifier."};
    MIN_TAGS            {"ML"};
    MIN_AUTHOR          {"Todd Ingalls"};
    MIN_RELATED         {"mlmat.svm"};
    
    attribute<int> output_neurons { this, "output_neurons", 1,
        description {
            "Number of output neurons."
        }
    };
    
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
        range { "sigmoid", "gaussian",  "relu", "tanh", "soft_plus",  "identity" },
        description {
        	"The activation function to use in the hidden layers."
        }
    };
    
    attribute<c74::min::symbol> optimizer { this, "optimizer", "rmsprop",
        range { "rmsprop", "sgd", "lbfgs", "adam"},
        description {
        	"The optimizer to use."
        }
        
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
    
    message<> train { this, "train", "train model.",
        MIN_FUNCTION {
            if(!m_labels) {
                (cerr << "unable to run training. no valid labels." << endl);
                return {};
            }
            if(!m_training) {
                (cerr << "unable to run training. no valid training data." << endl);
                return {};
            }
            
            
            arma::Mat<double> labels(*m_labels);
            arma::mat out_data;
            
            
            
            m_model.model = std::make_unique<FFN<>>();
                
            m_model.model->Add<Linear>(m_training->n_rows);
            
            for(auto i = 0;i<hidden_layers;i++) {
                add_layer(activation.get());
                m_model.model->Add<Linear>(hidden_neurons.get());
                //(std::cout << "hidden layer: " << hidden_neurons.get() << " inputs " << hidden_neurons.get() << " outputs" << std::endl);
            }
                
            m_model.model->Add<LinearType<arma::mat, NoRegularizer>>(hidden_neurons.get());
            m_model.model->Add<LogSoftMaxType<>>();
            //removing will test when labels inpput
            //double diff = m_labels_min - 1.0;
            //labels = labels - diff;
            
            scaler_fit(m_model, *m_training);
            out_data = scaler_transform(m_model, *m_training, out_data);
            
            
            //std::cout << labels << std::endl;
            
            if(optimizer.get() == "rmsprop") {
                // this is default
                m_model.model->Train(out_data, labels);
            } else if(optimizer.get() == "sgd") {
                ens::StandardSGD opt(step_size, batch_size, max_iterations, tolerance);
                m_model.model->Train(out_data, labels, opt);
            } else if(optimizer.get() == "lbfgs") {
                ens::L_BFGS opt;
                opt.MaxIterations() = max_iterations;
                opt.MinGradientNorm() = tolerance;
                m_model.model->Train(out_data, labels, opt);
            } else if(optimizer.get() == "adam") {
                ens::Adam opt(step_size, batch_size, 0.9, 0.999, 1e-8, max_iterations,
                tolerance);
                m_model.model->Train(out_data, labels, opt);
            } else {
                ///ERROR?
            }
            return {};
        }
    };
        
        
    void add_layer(const c74::min::symbol& layer_type) {
        const string layer_string = layer_type.c_str();
        
        if(layer_string == "sigmoid") {
            m_model.model->Add<mlpack::SigmoidType<>>();

        } else if(layer_string == "gaussian") {
            m_model.model->Add<GaussianType<>>();

        } else if(layer_string == "relu") {
            m_model.model->Add<ReLUType<>>();

        } else if(layer_string == "tanh") {
            m_model.model->Add<TanHType<>>();

        } else if(layer_string == "soft_plus") {
            m_model.model->Add<SoftPlusType<>>();

        } else if(layer_string == "linear") {
            m_model.model->Add<LinearType<arma::mat, NoRegularizer>>();
        } else if(layer_string == "identity") {
            m_model.model->Add<IdentityType<>>();
        }

    }
    
    message<> clear { this, "clear", "clear data and model",
        MIN_FUNCTION {
            m_labels.reset();
            m_training.reset();
            m_model.model.reset();
            
            return {};
        }
    };
    
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        // ignore last two inputs as they have already been processed
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_matrix_info, out_info, out_predictions_info;
        arma::mat query;
        arma::mat likelihoods;
        arma::Row<double> predictions;
        arma::Row<size_t> labels;

        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto out_predictions_matrix = object_method(outputs, _jit_sym_getindex, 0);
        auto out_likelihoods_matrix = object_method(outputs, _jit_sym_getindex, 1);
        auto in_matrix_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_predictions_savelock = object_method(out_predictions_matrix, _jit_sym_lock, 1);
        auto out_likelihoods_savelock = object_method(out_likelihoods_matrix, _jit_sym_lock, 1);
        object_method(in_matrix, _jit_sym_getinfo, &in_matrix_info);
        
        
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_matrix_info);
                
        if(!m_model.model ) {
            (cerr << "no mlp model has been trained" << endl);
            goto out;
        }
        
        try {
            check_mode(in_matrix_info, mode, "mlp classifier");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
            

        query = jit_to_arma(mode, static_cast<t_object*>(in_matrix64), query);
        
        try {
            size_t p = m_model.model->InputDimensions()[0];
            mlpack::util::CheckSameDimensionality(query, p, "mlp classifier", "query");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        try {
            arma::mat scaled_query;
            scaled_query = scaler_transform(m_model, query, scaled_query);
            m_model.model->Predict(std::move(scaled_query), likelihoods);
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        
        predictions.zeros(1, likelihoods.n_cols);
        
        for(auto i=0;i<likelihoods.n_cols;i++) {
            predictions(i) = arma::as_scalar(arma::find(arma::max(likelihoods.col(i)) == likelihoods.col(i), 1));
            
            
            //predictions(i) = arma::index_max(likelihoods.col(i));
        }
        
        //labels = arma::conv_to<arma::Row<size_t>>::from(predictions) - (size_t)m_labels_min;
        labels = arma::conv_to<arma::Row<size_t>>::from(predictions);
        out_info = in_matrix_info;
        out_info.planecount = 1;
        
        if(mode == 0) {
            out_info.dimcount = in_matrix_info.dimcount + (likelihoods.n_rows > 1 ? 1 : 0);
            
            switch(out_info.dimcount) {
                case 1: //seems useless but could happen
                    out_info.dim[0] = in_matrix_info.dim[0];
                    out_info.dim[1] = 1;//for loops
                    break;
                case 2:
                    out_info.dim[0] = in_matrix_info.dim[0];
                    out_info.dim[1] = likelihoods.n_rows;

                    break;
                case 3:
                    out_info.dim[0] = in_matrix_info.dim[0];
                    out_info.dim[1] = in_matrix_info.dim[1];
                    out_info.dim[2] = likelihoods.n_rows;;
                    break;
                default:
                    break;
            }
        }
        
        out_predictions_info = in_matrix_info;
        out_predictions_info.type = _jit_sym_long;
        out_predictions_info.planecount = 1;
        
        out_predictions_matrix = arma_to_jit(mode, labels, static_cast<t_object*>(out_predictions_matrix), out_predictions_info);
        out_likelihoods_matrix = arma_to_jit(mode, likelihoods, static_cast<t_object*>(out_likelihoods_matrix), out_info);

   out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        object_method(in_matrix,_jit_sym_lock,in_matrix_savelock);
        object_method(out_predictions_matrix,_jit_sym_lock,out_predictions_savelock);
        object_method(out_likelihoods_matrix,_jit_sym_lock,out_likelihoods_savelock);

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
            check_mode(minfo, mode, "mlp classifier");
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
        
    t_jit_err process_labels_matrix(t_object *matrix) {
        t_jit_matrix_info minfo;
        t_jit_err err = JIT_ERR_NONE;
        arma::mat dat;
        double labels_min = 0.;
        double labels_max = 0.;
        arma::Col<double> flat;
        
        long savelock = (long) object_method((t_object*)matrix, _jit_sym_lock, 1);
        object_method((t_object*)matrix, _jit_sym_getinfo, &minfo);
        
        if(minfo.dimcount > 2) {
            (cout << "expecting 1d or 2d matrix, received " << minfo.dimcount << "d matrix" << endl);
            err = JIT_ERR_INVALID_INPUT;
            goto out;
        }
        
        if(minfo.planecount > 1) {
            (cerr << "expected a single plane matrix for labels but received a matrix with planecount of " << minfo.planecount << endl);
            
            err = JIT_ERR_INVALID_INPUT;
            goto out;
        }
        
        dat = jit_to_arma(mode, matrix, dat);
        
        m_labels = std::make_unique<arma::Mat<double>>(dat);

        labels_min = m_labels->min();
        labels_max = m_labels->max();

        if(labels_min != 0.0) {
            (cerr << "labels must start at 0" << endl);
            m_labels.reset(nullptr);
            goto out;
        }
        
        flat = arma::vectorise(*m_labels);
        
        for(auto i=0;i<labels_max;i++) {
            if(arma::any(flat == i )) {
                continue;
            } else {
                (cerr << "labels must be continuous from 0 to " << labels_max << "." << endl);
                m_labels.reset(nullptr);
                goto out;
            }
            
        }
        
        
    out:
        object_method((t_object*)matrix, _jit_sym_lock, savelock);
        
        return err;
    }
    

private:
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
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
        class_addmethod(c, (method)mlmat_mlp_assist, "assist", A_CANT, 0);
        return {};

    }};
    
    std::unique_ptr<arma::Mat<double>> m_training;
    std::unique_ptr<arma::Mat<double>> m_labels;
//    double m_labels_min = 0.;
//    double m_labels_max = 0.;
    
};

MIN_EXTERNAL(mlmat_mlp_classifier);



void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc, t_atom *argv) {
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
        
        minwrap<mlmat_mlp_classifier>* job = (minwrap<mlmat_mlp_classifier>*)(j);
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
        minwrap<mlmat_mlp_classifier>* job = (minwrap<mlmat_mlp_classifier>*)(j);
        err = job->m_min_object.process_labels_matrix((t_object*)m);
        
        jit_error_code(x,err);
    }
    
}


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_mlp_classifier>* job = (minwrap<mlmat_mlp_classifier>*)(x);
        // call our custom matrix_calc function defined inside the C++ class
        err = job->m_min_object.matrix_calc(x, inputs, outputs);
    }

    return err;
}

void mlmat_mlp_assist(void* x, void* b, long io, long index, char* s) {
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
                    sprintf(s, "(matrix) likelihoods");
                    break;

                default:
                    sprintf(s, "dumpout");
                    break;
            }
           default:
            break;
    }
}

