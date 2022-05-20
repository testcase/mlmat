/// @file mlmat.variational_autoencoder.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// TODO: Need to make autoencode serializable
///
///   based on code from https://github.com/mlpack/models/blob/master/vae/
///   @author Atharva Khandait


#include "mlmat.hpp"
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/reconstruction_loss.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/dists/bernoulli_distribution.hpp>
#include <ensmallen.hpp>


using namespace c74::min;
using namespace c74::max;

using c74::max::t_linklist;

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::util;
using namespace ens;

// Calculates mean loss over batches.
template<typename NetworkType = FFN<MeanSquaredError<>, HeInitialization>,
         typename DataType = arma::mat>
double MeanTestLoss(NetworkType& model, DataType& testSet, size_t batchSize)
{
  double loss = 0;
  size_t nofPoints = testSet.n_cols;
  size_t i;

  for (i = 0; i < (size_t) nofPoints / batchSize; ++i)
  {
    loss +=
        model.Evaluate(testSet.cols(batchSize * i, batchSize * (i + 1) - 1),
                       testSet.cols(batchSize * i, batchSize * (i + 1) - 1));
  }

  if (nofPoints % batchSize != 0)
  {
    loss += model.Evaluate(testSet.cols(batchSize * i, nofPoints - 1),
                           testSet.cols(batchSize * i, nofPoints - 1));
    loss /= (int)nofPoints / batchSize + 1;
  }
  else
    loss /= nofPoints / batchSize;

  return loss;
}



typedef FFN<ReconstructionLoss<arma::mat,
                                arma::mat,
                                BernoulliDistribution<arma::mat> >,
                                HeInitialization> ReconModel;

typedef FFN<MeanSquaredError<>, HeInitialization> MeanSModel;



t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) ;
void mlmat_assist(void* x, void* b, long io, long index, char* s);
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);
void max_jit_mlmat_mproc(max_jit_wrapper *x, void *mop);

class mlmat_variational_autoencoder : public mlmat_object_writable<mlmat_variational_autoencoder, ReconModel> {
public:
    MIN_DESCRIPTION	{"variational autoencoder."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.mlp_regressor"};
    
    using ivec = vector<int>;
    attribute<ivec> hidden_sizes { this, "hidden_sizes", {0}, description {"Hidden layer sizes."} };
    
    attribute<int> latent_size { this, "latent_size", 0};
    
    attribute<int> batch_size { this, "batch_size", 100};

    attribute<double> step_size { this, "step_size", 0.001};
    
    attribute<double> train_ratio { this, "train_ratio", 0.8};
    
    attribute<double> test_ratio { this, "test_ratio", 0.045};
    
    attribute<int> iters_per_cycle { this, "iters_per_cycle", 56000};
    
    
    attribute<int> cycles { this, "cycles", 100};
    
    attribute<double> lambda { this, "lambda", .0001,
        description {
            "L2-regularization parameter."
        }
    };
    
    attribute<bool> autoclear {this, "autoclear", true,
        description {
            "Clear training data after training is run."
        }
    };
    
    attribute<c74::min::symbol> activation { this, "activation", "relu",
        range { "sigmoid", "gaussian",  "relu", "tanh", "soft_plus",  "identity" }
    };
    
    attribute<c74::min::symbol> file {this, "file", k_sym__empty,
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
    
    message<> write {this, "write",
        MIN_FUNCTION {
           try {
               m_model.autoscale = autoscale;
               save_model_file(args, m_model, "variational_autoencoder");
           } catch (const std::runtime_error& s) {
               (cout << s.what() << endl);
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
               mlpack::data::Load(string(p), "variational_autoencoder", m_model, true);
           } catch (const std::runtime_error& s) {
               std::throw_with_nested(std::runtime_error("Error reading model file to disk."));
           }
       }
    }
    
    message<> train { this, "train", "train model.",
        MIN_FUNCTION {
            if(m_data) {
                m_model.model = std::make_unique<ReconModel>();
                //arma::Mat<double> full_data;
                arma::mat scaled_data;
                
                const ivec& hidden_sizes = this->hidden_sizes;
                const size_t latent_size = this->latent_size;
                
                
                scaler_fit(m_model, *m_data);
                scaled_data = scaler_transform(m_model, *m_data, scaled_data);
               // std::cout << scaled_data << std::endl;
            
                arma::mat train, validation;
                data::Split(scaled_data, validation, train, train_ratio);

                // Loss is calculated on train_test data after each cycle.
                arma::mat train_test, dump;
                data::Split(train, dump, train_test, test_ratio);
            
                //need an identity layer to use Sequential object
                m_model.model->Add<IdentityLayer<> >();
                
                Sequential<>* encoder = new Sequential<>();

                encoder->Add<Linear<> >(train.n_rows, hidden_sizes[0]);
                add_layer(encoder, activation.get());

                
                for(auto i=0;i<hidden_sizes.size()-1;i++) {
                    encoder->Add<Linear<> >(hidden_sizes[i], hidden_sizes[i+1]);
                    add_layer(encoder, activation.get());
                }
                
                encoder->Add<Linear<> >(hidden_sizes.back(), 2 * latent_size);
                m_model.model->Add(encoder);

                // Reparametrization layer.
                m_model.model->Add<Reparametrization<> >(latent_size);

                // Decoder.
                Sequential<>* decoder = new Sequential<>();

                decoder->Add<Linear<> >(latent_size, hidden_sizes.back());
                add_layer(decoder, activation.get());
                
                for(auto i=hidden_sizes.size()-1;i>1;i--) {
                    decoder->Add<Linear<> >(hidden_sizes[i], hidden_sizes[i-1]);
                    add_layer(decoder, activation.get());
                }
            
                decoder->Add<Linear<> >(hidden_sizes[1], hidden_sizes[0]);
                m_model.model->Add(decoder);
            
                
                // Setting parameters for the Stochastic Gradient Descent (SGD) optimizer.
                SGD<AdamUpdate> optimizer(
                  // Step size of the optimizer.
                  step_size,
                  // Number of data points that are used in each iteration.
                  batch_size,
                  // Max number of iterations.
                  iters_per_cycle,
                  // Tolerance, used as a stopping condition. This small number means we never
                  // stop by this condition and continue to optimize up to reaching maximum of
                  // iterations.
                  1e-8,
                  // Shuffle, If optimizer should take random data points from the dataset at
                  // each iteration.
                  true,
                  // Adam update policy.
                  AdamUpdate());
                
                std::cout << "Initial loss -> " <<
                      MeanTestLoss<ReconModel>(*m_model.model, train_test, 50) << std::endl;
                
           
                
                
                for (int i = 0; i < cycles; i++) {
                  // Train neural network. If this is the first iteration, weights are
                  // random, using current values as starting point otherwise.
//                    m_model.model->Train(train, train, optimizer, MyPrintLoss(m_dumpoutlet), EarlyStopAtMinLoss());
                    m_model.model->Train(train, train, optimizer, PrintLoss(), EarlyStopAtMinLoss());
                  // Don't reset optimizer's parameters between cycles.
                    optimizer.ResetPolicy() = false;

                    std::cout << "Loss after cycle " << i << " -> " <<
                    MeanTestLoss<ReconModel>(*m_model.model, train_test, 50)
                    << std::endl;
                }
                

                if(autoclear) {
                    m_data.reset();
                }
            } else {
                (cerr << "no data available for training.");
            }
            return {};
        }
    };
    
    message<> clear { this, "clear", "clear data and model",
        MIN_FUNCTION {
            m_data.reset();
            m_model.model.reset();
            return {};
        }
    };
    
    
    // post to max window == but only when the class is loaded the first time
    message<> maxclass_setup { this, "maxclass_setup",
        MIN_FUNCTION {
            t_class* c = args[0];
            max_jit_class_mop_wrap(c, this_jit_class, 0);
            max_jit_class_wrap_standard(c, this_jit_class, 0);
            max_jit_classex_mop_mproc(c,this_jit_class,(void*)max_jit_mlmat_mproc);
            
            class_addmethod(c, (method)max_mlmat_jit_matrix, "jit_matrix", A_GIMME, 0);
            class_addmethod(c, (method)mlmat_assist, "assist", A_CANT, 0);
            return {};
        }
    };
    
    message<> jitclass_setup {this, "jitclass_setup",
        MIN_FUNCTION {
            t_class* c = args[0];
            // add mop
            t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 3, 2));
            
            // force type
            jit_mop_single_type(mop, _jit_sym_float64);
            
            jit_class_addadornment(c, mop);
            
            //unlink dimesions between left and right i/o
            //keep planecounts same for now.
            
            auto output1 = object_method(mop,_jit_sym_getoutput,1);
            jit_attr_setlong(output1,_jit_sym_dimlink,0);
            auto output2 = object_method(mop,_jit_sym_getoutput,2);
            jit_attr_setlong(output2,_jit_sym_dimlink,0);

            auto input2 = object_method(mop,_jit_sym_getinput,2);
            jit_attr_setlong(input2,_jit_sym_dimlink,0);
            auto input3 = object_method(mop,_jit_sym_getinput,3);
            jit_attr_setlong(input3,_jit_sym_dimlink,0);
           
            //always adapt
            object_method(input2,gensym("ioproc"),jit_mop_ioproc_copy_adapt);
            object_method(input3,gensym("ioproc"),jit_mop_ioproc_copy_adapt);
            // add our custom matrix_calc method
            jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
            return {};
        }
    };
    
    void add_layer(Sequential<>* seq, const c74::min::symbol& layer_type) {
        const string layer_string = layer_type.c_str();
    
        if(layer_string == "sigmoid") {
            seq->Add<SigmoidLayer<>>();

        } else if(layer_string == "gaussian") {
            seq->Add<GaussianFunctionLayer<>>();

        } else if(layer_string == "relu") {
            seq->Add<ReLULayer<>>();

        } else if(layer_string == "tanh") {
            seq->Add<TanHLayer<>>();

        } else if(layer_string == "soft_plus") {
            seq->Add<SoftPlusLayer<>>();

        } else if(layer_string == "identity") {
            seq->Add<IdentityLayer<>>();
        }

    }
    
    t_jit_err output_latent_matrix(t_object *matrix) {
        t_jit_matrix_info in_info, minfo;
        t_jit_err err = JIT_ERR_NONE;
        arma::mat latent;
        arma::mat data;
        arma::Mat<double> rescaled_data;
        void *o,*p;
        t_atom a;

        auto* mob = maxob_from_jitob(maxobj());
        auto *mop = max_jit_obex_adornment_get(mob , _jit_sym_jit_mop);
        auto * op =  object_method(mop,_jit_sym_getoutputlist);
        auto genmatrix = linklist_getindex(static_cast<t_linklist*>(op), 1);
        
        auto genmatrix_savelock = object_method(genmatrix, _jit_sym_lock, 1);
        
        object_method(matrix, _jit_sym_getinfo, &minfo);
        
        if(!m_model.model) {
            (cerr << "no Autoencoder model has been trained" << endl);
            goto out;
        }
        
        if(minfo.dimcount > 2) {
            (cout << "expecting 1d or 2d matrix, received " << minfo.dimcount << "d matrix" << endl);
            err = JIT_ERR_INVALID_INPUT;
            goto out;
        }
        
        if(mode == 1 || mode == 2) {
            if(minfo.planecount > 1) {
                (cerr << "mode " << mode << " expects a single plane matrix but received a matrix with planecount of " << minfo.planecount << endl);
                err = JIT_ERR_INVALID_INPUT;
                           goto out;
            }
           
        }
        
        latent = jit_to_arma(mode, matrix, latent);
        
        try {
            mlpack::util::CheckSameDimensionality(latent, (size_t)latent_size.get(), "variational autoencoder", "features");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
    
        m_model.model->Forward(latent, data, 3, 3);//
        
        rescaled_data = scaler_inverse_transform(m_model, data, rescaled_data);

        minfo = in_info;
        minfo.type = _jit_sym_float64;
        minfo.flags = 0;
        minfo.planecount = rescaled_data.n_rows;
        genmatrix = arma_to_jit(mode, rescaled_data, static_cast<t_object*>(genmatrix), minfo);
        
        if ((p=object_method(mop,_jit_sym_getoutput,2   )) && (o=max_jit_mop_io_getoutlet(p)))
        {
            atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
            outlet_anything(o,_jit_sym_jit_matrix,1,&a);
        }
        
    out:
        object_method(genmatrix,_jit_sym_lock,genmatrix_savelock);
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
            check_mode(minfo, mode, "variational autoencoder");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        dat = jit_to_arma(mode, matrix, dat);

        m_data = std::make_unique<arma::Mat<double>>(dat);

    out:
        object_method((t_object*)matrix, _jit_sym_lock, savelock);
        return err;
    }
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_query_info, out_features_info;
        arma::mat query;
        arma::mat features;
        
        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto out_features = object_method(outputs, _jit_sym_getindex, 0);

        
        auto in_matrix_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_features_savelock = object_method(out_features, _jit_sym_lock, 1);
        
        object_method(in_matrix, _jit_sym_getinfo, &in_query_info);
        
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_query_info);

        try {
            check_mode(in_query_info, mode, "variational autoencoder");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
    
        
        if(!m_model.model) {
            (cerr << "no Autoencoder model has been trained" << endl);
            goto out;
        }
        
        
        
        
        query = jit_to_arma(mode, static_cast<t_object*>(in_matrix64), query);
        
        try {
            size_t p = m_model.model->Predictors().n_rows;
            mlpack::util::CheckSameDimensionality(query, p, "variational autoencoder", "query");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
     

        
        try {
            arma::mat scaled_query;
            scaled_query = scaler_transform(m_model, query, scaled_query);
            m_model.model->Forward(scaled_query, features, 1, 2);
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        

        out_features_info = in_query_info;
        out_features_info.type = _jit_sym_float64;
        out_features_info.planecount = features.n_rows;
        
        out_features = arma_to_jit(mode, features, static_cast<t_object*>(out_features), out_features_info);
        
    out:
        
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        
        object_method(in_matrix,_jit_sym_lock,in_matrix_savelock);
        object_method(out_features,_jit_sym_lock,out_features_savelock);
        
        return err;
    }
    
private:
    std::unique_ptr<arma::Mat<double>> m_data;
};


MIN_EXTERNAL(mlmat_variational_autoencoder);

void max_jit_mlmat_mproc(max_jit_wrapper *x, void *mop)
{
    t_jit_err err =JIT_ERR_NONE;
    void *o,*p, *j;
    t_atom a;
    long outputmode = max_jit_mop_getoutputmode(x);

    if (outputmode==1) {

        // send in link list that includes only output 2
        t_linklist * op =  static_cast<t_linklist*>(object_method(mop,_jit_sym_getoutputlist));
        j = max_jit_obex_jitob_get(x);
        
        t_linklist *outputlist = linklist_new();
        linklist_append(outputlist, linklist_getindex(op, 0));

        err = (t_jit_err)object_method(max_jit_obex_jitob_get(x), _jit_sym_matrix_calc,
                                       object_method(mop, _jit_sym_getinputlist),
                                       outputlist);
        
        linklist_chuck(outputlist);
        
        if(err) {
            jit_error_code(x,err);
        } else {
            if ((p=object_method(mop,_jit_sym_getoutput,1)) &&
                (o=max_jit_mop_io_getoutlet(p)))
            {
                atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
                outlet_anything(o,_jit_sym_jit_matrix,1,&a);
            }
        }
    }
}

void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv) {
    //found need to call this first or the
    // info for the incoming matrix is incorrect
    // this does not seem right
    max_jit_mop_jit_matrix(x,s,argc,argv);
    //only want second inlet
    if (max_jit_obex_inletnumber_get(x) == 1) {
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
        
        minwrap<mlmat_variational_autoencoder>* job = (minwrap<mlmat_variational_autoencoder>*)(j);
        err = job->m_min_object.output_latent_matrix((t_object*)m);
        
        jit_error_code(x,err);
    } else if (max_jit_obex_inletnumber_get(x) == 2) {
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
        minwrap<mlmat_variational_autoencoder>* job = (minwrap<mlmat_variational_autoencoder>*)(j);
        err = job->m_min_object.process_training_matrix((t_object*)m);
        
        jit_error_code(x,err);
    }
    
}


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_variational_autoencoder>* job = (minwrap<mlmat_variational_autoencoder>*)(x);
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
                    sprintf(s, "(matrix) data.");
                    break;
                    
                case 1:
                    sprintf(s, "(matrix) features.");
                    break;
                    
                case 2:
                    sprintf(s, "(matrix) training dataset.");
                    break;
                    
                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) features");
                    break;
                case 1:
                    sprintf(s, "(matrix) predicted output");
                    break;
                default:
                    sprintf(s, "dumpout");
                    break;
            }
        default:
            break;
    }
}


