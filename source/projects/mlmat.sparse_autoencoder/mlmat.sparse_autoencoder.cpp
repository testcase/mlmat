/// @file mlmat.sparse_autoencoder.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved. Also based on examples provided with the mlpack library. Please see source/mlpack for license details
/// @license  Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"

#include "sparse_autoencoder_ext.hpp"

#include "mlmat.hpp"
#include <mlpack/core.hpp>


using namespace c74::min;
using namespace c74::max;
using namespace mlpack;
using c74::max::t_linklist;

t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) ;
void mlmat_assist(void* x, void* b, long io, long index, char* s);
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);
void max_jit_mlmat_mproc(max_jit_wrapper *x, void *mop);

class mlmat_sparse_autoencoder : public mlmat_object_writable<mlmat_sparse_autoencoder, SparseAutoencoderExt> {
public:
    MIN_DESCRIPTION	{"Sparse autoencoder. An autoencoder with a single hidden layer that attempt to impose a sparsity constraint."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.mlp_regressor"};
    
    attribute<int> hidden_size { this, "hidden_size", 2,
        description {
            "Size of input vector expected at the hidden layer."
        }
    };
    
    attribute<double> lambda { this, "lambda", .0001,
        description {
            "L2-regularization parameter."
        }
    };
    attribute<double> beta { this, "beta", 3.0,
        description {
            "KL divergence parameter."
        }
    };
    
    attribute<double> rho { this, "rho", .01,
        description {
            "Sparsity parameter."
        }
    };
    
    attribute<bool> autoclear {this, "autoclear", true,
        description {
            "Clear training data after training is run."
        }
    };

    attribute<bool> background_thread { this, "background_thread", true,
        description {
            "If set (the default), training runs on a background thread: the train message returns immediately, the object stays responsive, and queries keep using the previous model until the new one is ready. A done message is sent out the dumpout outlet when training finishes (error, if it failed). If 0, training blocks the main thread as before."
        }
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
               save_model_file(args, m_model, "sparse_autoencoder");
           } catch (const std::runtime_error& s) {
               (cout << s.what() << endl);
           }
           return {};
        }
    };
    
    message<> read {this, "read",
        MIN_FUNCTION {
            // min::error() THROWS std::runtime_error by design (it is
            // not a console post) -- canceling the file dialog throws
            // "file not chosen" from the path constructor, and a bad
            // model file throws from mlpack::Load. without this catch
            // the exception escapes into Max's event loop and
            // terminates the process
            try {
                load_model_file(args);
                autoscale = m_model.autoscale;
                m_mode_changed = false;
            } catch (const std::exception& s) {
                cerr << s.what() << endl;
            }
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
               mlpack::Load(string(p), "sparse_autoencoder", m_model, true);
           } catch (const std::runtime_error& s) {
               std::throw_with_nested(std::runtime_error("Error reading model file from disk."));
           }
       }
    }
    
    message<> train { this, "train", "train model.",
        MIN_FUNCTION {
            if(background_in_progress()) {
                return {};
            }

            if(m_training) {
                if (background_thread) {
                    BgParams p;
                    p.training = *m_training;
                    p.hidden_size = hidden_size;
                    p.lambda = lambda;
                    p.beta = beta;
                    p.rho = rho;
                    p.autoscale = autoscale;
                    p.scaler_type = scaler.get().c_str();
                    p.scaler_min = scaler_min;
                    p.scaler_max = scaler_max;
                    p.scaler_epsilon = scaler_epsilon;

                    background_launch([p = std::move(p)]() mutable -> BgResult {
                        BgResult r;
                        arma::mat out_data = p.training;
                        if (p.autoscale) {
                            r.scaler = mlmat::make_scaler(p.scaler_type, p.scaler_min, p.scaler_max, p.scaler_epsilon);
                            r.scaler->Fit(p.training);
                            r.scaler->Transform(p.training, out_data);
                        }
                        // the ext constructor performs the training
                        r.model = std::make_unique<SparseAutoencoderExt>(out_data, out_data.n_rows, p.hidden_size, p.lambda, p.beta, p.rho);
                        return r;
                    });
                    return {};
                }
                arma::mat out_data;
                scaler_fit(m_model, *m_training);
                out_data = scaler_transform(m_model, *m_training, out_data);
                m_model.model = std::make_unique<SparseAutoencoderExt>(out_data, out_data.n_rows, hidden_size, lambda, beta, rho);
        
                if(autoclear) {
                    m_training.reset();
                } 
            } else {
                (cerr << "no data available for training.");
            }
            return {};
        }
    };
    
    message<> clear { this, "clear", "clear data and model",
        MIN_FUNCTION {
            m_training.reset();
            m_model.model.reset();
            return {};
        }
    };

    // ---- background training -------------------------------------------
    struct BgParams {
        arma::Mat<double> training;
        int hidden_size;
        double lambda;
        double beta;
        double rho;
        bool autoscale;
        std::string scaler_type;
        double scaler_min;
        double scaler_max;
        double scaler_epsilon;
    };

    struct BgResult {
        std::unique_ptr<SparseAutoencoderExt> model;
        std::unique_ptr<mlpack::ScalingModel> scaler;
    };

    void bg_commit(BgResult&& r) {
        m_model.model = std::move(r.model);
        m_model.scaler = std::move(r.scaler);
        if (autoclear) {
            m_training.reset();
        }
    }
    
    
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
            
            auto input2 = object_method(mop,_jit_sym_getinput,2);
            auto input3 = object_method(mop,_jit_sym_getinput,3);
            
            jit_attr_setlong(input2,_jit_sym_dimlink,0);
            jit_attr_setlong(input3,_jit_sym_dimlink,0);
           
            //always adapt
            object_method(input2,gensym("ioproc"),jit_mop_ioproc_copy_adapt);
            object_method(input3,gensym("ioproc"),jit_mop_ioproc_copy_adapt);
            // add our custom matrix_calc method
            jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
            return {};
        }
    };
    
    
    t_jit_err output_prediction_matrix(t_object *matrix) {
        t_jit_matrix_info in_info, minfo;
        t_jit_err err = JIT_ERR_NONE;
        arma::mat features;
        arma::mat data;
        arma::mat scaled_data;
        
        void *o,*p;
        t_atom a;

        
        auto* mob = maxob_from_jitob(maxobj());
        auto *mop = max_jit_obex_adornment_get(mob , _jit_sym_jit_mop);
        auto * op =  object_method(mop,_jit_sym_getoutputlist);
        auto genmatrix = linklist_getindex(static_cast<t_linklist*>(op), 1);
        
        auto genmatrix_savelock = object_method(genmatrix, _jit_sym_lock, 1);
        
        
        object_method(matrix, _jit_sym_getinfo, &in_info);
        
        if(!m_model.model) {
            (cerr << "no Autoencoder model has been trained" << endl);
            goto out;
        }
        
        if(in_info.dimcount > 2) {
            (cout << "expecting 1d or 2d matrix, received " << in_info.dimcount << "d matrix" << endl);
            err = JIT_ERR_INVALID_INPUT;
            goto out;
        }
        
        if(mode == 1 || mode == 2) {
            if(in_info.planecount > 1) {
                (cerr << "mode " << mode << " expects a single plane matrix but received a matrix with planecount of " << in_info.planecount << endl);
                err = JIT_ERR_INVALID_INPUT;
                           goto out;
            }
           
        }
        
        features = jit_to_arma(mode, matrix, features);
        
        try {
            mlpack::util::CheckSameDimensionality(features, m_model.model->HiddenSize(), "sparse autoencoder", "features");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
    
        try {
            m_model.model->Predict(features, data);
            scaled_data = scaler_inverse_transform(m_model,data, scaled_data);
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        minfo = in_info;
        minfo.type = _jit_sym_float64;
        minfo.flags = 0;
        minfo.planecount = scaled_data.n_rows;
        
        genmatrix = arma_to_jit(mode, scaled_data, static_cast<t_object*>(genmatrix), minfo);
        
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
            check_mode(minfo, mode, "sparse autoencoder");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        dat = jit_to_arma(mode, matrix, dat);
        
        m_training = std::make_unique<arma::Mat<double>>(dat);

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
        //std::cout << "query_matrix" << std::endl;
        t_object* query_matrix = convert_to_float64(static_cast<t_object*>(in_matrix), in_query_info);

        try {
            check_mode(in_query_info, mode, "sparse autoencoder");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        if(!m_model.model) {
            (cerr << "no Autoencoder model has been trained" << endl);
            goto out;
        }
        
        query = jit_to_arma(mode, static_cast<t_object*>(query_matrix), query);
        
        try {
            mlpack::util::CheckSameDimensionality(query, m_model.model->VisibleSize(), "sparse autoencoder", "query");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        try {
            arma::mat scaled_query;
            scaled_query = scaler_transform(m_model, query, scaled_query);
            m_model.model->GetNewFeatures(scaled_query, features);

        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        out_features_info = in_query_info;
        out_features_info.type = _jit_sym_float64;
        out_features_info.planecount = features.n_rows;
        out_features = arma_to_jit(mode, features, static_cast<t_object*>(out_features), out_features_info);
        
    out:
        // free the conversion matrix (convert_to_float64 returns the
        // input itself when no conversion was needed -- the sibling
        // objects' pattern)
        if(query_matrix != nullptr && query_matrix != in_matrix) { jit_object_free(query_matrix); }
        object_method(in_matrix,_jit_sym_lock,in_matrix_savelock);
        object_method(out_features,_jit_sym_lock,out_features_savelock);

        return err;
    }
    
private:
    std::unique_ptr<arma::Mat<double>> m_training;
};


MIN_EXTERNAL(mlmat_sparse_autoencoder);

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
        
        minwrap<mlmat_sparse_autoencoder>* job = (minwrap<mlmat_sparse_autoencoder>*)(j);
        err = job->m_min_object.output_prediction_matrix((t_object*)m);
        
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
        minwrap<mlmat_sparse_autoencoder>* job = (minwrap<mlmat_sparse_autoencoder>*)(j);
        err = job->m_min_object.process_training_matrix((t_object*)m);
        
        jit_error_code(x,err);
    }
    
}


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_sparse_autoencoder>* job = (minwrap<mlmat_sparse_autoencoder>*)(x);
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
                    snprintf(s, 64, "(matrix) data.");
                    break;
                    
                case 1:
                    snprintf(s, 64, "(matrix) features.");
                    break;
                    
                case 2:
                    snprintf(s, 64, "(matrix) training dataset.");
                    break;
                    
                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    snprintf(s, 64, "(matrix) features");
                    break;
                case 1:
                    snprintf(s, 64, "(matrix) predicted output");
                    break;
                default:
                    snprintf(s, 64, "dumpout");
                    break;
            }
        default:
            break;
    }
}
