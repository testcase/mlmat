/// @file mlmat.som.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// http://www.ai-junkie.com/ann/som/som4.html
/// The trained map is always output as a 2d matrix: in mode 0 as
/// (cols x rows) with the map weights as planes, in modes 1 and 2 with the
/// nodes laid out along the matrix dimensions ((weights x cols*rows) and
/// (cols*rows x weights) respectively).

#include "mlmat.hpp"
#include "mlmat_som.hpp"

using namespace c74::min;
using namespace c74::max;
using namespace mlpack;

// C function declarations
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);
void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);
void max_jit_mlmat_mproc(max_jit_wrapper *x, void *mop);

class mlmat_som : public mlmat_object_writable<mlmat_som, SOM> {
public:
    MIN_DESCRIPTION	{"Self-organizing map"};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.kmeans, mlmat.mean_shift"};

    attribute<int> rows { this, "rows", 8,
    	description {"The number of rows in map."}
    };
    
    attribute<int> cols { this, "cols", 8,
    	description {"The number of columns in map."}
    };
    
    attribute<int> epochs { this, "epochs", 1,
    	description {"The number of passes over training set."}
    
    };
    
    attribute<double> learning_rate { this, "learning_rate", .01,
    	description {"Learning rate when not in <at>batch_mode</at>."}
    };
    
    attribute<bool> batch_process { this, "batch_process", false,
    	description {"Will use batch variant of som algorithm."}
    };
    
    attribute<bool> autoclear { this, "autoclear", false};
    attribute<bool> autotrain { this, "autotrain", false};
    
    attribute<c74::min::symbol> initialization { this, "initialization", "uniform",
        range { "uniform", "gaussian", "sample" },
        description {"Set how initialization of SOM is performed. uniform uses uniform random between 0-1, gaussian uses a guassian random distribution and sample will use randomly sampled values from the input matrix."}
    };

    attribute<int> seed { this, "seed", 0,
        description {
            "Random seed."
        }
    };
    
    attribute<int> neighborhood { this, "neighborhood", 0,
        description {
            "Neighborhood radius. Zero means use full map radius"
        }
    };
    
    attribute<c74::min::symbol> file {this, "file", k_sym__empty,
        description {
            "File"
        },
        
        setter { MIN_FUNCTION {
            if(args[0] != k_sym__empty) {
                load_model_file(args);
            }
            return args;
        }}
    };

    attribute<bool> background_thread { this, "background_thread", true,
        description {
            "If set (the default), training runs on a background thread: the train message returns immediately and the patcher stays responsive. When training finishes the trained map is written out (as the synchronous path does) and a done message is sent out the dumpout outlet (error, if training failed). If 0, training blocks the main thread as before."
        }
    };
    
    message<> write {this, "write",
        MIN_FUNCTION {
           try {
               m_model.autoscale = autoscale;
               save_model_file(args, m_model, "som");
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
               mlpack::Load(string(p), "som", m_model, true);
           } catch (const std::runtime_error& s) {
               std::throw_with_nested(std::runtime_error("Error reading model file from disk."));
           }
       }
    }
    
    message<> clear { this, "clear", "clear internal map",
        MIN_FUNCTION {
            if(m_model.model) {
                m_model.model = nullptr;
                m_data.reset();
            }
            return {};
        }
    };
    
    message<> bang { this, "bang", "bang.",
        MIN_FUNCTION {
            void *o,*p;
            t_atom a;
            arma::Mat<double> rescaled_data;
            t_object* mob = maxob_from_jitob(maxobj());
            t_object *mop = static_cast<t_object*>(max_jit_obex_adornment_get(mob , _jit_sym_jit_mop));
            t_linklist * op =  static_cast<t_linklist*>(object_method(mop,_jit_sym_getoutputlist));
            t_object* genmatrix = static_cast<t_object*>(linklist_getindex(op, 0));
            genmatrix = static_cast<t_object*>(object_method(genmatrix, _jit_sym_getmatrix));
            auto genmatrix_savelock = object_method(genmatrix, _jit_sym_lock, 1);
            if ((p=object_method((t_object*)mop,_jit_sym_getoutput,1)) && (o=max_jit_mop_io_getoutlet(p)))
            {
                atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
                outlet_anything(o,_jit_sym_jit_matrix,1,&a);
            }
            object_method(genmatrix,_jit_sym_lock,genmatrix_savelock);
            
            
            return {};
        }
        
    };


    // respond to the bang message to do something
    message<> train { this, "train", "train.",
        MIN_FUNCTION {
            void *o,*p;
            t_atom a;
            long num_epochs = epochs;
            if(args.size() > 0) {
                long iters = args[0];
                if(iters > 0) {
                    num_epochs = iters;
                }
            }
            
            if(!m_data) {
                (cerr << "no data to train on." << endl);
            } else if (background_in_progress()) {
                (cerr << "a training is already in progress." << endl);
            } else if (background_thread) {
                BgParams p;
                p.data = *m_data;
                p.num_epochs = num_epochs;
                p.cols = cols;
                p.rows = rows;
                p.neighborhood = neighborhood;
                p.learning_rate = learning_rate;
                p.initialization = initialization.get().c_str();
                p.batch_process = batch_process;
                p.autoclear = autoclear;
                p.reuse_model = (m_model.model != nullptr && !autoclear);
                p.seed = seed;
                p.autoscale = autoscale;
                p.scaler_type = scaler.get().c_str();
                p.scaler_min = scaler_min;
                p.scaler_max = scaler_max;
                p.scaler_epsilon = scaler_epsilon;

                background_launch([p = std::move(p)]() mutable -> BgResult {
                    BgResult r;
                    if (p.seed == 0) {
                        mlpack::RandomSeed(time(NULL));
                    } else {
                        mlpack::RandomSeed((size_t) p.seed);
                    }

                    SOM::initialization init_type = SOM::uniform;
                    if(p.initialization == "uniform") {
                        init_type = SOM::uniform;
                    } else if(p.initialization == "gaussian") {
                        init_type = SOM::gaussian;
                    } else if(p.initialization == "sample") {
                        init_type = SOM::sample;
                    }

                    if (p.reuse_model) {
                        // training continues on a copy of the live model;
                        // the trained copy is swapped in on the main thread
                        r.model = std::make_unique<SOM>(p.cols, p.rows, p.data.n_rows, p.num_epochs, p.neighborhood, p.learning_rate, init_type);
                    } else {
                        r.model = std::make_unique<SOM>(p.cols, p.rows, p.data.n_rows, p.num_epochs, p.neighborhood, p.learning_rate, init_type);
                    }

                    arma::Mat<double> scaled_data = p.data;
                    if (p.autoscale) {
                        r.scaler = mlmat::make_scaler(p.scaler_type, p.scaler_min, p.scaler_max, p.scaler_epsilon);
                        r.scaler->Fit(p.data);
                        r.scaler->Transform(p.data, scaled_data);
                    }

                    if (p.batch_process) {
                        r.model->run_batch_knn(scaled_data);
                    } else {
                        r.model->run_epochs(scaled_data);
                    }
                    return r;
                });
            } else {
                if (seed == 0) {
                  mlpack::RandomSeed(time(NULL));
                } else {
                  mlpack::RandomSeed((size_t) seed);
                }
                
                arma::Mat<double> dat(m_data->n_rows, cols*rows );
                arma::Mat<double> scaled_data;
                arma::Mat<double> rescaled_data;

                
                SOM::initialization init_type;
                if(initialization.get() == "uniform") {
                    init_type = SOM::uniform;
                } else if(initialization.get() == "gaussian") {
                    init_type = SOM::gaussian;
                } else if(initialization.get() == "sample") {
                    init_type = SOM::sample;
                }
                
                
                if(autoclear) {
                    m_model.model = std::make_unique<SOM>(cols, rows, m_data->n_rows, num_epochs, neighborhood, learning_rate, init_type);
                } else {
                    if(!m_model.model) {
                        m_model.model = std::make_unique<SOM>(cols, rows, m_data->n_rows, num_epochs, neighborhood, learning_rate, init_type);
                    } else {
                        m_model.model->set_epochs(num_epochs);
                    }
                }
                
                scaler_fit(m_model, *m_data);
                
                scaled_data = scaler_transform(m_model, *m_data, scaled_data);
                
                if(batch_process) {
                    m_model.model->run_batch_knn(scaled_data);
                } else {
                    m_model.model->run_epochs(scaled_data);
                }

                dat = *m_model.model->m_nodes;

                t_object* mob = maxob_from_jitob(maxobj());
                t_object *mop = static_cast<t_object*>(max_jit_obex_adornment_get(mob , _jit_sym_jit_mop));
                t_linklist * op =  static_cast<t_linklist*>(object_method(mop,_jit_sym_getoutputlist));
                t_object* genmatrix = static_cast<t_object*>(linklist_getindex(op, 0));
                genmatrix = static_cast<t_object*>(object_method(genmatrix, _jit_sym_getmatrix));

                auto genmatrix_savelock = object_method(genmatrix, _jit_sym_lock, 1);
    
                rescaled_data = scaler_inverse_transform(m_model, dat, rescaled_data);
                
                // the output is always a 2d matrix. mode 0: (cols x rows) with
                // the map weights as planes; modes 1/2: arma_to_jit derives
                // the dimensions from the node matrix ((weights x cols*rows)
                // or (cols*rows x weights))
                t_jit_matrix_info minfo;
                minfo.type = _jit_sym_float64;
                minfo.flags = 0;
                minfo.planecount = rescaled_data.n_rows;
                minfo.dimcount = 2;
                minfo.dim[0] = cols;
                minfo.dim[1] = rows;

                genmatrix = arma_to_jit(mode, rescaled_data, genmatrix, minfo);
                
                if ((p=object_method((t_object*)mop,_jit_sym_getoutput,1)) && (o=max_jit_mop_io_getoutlet(p)))
                {
                    atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
                    outlet_anything(o,_jit_sym_jit_matrix,1,&a);
                }
                object_method(genmatrix,_jit_sym_lock,genmatrix_savelock);
            }
            return {};
        }
    };
    

    
    message<> jitclass_setup {this, "jitclass_setup",
        MIN_FUNCTION {
            t_class* c = args[0];
            // add mop
            t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 1, 1));
            
            // force type
            jit_mop_single_type(mop, _jit_sym_float64);
    
            
            auto output1 = object_method(mop,_jit_sym_getoutput,1);

            jit_attr_setlong(output1,_jit_sym_dimlink,0);
        
            jit_class_addadornment(c, mop);
            // add our custom matrix_calc method
            jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
            return {};
        }
    };

    // post to max window == but only when the class is loaded the first time
    message<> maxclass_setup {this, "maxclass_setup",
        MIN_FUNCTION {
            t_class* c = args[0];
            max_jit_class_mop_wrap(c, this_jit_class, 0);
            max_jit_class_wrap_standard(c, this_jit_class, 0);
            max_jit_classex_mop_mproc(c,this_jit_class,(void*)max_jit_mlmat_mproc);
            class_addmethod(c, (method)mlmat_assist, "assist", A_CANT, 0);

            return {};
        }
    };
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        arma::mat dat;
        //arma::mat scaled_data;
        t_jit_matrix_info in_query_info;
        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto out_matrix = object_method(outputs, _jit_sym_getindex, 0);
        auto in_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_savelock = object_method(out_matrix, _jit_sym_lock, 1);
        
        //need to check if rows are same
        object_method(in_matrix, _jit_sym_getinfo, &in_query_info);
        
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_query_info);


        try {
            check_mode(in_query_info, mode, "som");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        dat = jit_to_arma(mode, static_cast<t_object*>(in_matrix64), dat);
        
        m_data = std::make_unique<arma::Mat<double>>(std::move(dat));

        if(autotrain && !batch_process) {
            train();
        }
        
     out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        
        object_method(in_matrix,_jit_sym_lock,in_savelock);
        object_method(out_matrix,_jit_sym_lock,out_savelock);
        return err;
    }
        
        
    std::unique_ptr<arma::Mat<double>> m_data { nullptr };

    // ---- background training -------------------------------------------
    struct BgParams {
        arma::Mat<double> data;
        long num_epochs;
        int cols;
        int rows;
        int neighborhood;
        double learning_rate;
        std::string initialization;
        bool batch_process;
        bool autoclear;
        bool reuse_model;
        int seed;
        bool autoscale;
        std::string scaler_type;
        double scaler_min;
        double scaler_max;
        double scaler_epsilon;
    };

    struct BgResult {
        std::unique_ptr<SOM> model;
        std::unique_ptr<mlpack::ScalingModel> scaler;
    };

    void bg_commit(BgResult&& r) {
        m_model.model = std::move(r.model);
        m_model.scaler = std::move(r.scaler);
        if (autoclear) {
            m_data.reset();
        }
        write_trained_map();
    }

    // the output half of the synchronous train path, factored so the
    // background commit can run it on the main thread
    void write_trained_map() {
        if (!m_model.model) {
            return;
        }
        void *o, *p;
        t_atom a;

        arma::Mat<double> dat = *m_model.model->m_nodes;
        arma::Mat<double> rescaled_data;

        auto* mob = maxob_from_jitob(maxobj());
        auto* mop = static_cast<t_object*>(max_jit_obex_adornment_get(mob , _jit_sym_jit_mop));
        auto* op = static_cast<t_linklist*>(object_method(mop,_jit_sym_getoutputlist));
        auto* genmatrix = static_cast<t_object*>(linklist_getindex(op, 0));
        genmatrix = static_cast<t_object*>(object_method(genmatrix, _jit_sym_getmatrix));

        auto genmatrix_savelock = object_method(genmatrix, _jit_sym_lock, 1);

        rescaled_data = scaler_inverse_transform(m_model, dat, rescaled_data);

        t_jit_matrix_info minfo;
        minfo.type = _jit_sym_float64;
        minfo.flags = 0;
        minfo.planecount = rescaled_data.n_rows;
        minfo.dimcount = 2;
        minfo.dim[0] = cols;
        minfo.dim[1] = rows;

        genmatrix = arma_to_jit(mode, rescaled_data, genmatrix, minfo);

        if ((p=object_method((t_object*)mop,_jit_sym_getoutput,1)) && (o=max_jit_mop_io_getoutlet(p)))
        {
            atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
            outlet_anything(o,_jit_sym_jit_matrix,1,&a);
        }
        object_method(genmatrix,_jit_sym_lock,genmatrix_savelock);
    }
};

MIN_EXTERNAL(mlmat_som);

void max_jit_mlmat_mproc(max_jit_wrapper *x, void *mop)
{
    t_jit_err err;

    err = (t_jit_err)object_method(max_jit_obex_jitob_get(x),
                                   _jit_sym_matrix_calc,
                                   object_method(mop, _jit_sym_getinputlist),
                                   object_method(mop, _jit_sym_getoutputlist));
    
    if(err) {
        jit_error_code(x,err);
    }
}

t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_som>* job = (minwrap<mlmat_som>*)(x);
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
                    snprintf(s, 64, "(matrix) new input to map.");
                    break;

                case 1:
                    snprintf(s, 64, "(matrix) dataset for map).");
                    break;

                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    snprintf(s, 64, "(matrix) map");
                    break;


                default:
                    snprintf(s, 64, "dumpout");
                    break;
            }

    }
}
