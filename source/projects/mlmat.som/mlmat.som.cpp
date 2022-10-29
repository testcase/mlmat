/// @file mlmat.som.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// http://www.ai-junkie.com/ann/som/som4.html
/// TODO: in mode 1 and 2 needs to output 3d matrix

#include "mlmat.hpp"
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/methods/neighbor_search/unmap.hpp>
#include <mlpack/methods/neighbor_search/ns_model.hpp>


using namespace c74::min;
using namespace c74::max;
using namespace mlpack;

typedef NeighborSearch<NearestNeighborSort, mlpack::SquaredEuclideanDistance> SomKNN;

class SOM {
public:
    enum initialization {uniform, gaussian, sample};
    SOM() {};
    
    SOM(long cols, long rows, long weights, long epochs, long neighborhood, double rate=.01, initialization init = uniform) {
        m_num_iterations = epochs;
        m_learning_rate = rate;
        m_initialization = init;
        m_cols = cols;
        m_rows = rows;
        //will have to redo sample anyway
        if((m_initialization == uniform) || (m_initialization == sample)) {
            m_nodes = std::make_unique<arma::Mat<double>>(weights, cols*rows, arma::fill::randu);
        } else if(m_initialization == gaussian) {
            m_nodes = std::make_unique<arma::Mat<double>>(weights, cols*rows, arma::fill::randn);
        }
        m_map_radius = std::max(cols, rows) / 2;

        if(neighborhood != 0) {
            m_map_radius = MIN(neighborhood, m_map_radius);
        }

        m_time_constant = m_num_iterations/log(m_map_radius);
    }
    
    
    void run_epochs(arma::Mat<double>& data) {
        long n = m_num_iterations;

        long iter_count = 0;
        double rate = m_learning_rate;
    
        if(m_first_run && (m_initialization == sample)) {
            for(auto i=0;i<m_nodes->n_cols;i++) {
                int elem = mlpack::RandInt(data.n_cols);
                m_nodes->col(i) = data.col(elem);
            }
            m_first_run = false;
        }
        
        while(n--) {
            arma::Col<double> data_elem;
            m_neighborhood_radius = m_map_radius * exp(-(double)iter_count/m_time_constant);
            double width_sq = m_neighborhood_radius * m_neighborhood_radius;
            iter_count++;
            
            for(auto k=0;k<data.n_cols;k++) {
                m_bmu = find_bmu(data.col(k));
                
                
                for(auto i=0;i<m_nodes->n_cols;i++) {
                    long dx = (m_bmu % m_rows) - (i % m_rows);
                    long dy = (m_bmu / m_rows) - (i / m_rows);
                    double dist_sq = double((dx*dx)+(dy*dy));//squared distance
                    
                    if(dist_sq < width_sq) {
                        data_elem = data.col(k);
                        m_influence = exp(-(dist_sq)/(2*width_sq));
                        adjust_weights(i, data_elem, rate, m_influence);
                    }
                }
            }
            rate = m_learning_rate * exp(-(double)iter_count/m_num_iterations);
           
        }
    }
    
    
    
    void run_batch_knn(arma::Mat<double>& data) {
        long n = m_num_iterations;
        long iter_count = 0;
        
        if(m_first_run && (m_initialization == sample)) {
            for(auto i=0;i<m_nodes->n_cols;i++) {
                int elem = mlpack::RandInt(data.n_cols);
                m_nodes->col(i) = data.col(elem);
            }
            m_first_run = false;
        }
        
        while(n--) {
            arma::Mat<double> numerator(m_nodes->n_rows, m_nodes->n_cols, arma::fill::zeros);
            std::vector<double> denominator(m_nodes->n_cols, 0.0);
            m_neighborhood_radius = m_map_radius * exp(-(double)iter_count/m_time_constant);
            double width_sq = m_neighborhood_radius * m_neighborhood_radius;
            iter_count++;
            
            for(auto k=0;k<data.n_cols;k++) {
                m_bmu = find_bmu(data.col(k));
            
                for(auto i=0;i<m_nodes->n_cols;i++) {
                    long dx = (m_bmu % m_rows) - (i % m_rows);
                    long dy = (m_bmu / m_rows) - (i / m_rows);
                    double dist_sq = double((dx*dx)+(dy*dy));//squared distance
                    double nh = exp(-(dist_sq)/(2*width_sq)); //neighborhood scaling
                    numerator.col(i) += data.col(k) * nh;
                    denominator[i] += nh;
                }
            }
            
            for(auto j=0;j<m_nodes->n_cols;j++) {
                m_nodes->col(j) = numerator.col(j)/denominator[j];
            }
        }
    }
    
    double get_euclidean_squared(const arma::Col<double>& target, const arma::Col<double>& weights) {
        return arma::accu(arma::square(target - weights));
    }
    
    void adjust_weights(long weight_index, arma::Col<double>& target, const double learning_rate, const double influence) {
        arma::Col<double> adjust = (target - m_nodes->col(weight_index)) * (learning_rate * influence );
        m_nodes->col(weight_index) += adjust;
    }
    
 
    
    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(CEREAL_NVP(m_nodes));
    }
    
    void set_epochs(long i) {
        m_num_iterations = i;
    }
    
    std::unique_ptr<arma::Mat<double>> m_nodes { nullptr };
    
private:
    size_t find_bmu(const arma::Col<double> &vec) {
        size_t winner = 0;
        double lowest_distance = std::numeric_limits<double>::max();
        double dist;
        for(auto i=0;i<m_nodes->n_cols;i++) {
            dist = get_euclidean_squared(vec, m_nodes->col(i));
           
            if(dist < lowest_distance) {
                lowest_distance = dist;
                winner = i;
            }
        }
        return winner;
    }
    
   
    long        m_bmu;
    double      m_map_radius;
    double      m_time_constant;
    long        m_num_iterations;
    long        m_rows;
    long        m_cols;
    double      m_neighborhood_radius;
    double      m_influence;
    double      m_learning_rate;
    initialization m_initialization;
    bool m_first_run = true;
};






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
               mlpack::data::Load(string(p), "som", m_model, true);
           } catch (const std::runtime_error& s) {
               std::throw_with_nested(std::runtime_error("Error reading model file to disk."));
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

                for(auto i=0;i<m_model.model->m_nodes->n_cols;i++) {
                    dat.col(i) = m_model.model->m_nodes->col(i);
                }

                t_object* mob = maxob_from_jitob(maxobj());
                t_object *mop = static_cast<t_object*>(max_jit_obex_adornment_get(mob , _jit_sym_jit_mop));
                t_linklist * op =  static_cast<t_linklist*>(object_method(mop,_jit_sym_getoutputlist));
                t_object* genmatrix = static_cast<t_object*>(linklist_getindex(op, 0));
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
                    sprintf(s,  "(matrix) new input to map.");
                    break;

                case 1:
                    sprintf(s, "(matrix) dataset for map).");
                    break;

                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) map");
                    break;


                default:
                    sprintf(s, "dumpout");
                    break;
            }

    }
}
