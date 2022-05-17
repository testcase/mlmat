/// @file mlmat.gmm.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.


#include "c74_min.h"
#include <mlpack/methods/gmm/gmm.hpp>
#include <mlpack/methods/gmm/diagonal_gmm.hpp>
#include <mlpack/methods/gmm/no_constraint.hpp>
#include <mlpack/methods/gmm/diagonal_constraint.hpp>
#include <mlpack/methods/kmeans/refined_start.hpp>
#include "mlmat_object.hpp"

using namespace c74::min;
using namespace c74::max;
using namespace mlpack::gmm;
using namespace mlpack::util;
using namespace mlpack::kmeans;
using max::t_linklist;


void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);
void mlmat_assist(void* x, void* b, long io, long index, char* s);
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);
void max_jit_mlmat_mproc(max_jit_wrapper *x, void *mop);


class mlmat_gmm : public mlmat_object_writable<mlmat_gmm, GMM>
{
public:

    MIN_DESCRIPTION	{"A Gaussian Mixture Model. A Gaussian Mixture Model takes a parametric estimate of a Gaussian mixture model (GMM) using the EM algorithm to find the maximum likelihood estimate."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.knn, mlmat.hmm, mlmat.linear_svm"};
    MIN_DISCUSSION  {"In addition to being able to determine the probability that input comes from the learned distribution, this object has a number of features for generating data based ont the learned model. The <at>generate</at> message can be used to create N number of randomly generated observations based on the model. The <at>component</at> message can be used to generate N observations from a single component of the model. The <at>weights</at> message can be used to alter the relative weights of the components when generating random observations using the <at>weighted_generate</at> message."};
    
    attribute<bool> autoclear { this, "autoclear", false, 
    	description {"Clear training data from memory after the model has been trained."}
    };
    
    
    //  define an optional argument for setting the message
    attribute<int> gaussians { this, "gaussians", 10,
        description { "Number of Gaussians in the GMM."},
    };
    
    attribute<int> seed { this, "seed", 0,
        description {
            "Random seed."
        }
    };
    
    attribute<int> trials { this, "trials", 1,
        description {
            "Number of trials to perform in training GMM."
        },
        setter { MIN_FUNCTION {
            int value = args[0];
            
            if (value < 1)
                value = 1;
                return {value};
        }}
    };
    
    attribute<bool> diagonal_covariance { this, "diagonal_covariance", true,
        description {
            "Force the covariance of the Gaussians to be diagonal.  This can accelerate training time significantly."
        }
    };
    
    attribute<bool> classify { this, "classify", false,
        description {
            "Classify the given observations as being from an individual component in this GMM. This will output a matrix out fourth outlet"
        }
    };
    
    attribute<double> noise { this, "noise", 0.,
        description {
            "Variance of zero-mean Gaussian noise to add to data."
        }
    };
    
    attribute<int> kmeans_max_iterations { this, "kmeans_max_iterations", 1000,
        description {
            "Maximum number of iterations for the k-means algorithm (used to initialize EM)."
        },
        setter { MIN_FUNCTION {
            int value = args[0];
            
            if (value < 1) {
                value = 1;
            }
            return {value};
        }}
    };
    
    attribute<int> max_iterations { this, "max_iterations", 250,
        description {
            "Maximum number of iterations of EM algorithm (passing 0 will run until convergence)."
        },
        setter { MIN_FUNCTION {
            int value = args[0];
            
            if (value < 1) {
                value = 1;
            }
            return {value};
        }}
    };
    
    attribute<bool> refined_start { this, "refined_start", false,
        description {
            "During the initialization, use refined initial positions for k-means clustering."
        }
    };
    
    
    attribute<bool> no_force_positive { this, "no_force_positive", false,
        description {
            "Do not force the covariance matrices to be positive definite. Recommend leaving this as false."
        }
    };
    
    attribute<int> samplings { this, "samplings", 100,
        description {
            "If using refined_start, specify the number of samplings used for initial points."
            
        },
        setter { MIN_FUNCTION {
            int value = args[0];
            
            if (value < 1) {
                value = 1;
            }
            return {value};
        }}
    };
    
    attribute<double> tolerance { this, "tolerance", 1e-5,
        description {
            "Tolerance for convergence of EM."
        }
    };
    
    attribute<double> percentage { this, "percentage", 0.02,
        description {
            "If using refined_start, specify the percentage of the dataset used for each sampling."
        },
        range {0.0, 1.0}
    };
    
    message<> clear { this, "clear", "clear data and model",
        MIN_FUNCTION {
            m_data.reset();
            m_model.model.reset();
            return {};
        }
    };
    
    message<> weights { this, "weights", "weights for gaussians",
        MIN_FUNCTION {
            // if just weights reset weights to default
            if(args.size() == 0) {
                m_weights.reset();
        
            } else if(args.size() != gaussians) {
                //error because weights don't equal gaussians
                cerr << "Number of weights must equal the number of gaussians. Received "
                << args.size()
                << " weights with "
                << gaussians
                << " gaussians."
                << endl;

            } else {
                // catch if any args are not numbers
                for(auto i=0;i<args.size();i++) {
                    if((args[i].a_type != A_FLOAT) && (args[i].a_type != A_LONG)) {
                        (cerr << "Weights must only contain numerical data." << endl);
                        return {};
                    }
                }
                m_weights = std::make_unique<arma::vec>(gaussians);
                for(auto i=0;i<args.size();i++) {
                    (*m_weights)(i) = double(args[i]);
                }
                double n = arma::sum(*m_weights);
                m_weights->transform( [&](double val) { return val / n; } );
            }
            return {};
        }
    };

    message<> train { this, "train", "Train model.",
        MIN_FUNCTION {
            auto trials = 1;
        
            if(args.size() > 0) {
                trials = args[0];
                if(trials < 1) {
                    trials = 1;
                }
            }

            if(!m_data) {
                (cerr << "no observations for training." << endl );
            } else if(m_data->n_cols < gaussians) {
                (cerr << "need more observations than gaussians. have " <<  m_data->n_cols << ", need at least " << gaussians << "." << endl );
                
            } else {
                arma::Mat<double> scaled_data;
                scaler_fit(m_model, *m_data);
                scaled_data = scaler_transform(m_model, *m_data, scaled_data);
                train_gmm(scaled_data);
                if(autoclear) m_data.reset();
            }
            return {};
        }
    };
    message<> weighted_generate { this, "weighted_generate",
        MIN_FUNCTION {
           // t_object *x = (t_object*)max_jit_obex_jitob_get(this);
            auto num = 1;
            if(args.size() > 0) {
                num = args[0];
            }
            
            void *o,*p;
            t_atom a;
            arma::mat samples;
            arma::mat scaled_samples;
            
            auto* mob = maxob_from_jitob(maxobj());
            auto *mop = max_jit_obex_adornment_get(mob , _jit_sym_jit_mop);
            auto * op =  object_method(mop,_jit_sym_getoutputlist);
            auto genmatrix = linklist_getindex(static_cast<t_linklist*>(op), 0);
            
            auto genmatrix_savelock = object_method(genmatrix, _jit_sym_lock, 1);

            if(!m_model.model) {
                (cerr << "no GMM model has been trained" << endl);
                goto out;
            }

            samples.set_size(m_model.model->Dimensionality(), num);
            
            if (seed == 0) {
              mlpack::math::RandomSeed(time(NULL));
            } else {
              mlpack::math::RandomSeed((size_t) seed);
            }
            
            if(!m_weights) {
                m_weights = std::make_unique<arma::vec>(gaussians, arma::fill::value(1.0/(double)gaussians));
            }
            
            for (size_t i = 0; i < num; i++) {
                samples.col(i) = WeightedRandom();
            }
            
        
            scaled_samples = scaler_inverse_transform(m_model, samples, scaled_samples);
            
            t_jit_matrix_info minfo;
            minfo.type = _jit_sym_float64;
            minfo.flags = 0;
            minfo.planecount = scaled_samples.n_rows;
            minfo.dimcount = 1;
            minfo.dim[0] = scaled_samples.n_cols;
            minfo.dim[1] = 1;

            genmatrix = arma_to_jit(mode, scaled_samples, static_cast<t_object*>(genmatrix), minfo);
                        
            if ((p=object_method(mop,_jit_sym_getoutput,1)) && (o=max_jit_mop_io_getoutlet(p)))
            {
                atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
                outlet_anything(o,_jit_sym_jit_matrix,1,&a);
            }
            
        out:
            object_method(genmatrix,_jit_sym_lock,genmatrix_savelock);
            return {};
        },
        "Generate points based on distribution and weighted by <at>weights</at>.",
        message_type::usurp_low
    };

    message<> generate { this, "generate",
        MIN_FUNCTION {
            auto num = 1;
            if(args.size() > 0) {
                num = args[0];
            }
            
            void *o,*p;
            t_atom a;
            arma::mat samples;
            arma::mat scaled_samples;
            
            auto* mob = maxob_from_jitob(maxobj());
            auto *mop = max_jit_obex_adornment_get(mob , _jit_sym_jit_mop);
            auto * op =  object_method(mop,_jit_sym_getoutputlist);
            auto genmatrix = linklist_getindex(static_cast<t_linklist*>(op), 0);
            
            auto genmatrix_savelock = object_method(genmatrix, _jit_sym_lock, 1);

            if(!m_model.model) {
                (cerr << "no GMM model has been trained" << endl);
                goto out;
            }

            samples.set_size(m_model.model->Dimensionality(), num);
            
            if (seed == 0) {
              mlpack::math::RandomSeed(time(NULL));
            } else {
              mlpack::math::RandomSeed((size_t) seed);
            }
        
            
            for (size_t i = 0; i < num; i++) {
                samples.col(i) = m_model.model->Random();
            }
            
        
            scaled_samples = scaler_inverse_transform(m_model, samples, scaled_samples);
            
            t_jit_matrix_info minfo;
            minfo.type = _jit_sym_float64;
            minfo.flags = 0;
            minfo.planecount = scaled_samples.n_rows;
            minfo.dimcount = 1;
            minfo.dim[0] = scaled_samples.n_cols;
            minfo.dim[1] = 1;

            genmatrix = arma_to_jit(mode, scaled_samples, static_cast<t_object*>(genmatrix), minfo);
                        
            if ((p=object_method(mop,_jit_sym_getoutput,1)) && (o=max_jit_mop_io_getoutlet(p)))
            {
                atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
                outlet_anything(o,_jit_sym_jit_matrix,1,&a);
            }
            
        out:
            object_method(genmatrix,_jit_sym_lock,genmatrix_savelock);
            return {};
        },
        "Generate points based on distribution.",
        message_type::usurp_low
    };
    
    
    message<> component { this, "component",
        MIN_FUNCTION {
            auto component = 0;
            auto num = 1;
            
            if(args.size() == 1) {
                component = args[0];
                
                if(component > (gaussians - 1)) {
                    component = (gaussians - 1);
                } else if(component < 0) {
                    component = 0;
                }
            }
            
            if(args.size() == 2) {
                component = args[0];
                num = args[1];
                
                if(component > (gaussians - 1)) {
                    component = (gaussians - 1);
                } else if(component < 0) {
                    component = 0;
                }
            }
            
            void *o,*p;
            t_atom a;
            arma::mat samples;
            arma::mat scaled_samples;
            
            auto* mob = maxob_from_jitob(maxobj());
            auto *mop = max_jit_obex_adornment_get(mob , _jit_sym_jit_mop);
            auto * op =  object_method(mop,_jit_sym_getoutputlist);
            auto genmatrix = linklist_getindex(static_cast<t_linklist*>(op), 0);
            
            auto genmatrix_savelock = object_method(genmatrix, _jit_sym_lock, 1);

            if(!m_model.model) {
                (cerr << "no GMM model has been trained" << endl);
                goto out;
            }

            samples.set_size(m_model.model->Dimensionality(), num);
            
            if (seed == 0) {
              mlpack::math::RandomSeed(time(NULL));
            } else {
              mlpack::math::RandomSeed((size_t) seed);
            }
            
            for (size_t i = 0; i < num; i++) {
                samples.col(i) = m_model.model->Component(component).Random();
            }
            
            scaled_samples = scaler_inverse_transform(m_model, samples, scaled_samples);

            t_jit_matrix_info minfo;
            minfo.type = _jit_sym_float64;
            minfo.flags = 0;
            minfo.planecount = scaled_samples.n_rows;
            minfo.dimcount = 1;
            minfo.dim[0] = scaled_samples.n_cols;
            minfo.dim[1] = 1;

            
            genmatrix = arma_to_jit(mode, scaled_samples, static_cast<t_object*>(genmatrix), minfo);
            
            
            if ((p=object_method(mop,_jit_sym_getoutput,1)) && (o=max_jit_mop_io_getoutlet(p)))
            {
                atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
                outlet_anything(o,_jit_sym_jit_matrix,1,&a);
            }
            
        out:
            object_method(genmatrix,_jit_sym_lock,genmatrix_savelock);
            return {};
        },
        "Generate points using specified component of the GMM.",
        message_type::usurp_low
    };
    
    t_jit_err process_observations_matrix(t_object *matrix) {
        t_jit_matrix_info minfo;
        t_jit_err err = JIT_ERR_NONE;
        arma::mat dat;
        arma::mat scaled_data;
        
        long savelock = (long) object_method((t_object*)matrix, _jit_sym_lock, 1);
        object_method((t_object*)matrix, _jit_sym_getinfo, &minfo);
        
        if(minfo.dimcount > 2) {
            (cerr << "expecting 1d or 2d matrix, received " << minfo.dimcount << "d matrix" << endl);
            err = JIT_ERR_INVALID_INPUT;
            goto out;
        }
        
        try {
            check_mode(minfo, mode, "gmm");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        dat = jit_to_arma(mode, matrix, dat);
        
        m_data = std::make_unique<arma::Mat<double>>(std::move(dat));


    out:
        object_method(matrix, _jit_sym_lock, savelock);
        return err;
    }
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_query_info, out_probabilities_info;
        arma::mat query;
        arma::mat scaled_query;
        arma::Row<double> probabilities;
        arma::Row<double> log_probabilities;


        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto observations_matrix = object_method(inputs, _jit_sym_getindex, 1);
        auto out_probabilities = object_method(outputs, _jit_sym_getindex, 0);
        auto out_log_probabilities = object_method(outputs, _jit_sym_getindex, 1);

        auto in_matrix_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto observations_matrix_savelock = object_method(observations_matrix, _jit_sym_lock, 1);
        auto out_probabilities_savelock = object_method(out_probabilities, _jit_sym_lock, 1);
        auto out_log_probabilities_savelock = object_method(out_log_probabilities, _jit_sym_lock, 1);
        
        object_method(in_matrix, _jit_sym_getinfo, &in_query_info);
        
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_query_info);
            
        if(!m_model.model) {
            (cerr << "no GMM model has been trained" << endl);
            goto out;
        }

        try {
            check_mode(in_query_info, mode, "gmm");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }

        query = jit_to_arma(mode, static_cast<t_object*>(in_matrix64), query);

        try {
            CheckSameDimensionality(query, m_model.model->Dimensionality(), "gmm");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        
        scaled_query = scaler_transform(m_model, query, scaled_query);
        
        probabilities.set_size(scaled_query.n_cols);
        log_probabilities.set_size(scaled_query.n_cols);
       
     
        for (size_t i = 0; i < scaled_query.n_cols; i++) {
            probabilities[i] = m_model.model->Probability(scaled_query.unsafe_col(i));
            log_probabilities[i] = m_model.model->LogProbability(scaled_query.unsafe_col(i));
        }
        
        if(classify) {
            auto out_labels = object_method(outputs, _jit_sym_getindex, 2);
            auto out_labels_savelock = object_method(out_labels, _jit_sym_lock, 1);
            t_jit_matrix_info minfo;
            minfo = in_query_info;
            minfo.planecount = 1;
            minfo.type = _jit_sym_long;
            arma::Row<size_t> labels;
            m_model.model->Classify(scaled_query, labels);
            
            out_labels = arma_to_jit(mode, labels,  static_cast<t_object*>(out_labels), minfo);
            
            object_method(out_labels,_jit_sym_lock,out_labels_savelock);
        }
        
        out_probabilities_info = in_query_info;
        out_probabilities_info.planecount = 1;
     
        out_probabilities = arma_to_jit(mode, probabilities, static_cast<t_object*>(out_probabilities), out_probabilities_info);
        out_log_probabilities = arma_to_jit(mode, log_probabilities, static_cast<t_object*>(out_log_probabilities), out_probabilities_info);
    
    out:
        
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        
        object_method(in_matrix,_jit_sym_lock,in_matrix_savelock);
        object_method(observations_matrix,_jit_sym_lock,observations_matrix_savelock);
        object_method(out_probabilities,_jit_sym_lock,out_probabilities_savelock);
        object_method(out_log_probabilities,_jit_sym_lock,out_log_probabilities_savelock);
       

        return err;
    }
    
    
private:
    
    message<> jitclass_setup {this, "jitclass_setup",
        MIN_FUNCTION {
            t_class* c = args[0];
            // add mop
            t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 2, 4));
            
            // force type
            jit_mop_single_type(mop, _jit_sym_float64);
            
            jit_class_addadornment(c, mop);
            
            
            auto output1 = object_method(mop,_jit_sym_getoutput,1);
            jit_attr_setlong(output1,_jit_sym_dimlink,0);
            auto output2 = object_method(mop,_jit_sym_getoutput,2);
            jit_attr_setlong(output2,_jit_sym_dimlink,0);
            auto output3 = object_method(mop,_jit_sym_getoutput,3);
            jit_attr_setlong(output3,_jit_sym_dimlink,0);
            auto output4 = object_method(mop,_jit_sym_getoutput,4);
            jit_attr_setlong(output4,_jit_sym_dimlink,0);
            auto input2 = object_method(mop,_jit_sym_getinput,2);
            jit_attr_setlong(input2,_jit_sym_dimlink,0);
            //always adapt
            object_method(input2,gensym("ioproc"),jit_mop_ioproc_copy_adapt);

            // add our custom matrix_calc method
            jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
            return {};
        }
    };
    
    message<> maxclass_setup {this, "maxclass_setup",
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
     
    
 
    //adapted from gmm.cpp
    //
    arma::vec WeightedRandom() {
        double gaussRand = mlpack::math::Random();
        size_t gaussian = 0;
        double sumProb = 0;
        for (size_t g = 0; g < gaussians; g++) {
            sumProb += (*m_weights)(g);
            if (gaussRand <= sumProb) {
                gaussian = g;
                break;
            }
        }
       // std::cout << *m_weights << std::endl;
        arma::mat cholDecomp;
        
        if (!arma::chol(cholDecomp, m_model.model->Component(gaussian).Covariance())) {
            cerr << "Cholesky decomposition failed." << endl;
        }
        return trans(cholDecomp) * arma::randn<arma::vec>(m_model.model->Dimensionality()) + m_model.model->Component(gaussian).Mean();
        
        
    }
     

        
    void train_gmm(arma::Mat<double>& data) {
        double likelihood;
        t_atom a[1];
        
        if(noise > 0.) { //adding noise if specified. noise is variance of noise
            data += noise * arma::randn(data.n_rows, data.n_cols);
        }
        
        if(refined_start) {
            m_model.model = std::make_unique<GMM>(gaussians,data.n_rows);
                  
            typedef KMeans<mlpack::metric::SquaredEuclideanDistance, RefinedStart> KMeansType;

            KMeansType k(kmeans_max_iterations, mlpack::metric::SquaredEuclideanDistance(), RefinedStart(samplings, percentage));

            if(diagonal_covariance) {
                DiagonalGMM dgmm(m_model.model->Gaussians(), m_model.model->Dimensionality());
                for (size_t i = 0; i < size_t(gaussians); ++i) {
                    dgmm.Component(i).Mean() = m_model.model->Component(i).Mean();
                    dgmm.Component(i).Covariance(std::move(arma::diagvec(m_model.model->Component(i).Covariance())));
                }
            
                dgmm.Weights() = m_model.model->Weights();

                EMFit<KMeansType, PositiveDefiniteConstraint, mlpack::distribution::DiagonalGaussianDistribution> em(max_iterations, tolerance, k);
                likelihood = dgmm.Train(data, trials, false, em);
                for (size_t i = 0; i < size_t(gaussians); ++i) {
                    m_model.model->Component(i).Mean() = dgmm.Component(i).Mean();
                    m_model.model->Component(i).Covariance(arma::diagmat(dgmm.Component(i).Covariance()));
                }
                m_model.model->Weights() = dgmm.Weights();
            
            } else if (no_force_positive) {
                // Compute the parameters of the model using the EM algorithm.
                EMFit<KMeansType> em(max_iterations, tolerance, k);
                likelihood = m_model.model->Train(data, trials, false, em);
            } else {
                EMFit<KMeansType, NoConstraint> em(max_iterations, tolerance, k);
                likelihood = m_model.model->Train(data, trials, false, em);
            }
        } else {
            m_model.model = std::make_unique<GMM>(gaussians,m_data->n_rows);
            typedef KMeans<mlpack::metric::SquaredEuclideanDistance, RefinedStart> KMeansType;

            KMeansType k(kmeans_max_iterations, mlpack::metric::SquaredEuclideanDistance(), RefinedStart(samplings, percentage));

            if(diagonal_covariance) {
                DiagonalGMM dgmm(m_model.model->Gaussians(), m_model.model->Dimensionality());
                for (size_t i = 0; i < size_t(gaussians); ++i) {
                    dgmm.Component(i).Mean() = m_model.model->Component(i).Mean();
                    dgmm.Component(i).Covariance(std::move(arma::diagvec(m_model.model->Component(i).Covariance())));
                }
            
                dgmm.Weights() = m_model.model->Weights();

                EMFit<KMeans<>, PositiveDefiniteConstraint, mlpack::distribution::DiagonalGaussianDistribution> em(max_iterations, tolerance, KMeans<>(kmeans_max_iterations));
                likelihood = dgmm.Train(data, trials, false, em);
                for (size_t i = 0; i < size_t(gaussians); ++i) {
                    m_model.model->Component(i).Mean() = dgmm.Component(i).Mean();
                    m_model.model->Component(i).Covariance(arma::diagmat(dgmm.Component(i).Covariance()));
                }
                m_model.model->Weights() = dgmm.Weights();
            
            } else if (no_force_positive) {
                // Compute the parameters of the model using the EM algorithm.
                EMFit<> em(max_iterations, tolerance, KMeans<>(kmeans_max_iterations));
                likelihood = m_model.model->Train(data, trials, false, em);
            } else {
                KMeans<> k(kmeans_max_iterations);
                EMFit<KMeans<>, NoConstraint> em(max_iterations, tolerance, k);
                likelihood = m_model.model->Train(data, trials, false, em);
            }
        }
        
        atom_setfloat(a,likelihood);
        outlet_anything(m_dumpoutlet, gensym("likelihood"), 1, a);
    }
    
    std::unique_ptr<arma::Mat<double>> m_data { nullptr };
    std::unique_ptr<arma::vec> m_weights { nullptr };
};


MIN_EXTERNAL(mlmat_gmm);


void max_jit_mlmat_mproc(max_jit_wrapper *x, void *mop)
{
    t_jit_err err =JIT_ERR_NONE;
    void *o,*p, *j;
    t_atom a;
    long outputmode = max_jit_mop_getoutputmode(x);

    if (outputmode==1) {
        
        bool classify = false;
        
        
        // send in link list that includes only outputs 2 3 4
        t_linklist * op =  static_cast<t_linklist*>(object_method(mop,_jit_sym_getoutputlist));
        j = max_jit_obex_jitob_get(x);
        minwrap<mlmat_gmm>* job = (minwrap<mlmat_gmm>*)(j);
    
        classify = job->m_min_object.classify;
        
        
        t_linklist *outputlist = linklist_new();
        linklist_append(outputlist, linklist_getindex(op, 1));
        linklist_append(outputlist, linklist_getindex(op, 2));
        
        if(classify) {
            linklist_append(outputlist, linklist_getindex(op, 3));
        }
        
        err = (t_jit_err)object_method(max_jit_obex_jitob_get(x),
                                       _jit_sym_matrix_calc,
                                       object_method(mop, _jit_sym_getinputlist),
                                       outputlist);
        
        linklist_chuck(outputlist);
        
        if(err) {
            jit_error_code(x,err);
        } else {
            if ((p=object_method(mop,_jit_sym_getoutput,2)) &&
                (o=max_jit_mop_io_getoutlet(p)))
            {
                atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
                outlet_anything(o,_jit_sym_jit_matrix,1,&a);
            }
           
            if ((p=object_method(mop,_jit_sym_getoutput,3)) &&
                    (o=max_jit_mop_io_getoutlet(p)))
            {
                atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
                outlet_anything(o,_jit_sym_jit_matrix,1,&a);
            }
            
            if(classify) {
                if ((p=object_method(mop,_jit_sym_getoutput,4)) &&
                        (o=max_jit_mop_io_getoutlet(p)))
                {
                    atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
                    outlet_anything(o,_jit_sym_jit_matrix,1,&a);
                }
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
        
        if (!(mop=max_jit_obex_adornment_get(x,_jit_sym_jit_mop)))
            jit_error_code(x,err);
        
        p = object_method((t_object*)mop,_jit_sym_getinput,2);
        m = object_method((t_object*)p,_jit_sym_getmatrix);
        j = max_jit_obex_jitob_get(x);
        
        minwrap<mlmat_gmm>* job = (minwrap<mlmat_gmm>*)(j);
        err = job->m_min_object.process_observations_matrix(static_cast<t_object*>(m));
        
        jit_error_code(x,err);
    }
    
}


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_gmm>* job = (minwrap<mlmat_gmm>*)(x);
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
                    sprintf(s, "Input matrix to calculate probabilities of");
                    break;
                    
                case 1:
                    sprintf(s, "Training data on which the model will be fit");
                    break;
                    
                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "Generated samples from the pre-trained GMM");
                    break;
                    
                case 1:
                    sprintf(s, "Probability that given points came from this GMM");
                    break;
                    
                case 2:
                    sprintf(s, "Log-probability that given points came from this GMM");
                    break;
                    
                default:
                    sprintf(s, "dumpout");
                    break;
            }
           default:
            break;
    }
}
