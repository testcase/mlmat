/// @file       mlmat.hmm.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved. Also based on examples provided with the mlpack library. Please see source/mlpack for license details
/// @license  Use of this source code is governed by the MIT License found in the License.md file.


#include "mlmat.hpp"
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/hmm/hmm_model.hpp>
#include <mlpack/methods/gmm/gmm.hpp>
#include <mlpack/methods/gmm/diagonal_gmm.hpp>


using namespace c74::min;
using namespace c74::max;
using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::distribution;
using namespace mlpack::util;
using namespace mlpack::gmm;
using namespace mlpack::math;



void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);
void mlmat_assist(void* x, void* b, long io, long index, char* s);
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);
void max_jit_mlmat_mproc(max_jit_wrapper *x, void *mop);

class mlmat_hmm : public mlmat_object_writable<mlmat_hmm, HMMModel>
{
public:
    MIN_DESCRIPTION	{"Hidden Markov Model. An implementation Hidden Markov Model (HMM). The model can be based on one of four basic types set by <at>hmm_type</at>."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.gmm"};
    
    // the actual attribute for the message
    attribute<bool> autoclear { this, "autoclear", true,
        description {"Clear training data from memory after the model has been trained."}
    };
    
    
    attribute<bool> use_labels { this, "use_labels", false,
        description {"Use labels in training phase. The labels must match the number of training samples. Labels must be used when using <at>hmm_type</at> diag_gmm or gmm."}
    };
    
    attribute<int> states { this, "states", 10,
        description {
            "Number of hidden states in HMM"
        }
    };
    
    attribute<int> gaussians { this, "gaussians", 10,
        description {
            "Number of gaussians in each GMM (necessary when type is 'gmm'."
        }
    };
    
    attribute<int> seed { this, "seed", 0,
        description {
            "Random seed. 0 indicates no seed."
        }
    };
    
    attribute<double> tolerance { this, "tolerance", 1e-5,
        description {
            "Tolerance of the Baum-Welch algorithm."
        }
    };
    
    attribute<c74::min::symbol> hmm_type { this, "hmm_type", "gaussian",
        description {
            "Type of HMM"
        },
        range {"discrete", "gaussian", "diag_gmm", "gmm"}
    };
    
    // respond to the bang message to do something
    message<> clear { this, "clear", "clear previous training input.",
        MIN_FUNCTION {
            m_observations.clear();
            m_labels.clear();
            m_model.model.reset();
            return {};
        }
    };
    
    
    message<> train { this, "train", "train model.",
        MIN_FUNCTION {
            arma::mat out_data;
            
            const string type_string = hmm_type.get().c_str();
            
            if(m_observations.empty()) {
                (cerr << "No training data. Training terminated!" << endl);
                goto out;
            }
            
            HMMType typeId;
            if (type_string == "discrete") {
                typeId = HMMType::DiscreteHMM;
            } else if (type_string == "gaussian") {
                typeId = HMMType::GaussianHMM;
            } else if (type_string == "gmm") {
                typeId = HMMType::GaussianMixtureModelHMM;
            } else {
                typeId = HMMType::DiagonalGaussianMixtureModelHMM;
            }
            
            m_model.model = std::make_unique<HMMModel>(typeId);
            
            
            if(typeId == HMMType::DiscreteHMM) {
                HMM<distribution::DiscreteDistribution>* hmm = m_model.model->DiscreteHMM();
                init_hmm(hmm);
            }
            else if (typeId == HMMType::GaussianHMM) {
                HMM<distribution::GaussianDistribution>* hmm = m_model.model->GaussianHMM();
                init_hmm(hmm);
                
            } else if (typeId == HMMType::GaussianMixtureModelHMM) {
                HMM<gmm::GMM>* hmm = m_model.model->GMMHMM();
                init_hmm(hmm);
            } else if(typeId == HMMType::DiagonalGaussianMixtureModelHMM) {
                HMM<gmm::DiagonalGMM>* hmm = m_model.model->DiagGMMHMM();
                init_hmm(hmm);
            }
            
            if(autoclear) {
                m_observations.clear();
                m_labels.clear();
            }
            
            
        out:
            
            return {};
        }
    };
    
    message<> generate { this, "generate",
        MIN_FUNCTION {
            auto num = 1;
            auto start = 0;
            if(args.size() > 0) {
                num = args[0];
                if(args.size() > 1) {
                    start = args[1];
                }
            }
            
            void *o,*p;
            t_atom a;
            arma::Mat<double> samples;
            arma::Row<size_t> states;
            const string type_string = hmm_type.get().c_str();
            
            auto* mob = maxob_from_jitob(maxobj());
            auto *mop = max_jit_obex_adornment_get(mob , _jit_sym_jit_mop);
            auto * op =  object_method(mop,_jit_sym_getoutputlist);
            auto genmatrix = linklist_getindex(static_cast<t_linklist*>(op), 0);
            
            auto genmatrix_savelock = object_method(genmatrix, _jit_sym_lock, 1);
            
            
            HMMType typeId;
            
            if (type_string == "discrete") {
                typeId = HMMType::DiscreteHMM;
            } else if (type_string == "gaussian") {
                typeId = HMMType::GaussianHMM;
            } else if (type_string == "gmm") {
                typeId = HMMType::GaussianMixtureModelHMM;
            } else {
                typeId = HMMType::DiagonalGaussianMixtureModelHMM;
            }
            
            if(!m_model.model) {
                (cerr << "no HMM model has been trained" << endl);
                goto out;
            }
            
            if (seed == 0) {
                mlpack::math::RandomSeed(time(NULL));
            } else {
                mlpack::math::RandomSeed((size_t) seed);
            }
            
            
            if(typeId == HMMType::DiscreteHMM) {
                HMM<distribution::DiscreteDistribution>* hmm = m_model.model->DiscreteHMM();
                hmm->Generate(num, samples, states, start);
            }
            else if (typeId == HMMType::GaussianHMM) {
                HMM<distribution::GaussianDistribution>* hmm = m_model.model->GaussianHMM();
                hmm->Generate(num, samples, states, start);
                
            } else if (typeId == HMMType::GaussianMixtureModelHMM) {
                HMM<gmm::GMM>* hmm = m_model.model->GMMHMM();
                hmm->Generate(num, samples, states, start);
            } else if(typeId == HMMType::DiagonalGaussianMixtureModelHMM) {
                HMM<gmm::DiagonalGMM>* hmm = m_model.model->DiagGMMHMM();
                hmm->Generate(num, samples, states, start);
            }
            
            t_jit_matrix_info minfo;
            minfo.type = _jit_sym_float64;
            minfo.flags = 0;
            minfo.planecount = samples.n_rows;
            minfo.dimcount = 1;
            minfo.dim[0] = samples.n_cols;
            minfo.dim[1] = 1;
            
            genmatrix = arma_to_jit(mode, samples, static_cast<t_object*>(genmatrix), minfo);
            
            
            if ((p=object_method(mop,_jit_sym_getoutput,1)) && (o=max_jit_mop_io_getoutlet(p)))
            {
                atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
                outlet_anything(o,_jit_sym_jit_matrix,1,&a);
            }
            
            std::cout << states << std::endl;
            
        out:
            object_method(genmatrix,_jit_sym_lock,genmatrix_savelock);
            return {};
        },
        "Generate points based on distribution.",
        message_type::usurp_low
    };
    
    
    
    message<> jitclass_setup {this, "jitclass_setup",
        MIN_FUNCTION {
            t_class* c = args[0];
            t_atom long_type[1];
            
            // add mop
            t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 3, 2));
            
            atom_setsym(long_type, _jit_sym_long);
            
            // force type
            jit_mop_single_type(mop, _jit_sym_float64);
            
            auto output1 = object_method(mop,_jit_sym_getoutput,1);
            auto output2 = object_method(mop,_jit_sym_getoutput,2);
            auto input2 = object_method(mop,_jit_sym_getinput,2);
            auto input3 = object_method(mop,_jit_sym_getinput,3);
            
            jit_attr_setlong(output1,_jit_sym_dimlink,0);
            jit_attr_setlong(output2,_jit_sym_dimlink,0);
            jit_attr_setlong(input2,_jit_sym_dimlink,0);
            jit_attr_setlong(input3,_jit_sym_dimlink,0);
            
            object_method_typed(input3, _jit_sym_types, 1, long_type, NULL);
            object_method_typed(output2, _jit_sym_types, 1, long_type, NULL);
            
            //always adapt
            object_method(input2,gensym("ioproc"),jit_mop_ioproc_copy_adapt);
            object_method(input3,gensym("ioproc"),jit_mop_ioproc_copy_adapt);
            
            jit_class_addadornment(c, mop);
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
    
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_query_info, out_info;
        arma::mat query;
        t_atom a[1];
        double loglik = 0;
        arma::Row<size_t> state_seq;
        HMMType typeId;
        
        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto out_states = object_method(outputs, _jit_sym_getindex, 0);
        auto in_matrix_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_states_savelock = object_method(out_states, _jit_sym_lock, 1);
        
        object_method(in_matrix, _jit_sym_getinfo, &in_query_info);
        
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_query_info);
        
        
        query = jit_to_arma(mode, static_cast<t_object*>(in_matrix64), query);
        
        if(!m_model.model) {
            (cerr << "no HMM model has been trained" << endl);
            goto out;
        }
        
        typeId = m_model.model->Type();
        
        if(typeId == HMMType::DiscreteHMM) {
            HMM<distribution::DiscreteDistribution>* hmm = m_model.model->DiscreteHMM();
            if(hmm != nullptr) {
                try {
                    CheckSameDimensionality(query, hmm->Emission()[0].Dimensionality(), "gmm", "sequence");
                } catch (std::invalid_argument& s) {
                    cerr << s.what() << endl;
                    goto out;
                }
            }
            loglik = hmm->LogLikelihood(query);
            
            const double p = hmm->Predict(query, state_seq);
            (cout << "predict " << p << endl);
        }  else if (typeId == HMMType::GaussianHMM) {
            HMM<distribution::GaussianDistribution>* hmm = m_model.model->GaussianHMM();
            if(hmm != nullptr) {
                try {
                    CheckSameDimensionality(query, hmm->Emission()[0].Dimensionality(), "gmm", "sequence");
                } catch (std::invalid_argument& s) {
                    cerr << s.what() << endl;
                    goto out;
                }
            }
            loglik = hmm->LogLikelihood(query);
            const double p = hmm->Predict(query, state_seq);
            (cout << "predict " << p << endl);
        } else if (typeId == HMMType::GaussianMixtureModelHMM) {
            HMM<gmm::GMM>* hmm = m_model.model->GMMHMM();
            if(hmm != nullptr) {
                try {
                    CheckSameDimensionality(query, hmm->Emission()[0].Dimensionality(), "gmm", "sequence");
                } catch (std::invalid_argument& s) {
                    cerr << s.what() << endl;
                    goto out;
                }
            }
            loglik = hmm->LogLikelihood(query);
            const double p = hmm->Predict(query, state_seq);
            (cout << "predict " << p << endl);
        } else if(typeId == HMMType::DiagonalGaussianMixtureModelHMM) {
            HMM<gmm::DiagonalGMM>* hmm = m_model.model->DiagGMMHMM();
            if(hmm != nullptr) {
                try {
                    CheckSameDimensionality(query, hmm->Emission()[0].Dimensionality(), "gmm", "sequence");
                } catch (std::invalid_argument& s) {
                    cerr << s.what() << endl;
                    goto out;
                }
            }
            loglik = hmm->LogLikelihood(query);
            const double p = hmm->Predict(query, state_seq);
            (cout << "predict " << p << endl);
        }
        
        out_info = in_query_info;
        in_query_info.type = _jit_sym_long;
        
        out_states = arma_to_jit(mode, state_seq, static_cast<t_object*>(out_states), out_info);
        atom_setfloat(a,loglik);
        outlet_anything(m_dumpoutlet, gensym("loglik"), 1, a);
        
    out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        object_method(in_matrix,_jit_sym_lock,in_matrix_savelock);
        object_method(out_states,_jit_sym_lock,out_states_savelock);
        return err;
        
    }
    
    t_jit_err init_hmm(HMM<DiscreteDistribution>* hmm) {
        t_jit_err err = JIT_ERR_NONE;
        arma::Col<size_t> maxEmissions(m_observations[0].n_rows);
        maxEmissions.zeros();
        for (vector<arma::mat>::iterator it = m_observations.begin(); it != m_observations.end(); ++it) {
            arma::Col<size_t> maxSeqs = arma::conv_to<arma::Col<size_t>>::from(arma::max(*it, 1)) + 1;
            maxEmissions = arma::max(maxEmissions, maxSeqs);
        }
        
        if(hmm != nullptr) {
            *hmm = HMM<DiscreteDistribution>(size_t(states), DiscreteDistribution(maxEmissions), tolerance);
            std::vector<DiscreteDistribution>& e = hmm->Emission();
            for (size_t i = 0; i < e.size(); ++i)
            {
                e[i].Probabilities().randu();
                e[i].Probabilities() /= arma::accu(e[i].Probabilities());
            }
        }
        if((err=check_observations(m_observations, hmm->Emission()[0].Dimensionality()))) {
            return err;
        };
        
        if(use_labels) {
            if(check_labels(m_labels, m_observations, hmm->Transition().n_cols)) {
                return err;;
            }
            hmm->Train(m_observations, m_labels);
        } else {
            hmm->Train(m_observations);
        }
        return err;
    }
    
    t_jit_err init_hmm(HMM<GaussianDistribution>* hmm) {
        t_jit_err err = JIT_ERR_NONE;
        // Find dimension of the data.
        const size_t dimensionality = m_observations[0].n_rows;
        
        // Verify dimensionality of data.
        for (size_t i = 0; i < m_observations.size(); ++i)
        {
            if (m_observations[i].n_rows != dimensionality)
            {
                (cerr << "Observation sequence " << i << " dimensionality ("
                 << m_observations[i].n_rows << " is incorrect (should be "
                 << dimensionality << ")!" << endl);
            }
        }
        
        if(hmm != nullptr) {
            *hmm = HMM<GaussianDistribution>(size_t(states), GaussianDistribution(dimensionality), tolerance);
            
            std::vector<GaussianDistribution>& e = hmm->Emission();
            for (size_t i = 0; i < e.size(); ++i) {
                const size_t dimensionality = e[i].Mean().n_rows;
                e[i].Mean().randu();
                // Generate random covariance.
                arma::mat r = arma::randu<arma::mat>(dimensionality, dimensionality);
                e[i].Covariance(r * r.t());
            }
        }
        if((err=check_observations(m_observations,hmm->Emission()[0].Dimensionality()))) {
            return err;
        };
        
        if(use_labels) {
            if(check_labels(m_labels, m_observations, hmm->Transition().n_cols)) {
                return err;
            }
            // std::cout << m_labels[0] << std::endl;
            hmm->Train(m_observations, m_labels);
        } else {
            hmm->Train(m_observations);
        }
        return err;
    }
    
    t_jit_err init_hmm(HMM<GMM>* hmm) {
        t_jit_err err = JIT_ERR_NONE;
        // Find dimension of the data.
        const size_t dimensionality = m_observations[0].n_rows;
        
        if (gaussians == 0) {
            (cerr << "Number of gaussians for each GMM must be specified "
             << "when type = 'gmm'!" << endl);
        }
        
        if (gaussians < 0) {
            (cerr << "Invalid number of gaussians (" << gaussians << "); must "
             << "be greater than or equal to 1." << endl);
        }
        
        if(hmm != nullptr) {
            *hmm = HMM<GMM>(size_t(states), GMM(size_t(gaussians), dimensionality), tolerance);
            
            std::vector<GMM>& e = hmm->Emission();
            
            for (size_t i = 0; i < e.size(); ++i)
            {
                // Random weights.
                e[i].Weights().randu();
                e[i].Weights() /= arma::accu(e[i].Weights());
                
                // Random means and covariances.
                for (int g = 0; g < gaussians; ++g)
                {
                    const size_t dimensionality = e[i].Component(g).Mean().n_rows;
                    e[i].Component(g).Mean().randu();
                    
                    // Generate random covariance.
                    arma::mat r = arma::randu<arma::mat>(dimensionality,
                                                         dimensionality);
                    e[i].Component(g).Covariance(r * r.t());
                }
            }
        }
        
        if((err=check_observations(m_observations,hmm->Emission()[0].Dimensionality()))) {
            return err;
        };
        
        if(use_labels) {
            if((err=check_labels(m_labels, m_observations, hmm->Transition().n_cols))) {
                return err;
            }
            hmm->Train(m_observations, m_labels);
        } else {
            (cwarn << "Unlabeled training of GMM HMMs is almost certainly not going to produce good results! Training terminated. " << endl);
            //hmm->Train(m_observations);
        }
        return err;
    }
    
    t_jit_err init_hmm(HMM<DiagonalGMM>* hmm) {
        t_jit_err err = JIT_ERR_NONE;
        
        const size_t dimensionality = m_observations[0].n_rows;
        
        if (gaussians == 0){
            (cerr << "Number of gaussians for each GMM must be specified " << "when type = 'diag_gmm'!" << endl);
        }
        if (gaussians < 0)  {
            (cerr << "Invalid number of gaussians (" << gaussians << "); must "  << "be greater than or equal to 1." << endl);
        }
        
        if(hmm != nullptr) {
            *hmm = HMM<DiagonalGMM>(size_t(states), DiagonalGMM(size_t(gaussians), dimensionality), tolerance);
            
            std::vector<DiagonalGMM>& e = hmm->Emission();
            
            for (size_t i = 0; i < e.size(); ++i) {
                // Random weights.
                e[i].Weights().randu();
                e[i].Weights() /= arma::accu(e[i].Weights());
                
                // Random means and covariances.
                for (int g = 0; g < gaussians; ++g) {
                    const size_t dimensionality = e[i].Component(g).Mean().n_rows;
                    e[i].Component(g).Mean().randu();
                    
                    // Generate random diagonal covariance.
                    arma::vec r = arma::randu<arma::vec>(dimensionality);
                    e[i].Component(g).Covariance(r);
                }
            }
        }
        
        if((err=check_observations(m_observations,hmm->Emission()[0].Dimensionality()))) {
            return err;
        };
        
        if(use_labels) {
            if((err=check_labels(m_labels, m_observations, hmm->Transition().n_cols))) {
                return err;
            }
            hmm->Train(m_observations, m_labels);
        } else {
            (cwarn << "Unlabeled training of Diagonal GMM HMMs is almost certainly not going to produce good results! Training terminated. " << endl);
            //hmm->Train(m_observations);
        }
        return err;
    }
    
    
    t_jit_err process_observations_matrix(t_object *matrix) {
        t_jit_matrix_info minfo;
        t_jit_err err = JIT_ERR_NONE;
        arma::mat dat;
        
        long savelock = (long) object_method((t_object*)matrix, _jit_sym_lock, 1);
        object_method((t_object*)matrix, _jit_sym_getinfo, &minfo);
        
        if(minfo.dimcount > 2) {
            (cerr << "expecting 1d or 2d matrix, received " << minfo.dimcount << "d matrix" << endl);
            err = JIT_ERR_INVALID_INPUT;
            goto out;
        }
        
        try {
            check_mode(minfo, mode, "hmm");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
        dat = jit_to_arma(mode, matrix, dat);
        
        m_observations.push_back(std::move(dat));
    out:
        object_method(matrix, _jit_sym_lock, savelock);
        return err;
    }
    
    t_jit_err process_labels_matrix(t_object *matrix) {
        t_jit_matrix_info minfo;
        t_jit_err err = JIT_ERR_NONE;
        arma::Mat<size_t> dat;
        
        long savelock = (long) object_method((t_object*)matrix, _jit_sym_lock, 1);
        
        if(!use_labels) {
            (cerr << "labels matrix not used as use_labels attribute is set to 0." << endl);
            goto out;
        }
        
        object_method((t_object*)matrix, _jit_sym_getinfo, &minfo);
        
        if(minfo.dimcount > 2) {
            (cerr << "expecting 1d or 2d matrix, received " << minfo.dimcount << "d matrix" << endl);
            err = JIT_ERR_INVALID_INPUT;
            goto out;
        }
        
        if(minfo.planecount > 1) {
            (cerr << "mode " << mode << " expects a single plane matrix but received a matrix with planecount of " << minfo.planecount << endl);
            err = JIT_ERR_INVALID_INPUT;
            goto out;
        }
        
        dat = jit_to_arma(mode, matrix, dat);
        
        m_labels.push_back(std::move(dat));
    out:
        object_method(matrix, _jit_sym_lock, savelock);
        return err;
    }
    
private:
    
    t_jit_err check_observations(vector<arma::Mat<double>>& obs, const size_t s ) {
        t_jit_err err = JIT_ERR_NONE;//
        for(size_t i=0;i<obs.size();i++) {
            if(obs[i].n_rows != s) {
                
                cerr << "Dimensionality of training sequence " << i << " (" << obs[i].n_rows << ") is not equal to the dimensionality of " << "the HMM (" << s << ")!"
                << endl;
                return JIT_ERR_GENERIC;
            }
        }
        return err;
    }
    
    t_jit_err check_labels(vector<arma::Row<size_t>>& labels,  vector<arma::Mat<double>>& obs, const size_t ) {
        t_jit_err err = JIT_ERR_NONE;//
        // are vectors same size
        if(labels.size() != obs.size()) {
            cerr << "Label sequence does not have the same number of points as observation sequence!" << endl;
            return JIT_ERR_GENERIC;
        }
        
        //do labels and observations have same number of entries
        for(size_t i=0;i<obs.size();i++) {
            if(obs[i].n_cols != labels[i].n_cols) {
                cerr << "Label sequence " <<  i << " does not have"
                << " the same number of points (" << labels[i].n_cols << ") as observation sequence "
                << obs[i].n_cols << "!" << endl;
                return JIT_ERR_GENERIC;
            }
            //do labels match hidden states of hmm
            for (size_t j = 0; j < labels[i].n_cols; j++) {
                if (labels[i][j] > states) {
                    cerr << "HMM has " << states << " hidden "
                    << "states, but labels contain " << labels[i][j]
                    << " (should be between 0 and "
                    << states << ")!" << endl;
                    return JIT_ERR_GENERIC;
                }
            }
        }
        return err;
    }
    vector<arma::Mat<double>> m_observations;
    vector<arma::Row<size_t>> m_labels;
};

MIN_EXTERNAL(mlmat_hmm);

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
        
        minwrap<mlmat_hmm>* job = (minwrap<mlmat_hmm>*)(j);
        err = job->m_min_object.process_observations_matrix(static_cast<t_object*>(m));
        
        jit_error_code(x,err);
    }
    
    if (max_jit_obex_inletnumber_get(x) == 2)
    {
        t_jit_err err = JIT_ERR_NONE;
        void *j = nullptr;
        
        void *p, *m, *mop;
        
        if (!(mop=max_jit_obex_adornment_get(x,_jit_sym_jit_mop)))
            jit_error_code(x,err);
        
        p = object_method((t_object*)mop,_jit_sym_getinput,3);
        m = object_method((t_object*)p,_jit_sym_getmatrix);
        j = max_jit_obex_jitob_get(x);
        
        minwrap<mlmat_hmm>* job = (minwrap<mlmat_hmm>*)(j);
        err = job->m_min_object.process_labels_matrix(static_cast<t_object*>(m));
        
        jit_error_code(x,err);
    }
    
}

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
        linklist_append(outputlist, linklist_getindex(op, 1));
        
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
        }
    }
}


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_hmm>* job = (minwrap<mlmat_hmm>*)(x);
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
                    sprintf(s, "Training sequence");
                    break;
                    
                case 2:
                    sprintf(s, "Training labels");
                    break;
                    
                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "Generated samples from the pre-trained HMM");
                    break;
                    
                case 1:
                    sprintf(s, "Most probable hidden state sequence");
                    break;
                    
                default:
                    sprintf(s, "dumpout");
                    break;
            }
        default:
            break;
    }
}


