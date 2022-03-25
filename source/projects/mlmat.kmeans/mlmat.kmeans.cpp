/// @file mlmat.kmeans.cpp
/// @ingroup mlmat
/// @copyright Copyright 2018 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
///
/// TODO: Input for initial centroids
/// TODO: Other distance Metrics?
/// TODO: Serialization read/write
/// TODO: Testing

#include "c74_min.h"
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/kmeans/allow_empty_clusters.hpp>
#include <mlpack/methods/kmeans/kill_empty_clusters.hpp>
#include <mlpack/methods/kmeans/refined_start.hpp>
#include <mlpack/methods/kmeans/elkan_kmeans.hpp>
#include <mlpack/methods/kmeans/hamerly_kmeans.hpp>
#include <mlpack/methods/kmeans/pelleg_moore_kmeans.hpp>
#include <mlpack/methods/kmeans/dual_tree_kmeans.hpp>
#include "mlmat_operator.hpp"
using namespace c74::min;
using namespace c74::max;
using namespace mlpack;
using namespace mlpack::kmeans;

// C function declarations
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);
void mlmat_assist(void* x, void* b, long io, long index, char* s);
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);
class mlmat_kmeans : public mlmat_operator<mlmat_kmeans> {
public:
    MIN_DESCRIPTION	{"K means clustering. An implementation of several strategies for efficient k-means clustering. Given a dataset and a value of k, this computes and returns a k-means."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.knn, mlmat.kfn"};
    MIN_DISCUSSION  {""};

    inlet<>  input1	    { this, "(matrix) Input dataset to perform clustering on.", "matrix" };
    inlet<>  input2     { this, "(matrix) Initial clustering centroids.", "matrix" };
    outlet<> output1    { this, "(matrix) Cluster assignments.", "matrix" };
    outlet<> output2    { this, "(matrix) Cluster centroids.", "matrix" };

    attribute<int> clusters { this, "clusters", 1,
        description {"Number of clusters to find."},
        setter { MIN_FUNCTION {
            int value = args[0];

            if (value < 1)
                value = 1;
                return {value};
        }}
    };
    
    attribute<min::symbol> algorithm { this, "algorithm", "naive",
        description {
            "Algorithm to use for the Lloyd iteration."
        },
        range{"naive", "pelleg-moore", "elkan", "hamerly", "dualtree", "dualtree-covertree"}
    };
    
    attribute<int> max_iterations { this, "max_iterations", 1000,
        description {"Maximum number of iterations before k-means terminates."}
    };
    
    // will not include kill_empty_clusters. if allow_empty_clusters false will
    // kill_empty_clusters. this means object will not use the 3rd slower
    // approach.
    attribute<bool> allow_empty_clusters { this, "allow_empty_clusters", false,
        description {"Allow empty clusters to be persist."}
    };
    
    attribute<bool> reuse_centroids { this, "reuse_centroids", false,
        description {"Use previous calculated centroids for current clustering."}
    };
    
    attribute<bool> reuse_assignments { this, "reuse_assignments", false,
        description {"Use previous assignments for current clustering. If both reuse_centroids and reuse_assignments are 1, reuse_assignments overrides."}
    };
    
    attribute<bool> refined_start { this, "refined_start", false,
        description {"Use refined start sampling to partition data."}
    };
    
    attribute<int> samplings { this, "samplings", 100,
        description {
            "Number of samplings to perform for refined start (use when refined_start is specified)."
        }
    };
    
    attribute<double> percentage { this, "percentage", .02,
        description {"Percentage of dataset to use for each refined start sampling (use when refined_start is specified)."}
    };
    
    attribute<int> seed { this, "seed", 0,
        description {
            "Random seed if random basis being used. 0 indicates no seed."
        }
    };

    // Given the type of initial partition policy, figure out the empty cluster
    // policy and run k-means.
    template<typename InitialPartitionPolicy>
    void FindEmptyClusterPolicy(const InitialPartitionPolicy& ipp,
                                arma::Mat<double>& dataset,
                                arma::Row<size_t>& assignments,
                                arma::Mat<double>& centroids,
                                const bool initialAssignmentGuess,
                                const bool initialCentroidGuess) {
        if (allow_empty_clusters) {
            FindLloydStepType<InitialPartitionPolicy, AllowEmptyClusters>(ipp, dataset, assignments, centroids, initialAssignmentGuess, initialCentroidGuess);
        }
        else {
            FindLloydStepType<InitialPartitionPolicy, KillEmptyClusters>(ipp, dataset, assignments, centroids, initialAssignmentGuess, initialCentroidGuess);
        }
    }

    // Given the initial partitionining policy and empty cluster policy, figure out
    // the Lloyd iteration step type and run k-means.
    template<typename InitialPartitionPolicy, typename EmptyClusterPolicy>
    void FindLloydStepType(const InitialPartitionPolicy& ipp,
                           arma::Mat<double>& dataset,
                           arma::Row<size_t>& assignments,
                           arma::Mat<double>& centroids,
                           const bool initialAssignmentGuess,
                           const bool initialCentroidGuess) {
        const string algorithm_string = algorithm.get().c_str();
        
        if (algorithm_string == "elkan") {
            RunKMeans<InitialPartitionPolicy, EmptyClusterPolicy, ElkanKMeans>(ipp, dataset, assignments, centroids, initialAssignmentGuess, initialCentroidGuess);
        }
        else if (algorithm_string == "hamerly") {
            RunKMeans<InitialPartitionPolicy, EmptyClusterPolicy, HamerlyKMeans>(ipp, dataset, assignments, centroids, initialAssignmentGuess, initialCentroidGuess);
        }
        else if (algorithm_string == "pelleg-moore") {
            RunKMeans<InitialPartitionPolicy, EmptyClusterPolicy, PellegMooreKMeans>(ipp, dataset, assignments, centroids, initialAssignmentGuess, initialCentroidGuess);
        }
        else if (algorithm_string == "dualtree") {
            RunKMeans<InitialPartitionPolicy, EmptyClusterPolicy, DefaultDualTreeKMeans>(ipp, dataset, assignments, centroids, initialAssignmentGuess, initialCentroidGuess);
        }
        else if (algorithm_string == "dualtree-covertree") {
            RunKMeans<InitialPartitionPolicy, EmptyClusterPolicy, CoverTreeDualTreeKMeans>(ipp, dataset, assignments, centroids, initialAssignmentGuess, initialCentroidGuess);
        }
        else if (algorithm_string == "naive") {
            RunKMeans<InitialPartitionPolicy, EmptyClusterPolicy, NaiveKMeans>(ipp, dataset, assignments, centroids, initialAssignmentGuess, initialCentroidGuess);
        }
    }
    
    template<typename InitialPartitionPolicy,
             typename EmptyClusterPolicy,
             template<class, class> class LloydStepType>
    void RunKMeans(const InitialPartitionPolicy& ipp,
                   arma::Mat<double>& dataset,
                   arma::Row<size_t>& assignments,
                   arma::Mat<double>& centroids,
                   const bool initialAssignmentGuess,
                   const bool initialCentroidGuess) {
        
        
        if (seed != 0) {
           math::RandomSeed((size_t) seed);
        } else {
           math::RandomSeed((size_t) std::time(NULL));
        }

        
        KMeans<metric::EuclideanDistance,
            InitialPartitionPolicy,
            EmptyClusterPolicy,
            LloydStepType> kmeans(max_iterations, metric::EuclideanDistance(), ipp);

        
        kmeans.Cluster(dataset, clusters, assignments, centroids, initialAssignmentGuess, initialCentroidGuess);
    }


    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        arma::Mat<double> dat;
        arma::Row<size_t> assignments;
        arma::Mat<double> centroids;
        KMeans<> kmeans;
        t_jit_matrix_info in_query_info, in_centroids_info, out_assignments_info, out_centroids_info;
        bool initial_assignment_guess;
        bool initial_centroid_guess;


        auto in_matrix = (t_object*)object_method(inputs, _jit_sym_getindex, 0);
        auto in_centroids = (t_object*)object_method(inputs, _jit_sym_getindex, 1);
        auto out_assignments = (t_object*)object_method(outputs, _jit_sym_getindex, 0);
        auto out_centroids = (t_object*)object_method(outputs, _jit_sym_getindex, 1);
        
        auto query_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto centroids_savelock = object_method(in_centroids, _jit_sym_lock, 1);
        auto out_assignments_savelock = object_method(out_assignments, _jit_sym_lock, 1);
        auto out_centroids_savelock = object_method(out_centroids, _jit_sym_lock, 1);
        
        
        object_method(in_matrix, _jit_sym_getinfo, &in_query_info);
        object_method(in_centroids, _jit_sym_getinfo, &in_centroids_info);
        
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_query_info);
        t_object* in_centroids64 = convert_to_float64(static_cast<t_object*>(in_centroids), in_centroids_info);
        
        try {
            check_mode(in_query_info, mode, "kmeans");
        } catch (const std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
      
       
        dat = jit_to_arma(mode, in_matrix64, dat);
        

        if(m_received_centroids) {
            centroids = jit_to_arma(mode, in_centroids64, centroids);
            
            if(centroids.n_rows != dat.n_rows) {
                (cerr << "incorrect number of elements in centroids. expecting " << dat.n_rows << " but got " << centroids.n_rows << ". ignoring." << endl);
                m_received_centroids = false;
                goto out;
            }
            
            if(centroids.n_cols != clusters) {
                (cerr << "incorrect number of centroids. expecting " << clusters << " but got " << centroids.n_cols << ". ignoring." << endl);
                m_received_centroids = false;
                goto out;
            }
        }
        

        if(m_previous_centroids) {
            if(!reuse_centroids || m_previous_centroids->n_cols != clusters) {
                m_previous_centroids.reset();
            }
        }
     
        
        if(m_previous_assignments) {
            if(!reuse_assignments || m_previous_assignments->n_cols != clusters ) {
                m_previous_assignments.reset();
            }
        }
        
                
        initial_assignment_guess = reuse_assignments && m_previous_assignments;
        initial_centroid_guess = reuse_centroids && m_previous_centroids;

        if(initial_assignment_guess) {
            assignments = arma::Row<size_t>(*m_previous_assignments);
        }
        
        if(initial_centroid_guess) {
            centroids = arma::Mat<double>(*m_previous_centroids);
        }
        
        if(refined_start) {
            try {
                FindEmptyClusterPolicy<RefinedStart>(RefinedStart(samplings, percentage), dat, assignments, centroids, initial_assignment_guess, initial_centroid_guess);
            } catch (const std::out_of_range& s) {
                cerr << "Not enough samples for refined start. Try increasing percentage attribute." << endl;
                goto out;
            }
        } else {
            FindEmptyClusterPolicy<SampleInitialization>(SampleInitialization(), dat, assignments, centroids, initial_assignment_guess, initial_centroid_guess);
        }
           
        
        if(reuse_centroids) {
            m_previous_centroids = std::make_unique<arma::Mat<double>>(centroids);
        }
        
        if(reuse_assignments) {
            m_previous_assignments = std::make_unique<arma::Row<size_t>>(assignments);
        }
        

    
        out_assignments_info = in_query_info;
        out_assignments_info.type = _jit_sym_long;
        out_assignments_info.planecount = 1;

        out_assignments = arma_to_jit(mode, assignments, out_assignments, out_assignments_info);
        
        out_centroids_info.type = _jit_sym_float64;
        out_centroids_info.dimcount = 1;
        out_centroids_info.dim[0] = centroids.n_cols;
        out_centroids_info.planecount = centroids.n_rows;
        out_centroids = arma_to_jit(mode, centroids, out_centroids,out_centroids_info);

    out:
        if(in_centroids != in_centroids64) { jit_object_free(in_centroids64); }
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        m_received_centroids = false;
        
        object_method(in_matrix,_jit_sym_lock,query_savelock);
        object_method(in_centroids,_jit_sym_lock,centroids_savelock);
        object_method(out_assignments,_jit_sym_lock,out_assignments_savelock);
        object_method(out_centroids,_jit_sym_lock,out_centroids_savelock);

        return err;
    }

    void set_received_centroids(bool b) {
        m_received_centroids = b;
    }
    

private:
    
    
// override jitclass_setup so we can have our own matrix_calc. jitclass_setup is called first (and only once when the object is loaded for the first time) during the intitialization of the object.
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        // add mop
        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 2, 2));
        
        // force type
        jit_mop_single_type(mop, _jit_sym_float64);
        
        jit_class_addadornment(c, mop);
        
        //unlink dimesions between left and right i/o
        //keep planecounts same for now.
        auto input2 = object_method(mop,_jit_sym_getinput,2);
        auto output1 = object_method(mop,_jit_sym_getoutput,1);
        auto output2 = object_method(mop,_jit_sym_getoutput,2);
        
        jit_attr_setlong(input2,_jit_sym_dimlink,0);
        jit_attr_setlong(output1,_jit_sym_dimlink,0);
        jit_attr_setlong(output2,_jit_sym_dimlink,0);

        //always adapt
        object_method((t_object*)input2,gensym("ioproc"),(void*)jit_mop_ioproc_copy_adapt);
        // add our custom matrix_calc method
        jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
        return {};
    }};
    
    message<> maxob_setup {this, "maxob_setup",
        MIN_FUNCTION {
            t_object* mob = maxob_from_jitob(maxobj());
            m_dumpoutlet = max_jit_obex_dumpout_get(mob);
            return {};
    }};
    
    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        t_class* c = args[0];

        max_jit_class_mop_wrap(c, this_jit_class, 0);
        max_jit_class_wrap_standard(c, this_jit_class, 0);
        class_addmethod(c, (method)mlmat_assist, "assist", A_CANT, 0);
        class_addmethod(c, (method)max_mlmat_jit_matrix, "jit_matrix", A_GIMME, 0);

        return {};
    }};
    
    std::unique_ptr<arma::Mat<double>> m_previous_centroids;
    std::unique_ptr<arma::Row<size_t>> m_previous_assignments;
    bool m_received_centroids = false;
};


MIN_EXTERNAL(mlmat_kmeans);


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_kmeans>* job = (minwrap<mlmat_kmeans>*)(x);
        // call our custom matrix_calc function defined inside the C++ class
        err = job->m_min_object.matrix_calc(x, inputs, outputs);
    }
    return err;
}

void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv)
{
    //found need to call this first or the
    // info for the incoming matrix is incorrect
    max_jit_mop_jit_matrix(x,s,argc,argv);
    //only want second inlet
    if (max_jit_obex_inletnumber_get(x) == 1)
    {

        void *j = nullptr;
               
        j = max_jit_obex_jitob_get(x);
        
        //m_received_centroids = true;
        minwrap<mlmat_kmeans>* job = (minwrap<mlmat_kmeans>*)(j);
        job->m_min_object.set_received_centroids(true);

    }
}

void mlmat_assist(void* x, void* b, long io, long index, char* s) {
    switch(io) {
        case 1:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) query points.");
                    break;
                    
                case 1:
                    sprintf(s, "(matrix) centroids.");
                break;

                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) cluster assignments");
                    break;
                case 1:
                    sprintf(s, "(matrix) cluster centroids");
                    break;
                default:
                    sprintf(s, "dumpout");
                    break;
            }

    }
}





