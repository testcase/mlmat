#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <thread>

#include "c74_min.h"
#include "c74_min_logger.h"
#include <mlpack/prereqs.hpp>
//#include <mlpack/core/util/io.hpp>
#include <mlpack/methods/preprocess/scaling_model.hpp>


// retrieve maxob from jitter obj
c74::max::t_object* maxob_from_jitob(c74::max::t_object* job) {
    c74::max::t_object* mwrap = NULL;
    c74::max::object_obex_lookup(job, c74::max::gensym("maxwrapper"), &mwrap);
    return mwrap;
}


template<typename ModelType>
inline void save_model_file(const c74::min::atoms& args,
    const ModelType& model,
    const std::string& model_description) {
    if (model.model) {
        short path = 0;
        char filename[c74::max::MAX_FILENAME_CHARS] = { 0 };
        char fullpath[c74::max::MAX_PATH_CHARS] = { 0 };
        char native_path[c74::max::MAX_PATH_CHARS] = { 0 };

        if (!args.empty()) {
            std::string name = std::string(args[0]);
            path = c74::max::path_getdefault();
            strncpy_zero(filename, name.c_str(), c74::max::MAX_FILENAME_CHARS);
            c74::max::path_toabsolutesystempath(path, filename, fullpath);
            c74::max::path_nameconform(fullpath, native_path, c74::max::PATH_STYLE_NATIVE, c74::max::PATH_TYPE_PATH);
            try {
                mlpack::Save(std::string(fullpath), model_description, model, true);
            }
            catch (const std::runtime_error& s) {
                std::throw_with_nested(std::runtime_error("Error writing model file to disk."));
            }
        }
        else {
            if (!c74::max::saveas_dialog(filename, &path, NULL)) {
                c74::max::path_toabsolutesystempath(path, filename, fullpath);
                c74::max::path_nameconform(fullpath, native_path, c74::max::PATH_STYLE_NATIVE, c74::max::PATH_TYPE_PATH);
                try {
                    mlpack::Save(std::string(fullpath), model_description, model, true);
                }
                catch (const std::runtime_error& s) {
                    std::throw_with_nested(std::runtime_error("Error writing model file to disk."));
                }
            }
        }
    }
    else {
        throw std::runtime_error("No trained model to save!");
    }
}


#include "matrix_conversions.hpp"

template<class class_type>
class mlmat_serializable_model
{
public:
    std::unique_ptr<class_type> model{ nullptr };
    //bool model_trained = false;
    std::unique_ptr<mlpack::ScalingModel> scaler{ nullptr };
    bool scaler_changed = false;
    long dim0 = 0;//this is lame but cant see how else to do this. only needed for mode 0 with 2d reference matrix
    long dimcount = 1; //the horror
    bool autoscale = false;
    template<typename Archive>

    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(CEREAL_NVP(model));
        ar(CEREAL_NVP(autoscale));
        ar(CEREAL_NVP(scaler));
        ar(CEREAL_NVP(scaler_changed));
        ar(CEREAL_NVP(dim0));
        ar(CEREAL_NVP(dimcount));
    }
};

namespace mlmat {

/// Construct the scaling model for a scaler type name, as the scaler
/// attributes define it (used by scaler_fit and the background
/// training workers).
inline std::unique_ptr<mlpack::ScalingModel> make_scaler(const std::string& type,
                                                         double smin, double smax, double seps)
{
    auto scaler = std::make_unique<mlpack::ScalingModel>(smin, smax, seps);
    if (type == "standard") {
        scaler->ScalerType() = mlpack::ScalingModel::STANDARD_SCALER;
    } else if (type == "min_max") {
        scaler->ScalerType() = mlpack::ScalingModel::MIN_MAX_SCALER;
    } else if (type == "normalization") {
        scaler->ScalerType() = mlpack::ScalingModel::MEAN_NORMALIZATION;
    } else if (type == "abs") {
        scaler->ScalerType() = mlpack::ScalingModel::MAX_ABS_SCALER;
    } else if (type == "pca_whitening") {
        scaler->ScalerType() = mlpack::ScalingModel::PCA_WHITENING;
    } else if (type == "zca_whitening") {
        scaler->ScalerType() = mlpack::ScalingModel::ZCA_WHITENING;
    }
    return scaler;
}

} // namespace mlmat

// ---------------------------------------------------------------------------
// background training support
//
// Objects with potentially slow training adopt this through their base
// class (mlmat_object / mlmat_object_autoscale / mlmat_object_writable).
// An object opts in by:
//   * gating its train message on a @background attribute (default on)
//   * calling background_launch(lambda) with a lambda that runs on the
//     worker thread, touches only its own copies, and returns a result
//     (any movable type; it may throw)
//   * providing a public bg_commit(result) member, called on the main
//     thread when training finishes, which installs the result (model
//     swap, object-specific dumpout output such as likelihood, ...)
// The machinery provides the worker thread, the marshalling to the
// main thread (via min's queue), the in-progress guard, done and error
// messages out the dumpout, and joining on destruction.
//
// The dumpout outlet pointer lives here so the machinery can post to
// it; the base classes' maxob_setup assigns it as before.
// ---------------------------------------------------------------------------

template <class min_class_type>
class mlmat_background_trainable {
public:
    /// Launch work on a background thread. Returns false (and reports
    /// to the console) if a training is already in progress.
    template <class WorkerFn>
    bool background_launch(WorkerFn&& work) {
        if (m_bg_active) {
            bg_notify_busy();
            return false;
        }
        m_bg_active = true;
        m_bg_worker = std::thread([this, work = std::forward<WorkerFn>(work)]() mutable {
            using R = std::decay_t<decltype(work())>;
            // shared_ptr: the handoff into std::function must be copyable
            auto result = std::make_shared<R>();
            bool ok = true;
            std::string error;
            try {
                *result = work();
            } catch (const std::exception& e) {
                ok = false;
                error = e.what();
            } catch (...) {
                ok = false;
                error = "unknown exception";
            }
            {
                std::lock_guard<std::mutex> lock(m_bg_mutex);
                m_bg_commit = [this, ok, error = std::move(error),
                               result = std::move(result)]() mutable {
                    if (ok) {
                        static_cast<min_class_type*>(this)->bg_commit(std::move(*result));
                        post_bg_dumpout("done");
                    } else {
                        bg_notify_failure(error);
                    }
                };
            }
            // thread-safe: fires the queue on the main thread
            m_bg_queue.set();
        });
        return true;
    }

    /// Whether a background training is in flight.
    bool background_in_progress() const {
        return m_bg_active;
    }

    /// Post a message out the dumpout outlet. The machinery posts done
    /// and error; bg_commit implementations can use it for
    /// object-specific output (e.g. gmm's likelihood).
    void post_bg_dumpout(const char* name, long argc = 0, c74::max::t_atom* argv = nullptr) {
        if (m_dumpoutlet) {
            outlet_anything(static_cast<c74::max::t_object*>(m_dumpoutlet),
                            c74::max::gensym(name), argc, argv);
        }
    }

protected:
    /// Called (main thread) when a launch is rejected because a
    /// training is in progress; the base classes report to the console.
    virtual void bg_notify_busy() = 0;
    /// Called (main thread) when training failed; the base classes
    /// report to the console and post error out the dumpout.
    virtual void bg_notify_failure(const std::string& error) = 0;

    void* m_dumpoutlet { nullptr };
    std::atomic<bool> m_bg_active {false};

    ~mlmat_background_trainable() {
        // join before the queue member is destroyed: if training is
        // still running this blocks until it completes
        if (m_bg_worker.joinable()) {
            m_bg_worker.join();
        }
    }

private:
    // fires on the main thread when the worker has finished
    c74::min::queue<> m_bg_queue { static_cast<min_class_type*>(this),
        MIN_FUNCTION {
            std::function<void()> commit;
            {
                std::lock_guard<std::mutex> lock(m_bg_mutex);
                commit = std::move(m_bg_commit);
            }
            if (m_bg_worker.joinable()) {
                m_bg_worker.join();
            }
            m_bg_active = false;
            if (commit) {
                commit();
            }
            return {};
        }
    };
    std::mutex m_bg_mutex;
    std::function<void()> m_bg_commit;
    std::thread m_bg_worker;
};

template<class min_class_type, c74::min::threadsafe threadsafety = c74::min::threadsafe::no>
class mlmat_object :  public c74::min::object<min_class_type>, public c74::min::matrix_operator<>, public mlmat_background_trainable<min_class_type> {
public:
    c74::min::attribute<int, c74::min::threadsafe::no, c74::min::limit::clamp, c74::min::allow_repetitions::no> mode { this, "mode", 0,
        c74::min::range {0,2},
        c74::min::description {
            "Input mode for matrices. "
        },
        c74::min::setter { MIN_FUNCTION {
            m_mode_changed = true;
            return args;
        }}
    };
    
    c74::min::message<> maxob_setup {this, "maxob_setup",
        MIN_FUNCTION {
            c74::max::t_object* mob = maxob_from_jitob(c74::min::object_base::maxobj());
            this->m_dumpoutlet = max_jit_obex_dumpout_get(mob);
            return {};
        }};
        // mlmat_background_trainable hooks: console + dumpout reporting
    void bg_notify_busy() override {
        using namespace c74::min;
        (this->cerr << "a training is already in progress." << endl);
    }

    void bg_notify_failure(const std::string& error) override {
        using namespace c74::min;
        (this->cerr << "background training failed: " << error << endl);
        if (this->m_dumpoutlet) {
            outlet_anything(static_cast<c74::max::t_object*>(this->m_dumpoutlet),
                            c74::max::gensym("error"), 0, nullptr);
        }
    }

    template<typename matrix_type>
    matrix_type calc_cell(matrix_type input, const c74::min::matrix_info& info, c74::min::matrix_coord& position) {
        matrix_type output;
        return output;
    }
    
    
protected:
    
    bool m_mode_changed = true;
private:
    mlmat_object() {};
    friend min_class_type;
};

template<class min_class_type, class model_type, c74::min::threadsafe threadsafety = c74::min::threadsafe::no>
class mlmat_object_autoscale :  public c74::min::object<min_class_type>, public c74::min::matrix_operator<>, public mlmat_background_trainable<min_class_type> {
public:
    using c74::min::object_base::classname;
    using c74::min::object<min_class_type>::cerr;
    
    c74::min::attribute<int, c74::min::threadsafe::no, c74::min::limit::clamp, c74::min::allow_repetitions::no> mode { this, "mode", 0,
        c74::min::range {0,2},
        c74::min::description {
            "Input mode for matrices. "
        },
        c74::min::setter { MIN_FUNCTION {
            m_mode_changed = true;
            return args;
        }}
    };
    
    //this will likely be moved to mlmat_operator
    c74::min::attribute<bool> autoscale { this, "autoscale", false,
        c74::min::description {
            "Automatically scale data based on scaling_type."
        },
        c74::min::setter { MIN_FUNCTION {
            // changing the scaling configuration invalidates the fitted
            // scaler, so the reference/training set must be resubmitted
            m_mode_changed = true;
            return args;
        }}
    };
    
    c74::min::attribute<c74::min::symbol> scaler { this, "scaler", "normalization",
        c74::min::description {
            "The scaler type."
        },
        c74::min::setter { MIN_FUNCTION {
            m_scaler_changed = true;
            return args;
        }},
        c74::min::range {"standard","min_max", "normalization", "abs", "pca_whitening", "zca_whitening"}
    };
    
    c74::min::attribute<int> scaler_min { this, "scaler_min", 0,
        c74::min::description {
            "Minimum value. "
            "Minimum value. "
        }
    };
    
    c74::min::attribute<int> scaler_max { this, "scaler_max", 1,
        c74::min::description {
            "Maximum value. "
            "Maximum value. "
        }
    };
    
    c74::min::attribute<double> scaler_epsilon { this, "scaler_epsilon", .00005,
        c74::min::description {
            "Epsilon value. "
            "Epsilon value. "
        }
    };
    
    c74::min::attribute<c74::min::symbol> file {this, "file", c74::min::k_sym__empty,
        c74::min::description {
            "File"
        },
        c74::min::title {
            "File"
        },
        c74::min::setter { MIN_FUNCTION {
            if(args[0] != c74::min::k_sym__empty) {
                load_model_file(args);
            }
            return args;
        }}
    };
    
    c74::min::message<> write {this, "write",
        MIN_FUNCTION {
            using namespace c74::min;
            try {
                m_model.autoscale = autoscale;
                save_model_file(args, m_model, std::string(classname()));
            } catch (const std::runtime_error& s) {
                cerr << s.what() << endl;
            }
            return {};
        }
    };
    
    c74::min::message<> read {this, "read",
        MIN_FUNCTION {
            using namespace c74::min;
            try {
                load_model_file(args);
                autoscale = m_model.autoscale;
                m_mode_changed = false;
            } catch (const std::runtime_error& s) {
                cerr << s.what() << endl;
            }
            return {};
        }
    };
    
    
    
    void load_model_file(const c74::min::atoms& args) {
        c74::min::atoms f{};
        
        if(!args.empty()) {
            f.push_back(args[0]);
        }
        
        c74::min::path p {f, c74::min::path::filetype::any};
        
        if(p) {
            try {
                mlpack::Load(std::string(p), classname(), m_model, true);
            } catch (const std::runtime_error& s) {
                std::throw_with_nested(std::runtime_error("Error reading model file from disk."));
            }
        }
    }
    
    
    
    template<typename matrix_type>
    matrix_type calc_cell(matrix_type input, const c74::min::matrix_info& info, c74::min::matrix_coord& position) {
        matrix_type output;
        return output;
    }
    
    
    template<typename ModelType>
    void scaler_fit(ModelType& s, arma::Mat<double>& arma_matrix) {
        const std::string scalertype_string = scaler.get().c_str();
        
        if(autoscale) {
            s.scaler = mlmat::make_scaler(scalertype_string, scaler_min, scaler_max, scaler_epsilon);
            s.scaler->Fit(arma_matrix);
        }
    }
    
    template<typename ModelType>
    arma::mat& scaler_transform(ModelType& s, arma::Mat<double>& input, arma::Mat<double> &output) {
        if(autoscale) {
            if(!s.scaler) {
                // no scaler was fitted (e.g. autoscale toggled on after
                // training); pass the data through rather than dereferencing
                // null
                cerr << "autoscale is on but no scaler was fit. resubmit the training/reference data.\n";
                return input;
            }
            try {
                s.scaler->Transform(input, output);
            } catch (const std::runtime_error& s) {
                std::throw_with_nested(std::runtime_error("Fit must be called before transform"));
            }
            return output;
        } else {
            return input;
        }
    }
    
    template<typename ModelType>
    arma::mat& scaler_inverse_transform(ModelType& s, arma::Mat<double>& input, arma::Mat<double> &output) {
        if(autoscale) {
            if(!s.scaler) {
                cerr << "autoscale is on but no scaler was fit. resubmit the training/reference data.\n";
                return input;
            }
            s.scaler->InverseTransform(input, output);
            return output;
        } else {
            return input;
        }
    }
    
    
    c74::min::message<> maxob_setup {this, "maxob_setup",
        MIN_FUNCTION {
            c74::max::t_object* mob = maxob_from_jitob(c74::min::object_base::maxobj());
            this->m_dumpoutlet = max_jit_obex_dumpout_get(mob);
            return {};
    }};
        // mlmat_background_trainable hooks: console + dumpout reporting
    void bg_notify_busy() override {
        using namespace c74::min;
        (this->cerr << "a training is already in progress." << endl);
    }

    void bg_notify_failure(const std::string& error) override {
        using namespace c74::min;
        (this->cerr << "background training failed: " << error << endl);
        if (this->m_dumpoutlet) {
            outlet_anything(static_cast<c74::max::t_object*>(this->m_dumpoutlet),
                            c74::max::gensym("error"), 0, nullptr);
        }
    }

protected:
    bool m_mode_changed = true;
    bool m_scaler_changed = true;
    mlmat_serializable_model<model_type> m_model;
};


template<class min_class_type, class model_type, c74::min::threadsafe threadsafety = c74::min::threadsafe::no>
class mlmat_object_writable :  public c74::min::object<min_class_type>, public c74::min::matrix_operator<>, public mlmat_background_trainable<min_class_type> {
public:
    using c74::min::object_base::classname;
    using c74::min::object<min_class_type>::cerr;
    
    c74::min::attribute<int, c74::min::threadsafe::no, c74::min::limit::clamp, c74::min::allow_repetitions::no> mode { this, "mode", 0,
        c74::min::range {0,2},
        c74::min::description {
            "Input mode for matrices. "
        },
        c74::min::setter { MIN_FUNCTION {
            m_mode_changed = true;
            return args;
        }}
    };
    
    //this will likely be moved to mlmat_operator
    c74::min::attribute<bool> autoscale { this, "autoscale", false,
        c74::min::description {
            "Automatically scale data based on scaling_type."
        },
        c74::min::setter { MIN_FUNCTION {
            // changing the scaling configuration invalidates the fitted
            // scaler, so the reference/training set must be resubmitted
            m_mode_changed = true;
            return args;
        }}
    };
    
    c74::min::attribute<c74::min::symbol> scaler { this, "scaler", "normalization",
        c74::min::description {
            "The scaler type."
        },
        c74::min::setter { MIN_FUNCTION {
            m_scaler_changed = true;
            return args;
        }},
        c74::min::range {"standard","min_max", "normalization", "abs", "pca_whitening", "zca_whitening"}
    };
    
    c74::min::attribute<int> scaler_min { this, "scaler_min", 0,
        c74::min::description {
            "Minimum value. "
            "Minimum value. "
        }
    };
    
    c74::min::attribute<int> scaler_max { this, "scaler_max", 1,
        c74::min::description {
            "Maximum value. "
            "Maximum value. "
        }
    };
    
    c74::min::attribute<double> scaler_epsilon { this, "scaler_epsilon", .00005,
        c74::min::description {
            "Epsilon value. "
            "Epsilon value. "
        }
    };
    
    c74::min::attribute<c74::min::symbol> file {this, "file", c74::min::k_sym__empty,
        c74::min::description {
            "File"
        },
        c74::min::title {
            "File"
        },
        c74::min::setter { MIN_FUNCTION {
            if(args[0] != c74::min::k_sym__empty) {
                load_model_file(args);
            }
            return args;
        }}
    };
    
    c74::min::message<> write {this, "write",
        MIN_FUNCTION {
            using namespace c74::min;
            try {
                m_model.autoscale = autoscale;
                c74::min::symbol c = classname();
                const char* buf = (const char*)c;
                const std::string s = std::string(buf);
                save_model_file(args, m_model, s);
            } catch (const std::runtime_error& s) {
                cerr << s.what() << endl;
            }
            return {};
        }
    };
    
    c74::min::message<> read {this, "read",
        MIN_FUNCTION {
            using namespace c74::min;
            try {
                load_model_file(args);
                autoscale = m_model.autoscale;
                m_mode_changed = false;
            } catch (const std::runtime_error& s) {
                cerr << s.what() << endl;
            }
            return {};
        }
    };
    
    
    
    void load_model_file(const c74::min::atoms& args) {
        c74::min::atoms f{};
        
        if(!args.empty()) {
            f.push_back(args[0]);
        }
        
        c74::min::path p {f, c74::min::path::filetype::any};
        
        if(p) {
            try {
                mlpack::Load(std::string(p), classname(), m_model, true);
            } catch (const std::runtime_error& s) {
                std::throw_with_nested(std::runtime_error("Error reading model file from disk."));
            }
        }
    }
    
    
    
    template<typename matrix_type>
    matrix_type calc_cell(matrix_type input, const c74::min::matrix_info& info, c74::min::matrix_coord& position) {
        matrix_type output;
        return output;
    }
    
    
    template<typename ModelType>
    void scaler_fit(ModelType& s, arma::Mat<double>& arma_matrix) {
        const std::string scalertype_string = scaler.get().c_str();
        
        if(autoscale) {
            s.scaler = mlmat::make_scaler(scalertype_string, scaler_min, scaler_max, scaler_epsilon);
            s.scaler->Fit(arma_matrix);
        }
    }
    
    template<typename ModelType>
    arma::mat& scaler_transform(ModelType& s, arma::Mat<double>& input, arma::Mat<double> &output) {
        if(autoscale) {
            if(!s.scaler) {
                // no scaler was fitted (e.g. autoscale toggled on after
                // training); pass the data through rather than dereferencing
                // null
                cerr << "autoscale is on but no scaler was fit. resubmit the training/reference data.\n";
                return input;
            }
            try {
                s.scaler->Transform(input, output);
            } catch (const std::runtime_error& s) {
                std::throw_with_nested(std::runtime_error("Fit must be called before transform"));
            }
            return output;
        } else {
            return input;
        }
    }
    
    template<typename ModelType>
    arma::mat& scaler_inverse_transform(ModelType& s, arma::Mat<double>& input, arma::Mat<double> &output) {
        if(autoscale) {
            if(!s.scaler) {
                cerr << "autoscale is on but no scaler was fit. resubmit the training/reference data.\n";
                return input;
            }
            s.scaler->InverseTransform(input, output);
            return output;
        } else {
            return input;
        }
    }
    
    c74::min::message<> maxob_setup {this, "maxob_setup",
        MIN_FUNCTION {
            c74::max::t_object* mob = maxob_from_jitob(c74::min::object_base::maxobj());
            this->m_dumpoutlet = max_jit_obex_dumpout_get(mob);
            return {};
    }};
        // mlmat_background_trainable hooks: console + dumpout reporting
    void bg_notify_busy() override {
        using namespace c74::min;
        (this->cerr << "a training is already in progress." << endl);
    }

    void bg_notify_failure(const std::string& error) override {
        using namespace c74::min;
        (this->cerr << "background training failed: " << error << endl);
        if (this->m_dumpoutlet) {
            outlet_anything(static_cast<c74::max::t_object*>(this->m_dumpoutlet),
                            c74::max::gensym("error"), 0, nullptr);
        }
    }

    
protected:
    bool m_mode_changed = true;
    bool m_scaler_changed = true;
    mlmat_serializable_model<model_type> m_model;
};



//class MyPrintLoss
//{
//public:
//    MyPrintLoss(void* outlet) : outlet(outlet)
//    {}
//
//    template<typename OptimizerType, typename FunctionType, typename MatType>
//    void EndEpoch(OptimizerType& /* optimizer */,
//                  FunctionType& /* function */,
//                  const MatType& /* coordinates */,
//                  const size_t /* epoch */,
//                  const double objective)
//    {
//
//        t_atom a[1];
//
//        atom_setfloat(a,objective);
//        outlet_anything(outlet, gensym("loss"), 1, a);
//
//    }
//private:
//
//    void* outlet;
//
//
//};
