#pragma once

#include "c74_min.h"
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/methods/preprocess/scaling_model.hpp>
#include "matrix_conversions.hpp"
using namespace c74;
using namespace c74::min;
using namespace c74::max;
using mlpack::data::ScalingModel;

template<typename ModelType>
inline void save_model_file(const atoms& args,
                            const ModelType& model,
                            const std::string& model_description) {
    if(model.model) {
        short path = 0;
        char filename[max::MAX_FILENAME_CHARS] = {0};
        char fullpath[max::MAX_PATH_CHARS] = {0};
        char native_path[max::MAX_PATH_CHARS]={0};
        
        if(!args.empty()) {
            string name = string(args[0]);
            path = max::path_getdefault();
            strncpy_zero(filename, name.c_str(), max::MAX_FILENAME_CHARS);
            max::path_toabsolutesystempath(path, filename, fullpath);
            max::path_nameconform(fullpath, native_path, max::PATH_STYLE_NATIVE, max::PATH_TYPE_PATH);
            try {
                mlpack::data::Save(string(fullpath), "knn_model", model, true);
            } catch (const std::runtime_error& s) {
                std::throw_with_nested(std::runtime_error("Error writing model file to disk."));
            }
        } else {
            if(!saveas_dialog(filename,&path, NULL)) {
                path_toabsolutesystempath(path, filename, fullpath);
                path_nameconform(fullpath, native_path, max::PATH_STYLE_NATIVE, max::PATH_TYPE_PATH);
                try {
                    mlpack::data::Save(string(fullpath), model_description, model, true);
                } catch (const std::runtime_error& s) {
                    std::throw_with_nested(std::runtime_error("Error writing model file to disk."));
                }
            }
        }
    } else {
        throw std::runtime_error("No trained model to save!");
    }
}


inline void check_mode(const t_jit_matrix_info& minfo,
                       const int mode,
                       const std::string& callerDescription,
                       const std::string& addInfo = "matrix")
{
    if(mode > 0) {
        if(minfo.planecount > 1) {
            std::ostringstream oss;
            oss << callerDescription << ": number of planes (" << minfo.planecount << ") in matrix "
            << "is not valid in mode "  << mode << ". Expecting 1 plane!";
            throw std::invalid_argument(oss.str());
        }
    }
}


t_object* maxob_from_jitob(c74::max::t_object* job) {
    c74::max::t_object* mwrap = NULL;
    object_obex_lookup(job, gensym("maxwrapper"), &mwrap);
    return mwrap;
}

class MyPrintLoss
{
public:
    MyPrintLoss(void* outlet) : outlet(outlet)
    {}
    
    template<typename OptimizerType, typename FunctionType, typename MatType>
    void EndEpoch(OptimizerType& /* optimizer */,
                  FunctionType& /* function */,
                  const MatType& /* coordinates */,
                  const size_t /* epoch */,
                  const double objective)
    {
        
        t_atom a[1];
        
        atom_setfloat(a,objective);
        outlet_anything(outlet, gensym("loss"), 1, a);
        
    }
private:
    
    void* outlet;
    
    
};

template<class class_type>
class mlmat_serializable_model
{
 public:
    std::unique_ptr<class_type> model { nullptr };
    //bool model_trained = false;
    std::unique_ptr<ScalingModel> scaler { nullptr };
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
        ar(CEREAL_NVP(scaler));
        ar(CEREAL_NVP(scaler_changed));
        ar(CEREAL_NVP(dim0));
        ar(CEREAL_NVP(dimcount));
    }
};

template<class min_class_type, threadsafe threadsafety = threadsafe::no>
class mlmat_operator :  public c74::min::object<min_class_type>, public matrix_operator<> {
public:
    attribute<int, threadsafe::no, limit::clamp, allow_repetitions::no> mode { this, "mode", 0,
        range {0,2},
        description {
            "Input mode for matrices. "
        },
        setter { MIN_FUNCTION {
            m_mode_changed = true;
            return args;
        }}
    };
    
    
    template<typename matrix_type>
    matrix_type calc_cell(matrix_type input, const matrix_info& info, matrix_coord& position) {
        matrix_type output;
        return output;
    }
    

protected:
    bool m_mode_changed = true;
    void* m_dumpoutlet { nullptr };
};


template<class min_class_type, threadsafe threadsafety = threadsafe::no>
class mlmat_operator_autoscale :  public c74::min::object<min_class_type>, public matrix_operator<> {
public:
    attribute<int, threadsafe::no, limit::clamp, allow_repetitions::no> mode { this, "mode", 0,
        range {0,2},
        description {
            "Input mode for matrices. "
        },
        setter { MIN_FUNCTION {
            m_mode_changed = true;
            return args;
        }}
    };
    
    //this will likely be moved to mlmat_operator
    attribute<bool> autoscale { this, "autoscale", false,
        description {
            "Automatically scale data based on scaling_type."
        }
    };
    
    attribute<min::symbol> scaler { this, "scaler", "normalization",
        description {
            "The scaler type."
        },
        setter { MIN_FUNCTION {
            m_scaler_changed = true;
            return args;
        }},
        range {"standard","min_max", "normalization", "abs", "pca_whitening", "zca_whitening"}
    };
    
    attribute<int> scaler_min { this, "scaler_min", 0,
        description {
            "Minimum value. "
            "Minimum value. "
        }
    };
    
    attribute<int> scaler_max { this, "scaler_max", 1,
        description {
            "Maximum value. "
            "Maximum value. "
        }
    };
    
    attribute<double> scaler_epsilon { this, "scaler_epsilon", .00005,
        description {
            "Epsilon value. "
            "Epsilon value. "
        }
    };
    
    template<typename matrix_type>
    matrix_type calc_cell(matrix_type input, const matrix_info& info, matrix_coord& position) {
        matrix_type output;
        return output;
    }
    

    template<typename ModelType>
    void scaler_fit(ModelType& s, arma::Mat<double>& arma_matrix) {
        const string scalertype_string = scaler.get().c_str();
        
        if(autoscale) {
            s.scaler = std::make_unique<ScalingModel>(scaler_min, scaler_max, scaler_epsilon);
            if(scalertype_string == "standard") {
                s.scaler->ScalerType() = ScalingModel::STANDARD_SCALER;
            } else if(scalertype_string == "min_max") {
                s.scaler->ScalerType() = ScalingModel::MIN_MAX_SCALER;
            } else if(scalertype_string == "normalization") {
                s.scaler->ScalerType() = ScalingModel::MEAN_NORMALIZATION;
            } else if(scalertype_string == "abs") {
                s.scaler->ScalerType() = ScalingModel::MAX_ABS_SCALER;
            } else if(scalertype_string == "pca_whitening") {
                s.scaler->ScalerType() = ScalingModel::PCA_WHITENING;
            } else if(scalertype_string == "zca_whitening") {
                s.scaler->ScalerType() = ScalingModel::PCA_WHITENING;
            }
            s.scaler->Fit(arma_matrix);
        }
    }
    
    template<typename ModelType>
    arma::mat& scaler_transform(ModelType& s, arma::Mat<double>& input, arma::Mat<double> &output) {
        if(autoscale) {
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
            s.scaler->InverseTransform(input, output);
            return output;
        } else {
            return input;
        }
    }

protected:
    bool m_mode_changed = true;
    bool m_scaler_changed = true;
    void* m_dumpoutlet { nullptr };
};



