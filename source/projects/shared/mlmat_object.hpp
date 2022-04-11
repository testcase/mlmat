#pragma once

#include "c74_min.h"
#include "mlmat.hpp"
#include "matrix_conversions.hpp"
using namespace c74;
using namespace c74::min;
using namespace c74::max;
using mlpack::data::ScalingModel;



template<class min_class_type, threadsafe threadsafety = threadsafe::no>
class mlmat_object :  public c74::min::object<min_class_type>, public matrix_operator<> {
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
    
    message<> maxob_setup {this, "maxob_setup",
        MIN_FUNCTION {
            c74::max::t_object* mob = maxob_from_jitob(c74::min::object_base::maxobj());
            m_dumpoutlet = max_jit_obex_dumpout_get(mob);
            return {};
        }};
    
    template<typename matrix_type>
    matrix_type calc_cell(matrix_type input, const matrix_info& info, matrix_coord& position) {
        matrix_type output;
        return output;
    }
    
    
protected:
    
    bool m_mode_changed = true;
    void* m_dumpoutlet { nullptr };
private:
    mlmat_object() {};
    friend min_class_type;
};

template<class min_class_type, class model_type, threadsafe threadsafety = threadsafe::no>
class mlmat_object_autoscale :  public c74::min::object<min_class_type>, public matrix_operator<> {
public:
    using c74::min::object_base::classname;
    using c74::min::object<min_class_type>::cerr;
    
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
    
    attribute<min::symbol> file {this, "file", k_sym__empty,
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
                save_model_file(args, m_model, std::string(classname()));
            } catch (const std::runtime_error& s) {
                cerr << s.what() << endl;
            }
            return {};
        }
    };
    
    message<> read {this, "read",
        MIN_FUNCTION {
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
    
    
    
    void load_model_file(const atoms& args) {
        atoms f{};
        
        if(!args.empty()) {
            f.push_back(args[0]);
        }
        
        path p {f, path::filetype::any};
        
        if(p) {
            try {
                mlpack::data::Load(string(p), classname(), m_model, true);
            } catch (const std::runtime_error& s) {
                std::throw_with_nested(std::runtime_error("Error reading model file to disk."));
            }
        }
    }
    
    
    
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
    mlmat_serializable_model<model_type> m_model;
};


template<class min_class_type, class model_type, threadsafe threadsafety = threadsafe::no>
class mlmat_object_writable :  public c74::min::object<min_class_type>, public matrix_operator<> {
public:
    using c74::min::object_base::classname;
    using c74::min::object<min_class_type>::cerr;
    
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
    
    attribute<min::symbol> file {this, "file", k_sym__empty,
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
                save_model_file(args, m_model, std::string(classname()));
            } catch (const std::runtime_error& s) {
                cerr << s.what() << endl;
            }
            return {};
        }
    };
    
    message<> read {this, "read",
        MIN_FUNCTION {
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
    
    
    
    void load_model_file(const atoms& args) {
        atoms f{};
        
        if(!args.empty()) {
            f.push_back(args[0]);
        }
        
        path p {f, path::filetype::any};
        
        if(p) {
            try {
                mlpack::data::Load(string(p), classname(), m_model, true);
            } catch (const std::runtime_error& s) {
                std::throw_with_nested(std::runtime_error("Error reading model file to disk."));
            }
        }
    }
    
    
    
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
