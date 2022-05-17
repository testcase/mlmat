/// @file mlmat.scaling.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// TODO: Use scaling_model.hpp for serialization
/// TODO: Need to fix up error messagesß

#include "c74_min.h"

#include "mlmat_object.hpp"
#include <mlpack/methods/preprocess/scaling_model.hpp>


using namespace c74;
using namespace c74::min;
using namespace c74::max;
using namespace mlpack::data;

void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);
void max_jit_mlmat_mproc(max_jit_wrapper *x, void *mop);

class mlmat_scaling : public mlmat_object<mlmat_scaling> {
public:
    MIN_DESCRIPTION	{"Apply various scaling strategies. This will apply different type of scaling to datasets. Further data can be scaled using the fitting of original dataset. Inverse transform can also be applied."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.lookup"};

    attribute<min::symbol> scaler { this, "scaler", "standard",
        description {
            "The scaler type."
        },
        setter { MIN_FUNCTION {
            m_scaler_changed = true;
            return args;
        }},
        range {"standard", "min_max", "normalization", "abs", "pca_whitening", "zca_whitening"}
    };
    
    attribute<int> min { this, "min", 0,
        description {
            "Minimum value when using the min_max scaler."
        }
    };
    
    attribute<int> max { this, "max", 1,
        description {
            "Maximum value when using the min_max scaler."
        }
    };
    
    attribute<double> epsilon { this, "epsilon", .00005,
           description {
               "Regularization parameter for pca_whitening and zca_whitening scalers."
        }
    };
    
    attribute<bool> inverse { this, "inverse", false,
           description {
               "Apply inverse transform to matrix."
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
                load_file(args);
            }
            return args;
        }}
    };
    
    
    message<> write {this, "write",
        MIN_FUNCTION {
            save_file(args);
            return {};
        }
    };

    void save_file(const atoms& args) {
        if(m_model && m_model_trained) {
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
               // std::cout << native_path << std::endl;
                try {
                    mlpack::data::Save(string(fullpath), "scaling_model", *m_model, true);
                } catch (const std::runtime_error& s) {
                    (cerr  <<  "error writing model file."  << endl);
                }

            } else {

                if(!saveas_dialog(filename,&path, NULL)) {
                    path_toabsolutesystempath(path, filename, fullpath);
                    path_nameconform(fullpath, native_path, max::PATH_STYLE_NATIVE, max::PATH_TYPE_PATH);
                    //std::cout << native_path << std::endl;
                    try {
                        mlpack::data::Save(string(fullpath), "scaling_model", *m_model, true);
                    } catch (const std::runtime_error& s) {
                        (cerr << "error writing model file." << endl);
                    }
                }
            }
      } else {
            (cerr << "No trained model to save." << endl);
        }
    }
    
    message<> read {this, "read",
        MIN_FUNCTION {
            load_file(args);
            return {};
        }
    };
                
    void load_file(const atoms& args) {
       atoms f{};

        if(!args.empty()) {
            f.push_back(args[0]);
        }
        
        path p {f, path::filetype::any};
    
        try {
            ScalingModel *tmp = new ScalingModel();
            mlpack::data::Load(string(p), "scaling_model", *tmp, true);
            m_model.reset( tmp );
            m_model_trained = true;
        } catch (const std::runtime_error& s) {
            (cerr << "error reading model file." << endl);
        }
    }
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_matrix_info;
        arma::Mat<double> dat;
        arma::Mat<double> output;
        auto in_matrix = (t_object*)object_method(inputs, _jit_sym_getindex, 0);
        auto out_matrix = (t_object*)object_method(outputs, _jit_sym_getindex, 0);
        auto in_matrix_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_matrix_savelock = object_method(out_matrix, _jit_sym_lock, 1);
        
        object_method(in_matrix, _jit_sym_getinfo, &in_matrix_info);
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_matrix_info);
        
        if(!m_model) {
            (cerr << "no reference data input for scaling" << endl);
            goto out;
        }
        
        if(m_scaler_changed) {
            (cerr << "scaler mode changed, need new reference set" << endl);
            goto out;
        }
        
        try {
            check_mode(in_matrix_info, mode, "scaling");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
         
        dat = jit_to_arma(mode, in_matrix64, dat);
        
        if(inverse) {
            m_model->InverseTransform(dat, output);
        } else {
            m_model->Transform(dat, output);
        }
         
        out_matrix = arma_to_jit(mode, output, out_matrix, in_matrix_info);
         
    out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        object_method(in_matrix,_jit_sym_lock,in_matrix_savelock);
        object_method(out_matrix,_jit_sym_lock,out_matrix_savelock);
        return err;
        
    }
    
    t_jit_err process_reference_set_matrix(t_object *matrix) {
        t_jit_matrix_info minfo;
        t_jit_err err = JIT_ERR_NONE;
        t_atom a;
        arma::mat dat;
        arma::mat out_data;
        void *o,*p;
        const string scalertype_string = scaler.get().c_str();
        auto* mob = maxob_from_jitob(maxobj());
        auto *mop = max_jit_obex_adornment_get(mob , _jit_sym_jit_mop);
        auto * op =  object_method(mop,_jit_sym_getoutputlist);
        auto genmatrix = linklist_getindex(static_cast<t_linklist*>(op), 1);
        
        auto genmatrix_savelock = object_method(genmatrix, _jit_sym_lock, 1);

        long savelock = (long) object_method(matrix, _jit_sym_lock, 1);
        
        object_method(matrix, _jit_sym_getinfo, &minfo);
               
        if(minfo.dimcount > 2) {
            cout << "expecting 1d or 2d matrix, received " << minfo.dimcount << "d matrix" << endl;
            err = JIT_ERR_INVALID_INPUT;
            goto out;
        }
        
        try {
            check_mode(minfo, mode, "scaling");
        } catch (std::invalid_argument& s) {
            cerr << s.what() << endl;
            goto out;
        }
        
           
        dat = jit_to_arma(mode, matrix, dat);
               
        m_model = std::make_unique<ScalingModel>(min, max, epsilon);
    
        if(scalertype_string == "standard") {
            m_model->ScalerType() = ScalingModel::STANDARD_SCALER;
        } else if(scalertype_string == "min_max") {
           m_model->ScalerType() = ScalingModel::MIN_MAX_SCALER;
        } else if(scalertype_string == "normalization") {
            m_model->ScalerType() = ScalingModel::MEAN_NORMALIZATION;
        } else if(scalertype_string == "abs") {
            m_model->ScalerType() = ScalingModel::MAX_ABS_SCALER;
        } else if(scalertype_string == "pca_whitening") {
            m_model->ScalerType() = ScalingModel::PCA_WHITENING;
        } else if(scalertype_string == "zca_whitening") {
            m_model->ScalerType() = ScalingModel::PCA_WHITENING;
        } else {
            (cerr << "scaler attribute not valid" << endl);
            m_model.reset(nullptr);
            goto out;
        }
        
        m_model->Fit(dat);
        m_model->Transform(dat, out_data);
        
        m_model_trained = true;
        
        genmatrix = arma_to_jit(mode, out_data, static_cast<t_object*>(genmatrix), minfo);
        
        if ((p=object_method(mop,_jit_sym_getoutput,2)) && (o=max_jit_mop_io_getoutlet(p)))
        {
            atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
            outlet_anything(o,_jit_sym_jit_matrix,1,&a);
        }
        m_scaler_changed = false;
        m_mode_changed = false;
    out:
        object_method(matrix, _jit_sym_lock, savelock);
        object_method(genmatrix,_jit_sym_lock,genmatrix_savelock);
        return err;
    }

private:
    // override jitclass_setup so we can have our own matrix_calc. jitclass_setup is called first (and only once when the object is loaded for the first time) during the intitialization of the object.
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        // add mop
        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 2, 2));
        
        // force type
        jit_mop_single_type(mop, _jit_sym_float64);
        jit_mop_input_nolink(mop, 2);
        jit_mop_output_nolink(mop, 1);
        jit_mop_output_nolink(mop, 2);
  
        auto input2 = jit_object_method(mop,_jit_sym_getinput,2);
           
        //always adapt
        jit_object_method(input2,_jit_sym_ioproc,jit_mop_ioproc_copy_adapt);
               
        jit_class_addadornment(c, mop);
               
        // add our custom matrix_calc method
        jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
                
        return {};
    }};

    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        
        max_jit_class_mop_wrap(c, this_jit_class, 0);
        max_jit_class_wrap_standard(c, this_jit_class, 0);
        max_jit_classex_mop_mproc(c,this_jit_class,(void*)max_jit_mlmat_mproc);
        class_addmethod(c, (method)mlmat_assist, "assist", A_CANT, 0);
        class_addmethod(c, (method)max_mlmat_jit_matrix, "jit_matrix", A_GIMME, 0);
        return {};
    }};
    
    std::unique_ptr<ScalingModel> m_model { nullptr };
    bool m_model_trained = false;
    bool m_scaler_changed = false;
};


MIN_EXTERNAL(mlmat_scaling);

void max_jit_mlmat_mproc(max_jit_wrapper *x, void *mop)
{
    t_jit_err err =JIT_ERR_NONE;
    void *o,*p, *j;
    t_atom a;
    long outputmode = max_jit_mop_getoutputmode(x);

    if (outputmode==1) {

        // send in link list that includes only output 1
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


void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv)
{
    //found need to call this first or the
    // info for the incoming matrix is incorrect
    max_jit_mop_jit_matrix(x,s,argc,argv);
    //only want second inlet
    if (max_jit_obex_inletnumber_get(x) == 1)
    {
        t_jit_err err = JIT_ERR_NONE;
        void *j = nullptr;
        
        void *p, *m, *mop;


        if (!(mop=max_jit_obex_adornment_get(x,_jit_sym_jit_mop)))
            jit_error_code(x,err);

        p = object_method(mop,_jit_sym_getinput,2);
        m = object_method(p,_jit_sym_getmatrix);
        j = max_jit_obex_jitob_get(x);
    
        minwrap<mlmat_scaling>* job = (minwrap<mlmat_scaling>*)(j);
        err = job->m_min_object.process_reference_set_matrix(static_cast<t_object*>(m));

        jit_error_code(x,err);
    }
}

t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_scaling>* job = (minwrap<mlmat_scaling>*)(x);
        err = job->m_min_object.matrix_calc(x, inputs, (t_object*)object_method(outputs, _jit_sym_getindex, 0));
    }

    return err;
}


void mlmat_assist(void* x, void* b, long io, long index, char* s) {
    switch(io) {
        case 1:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) data");
                    break;

                case 1:
                    sprintf(s, "(matrix) reference dataset");
                    break;

                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) scaled data");
                    break;

                case 1:
                    sprintf(s, "(matrix) scaled reference set");
                    break;

                default:
                    sprintf(s, "dumpout");
                    break;
            }

    }
}



