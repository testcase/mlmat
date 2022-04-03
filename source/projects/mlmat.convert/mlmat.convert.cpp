/// @file mlmat.convert.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "mlmat_operator.hpp"

using namespace c74;
using namespace c74::min;
using namespace c74::max;

void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);


class mlmat_convert : public mlmat_operator<mlmat_convert> {
public:
    MIN_DESCRIPTION	{"Convert matrices between mlmat modes. Converts a matrix to between different mlmat modes. <at>input</at> specifies the expected input mode and <at>output</at> indicates the mode to convert to. Note that this will fail if attempting to convert from mode 1 or 2 to mode 0 if the resulting matrix would require more than 32 planes."};
    MIN_TAGS		{"ml"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.scaling, mlmat.lookup"};

    attribute<int, threadsafe::no, limit::clamp, allow_repetitions::no> input_mode { this, "input", 0,
        range {0,2},
        description {
            "Input mode for matrices. "
        }
    };
    
    attribute<int, threadsafe::no, limit::clamp, allow_repetitions::no> output_mode { this, "output", 0,
        range {0,2},
        description {
            "Output mode for matrices. "
        }
    };
    
    mlmat_convert() {
        static c74::max::t_object *scanwrap=NULL;
        if (!scanwrap) {
            if((scanwrap = (c74::max::t_object*)c74::max::newinstance(c74::max::gensym("jit.scanwrap"),0,NULL)))
                c74::max::freeobject(scanwrap);
        }
        
        m_jit_scanwrap = (t_object*)jit_object_new(max::gensym("jit_scanwrap"));
        
        static c74::max::t_object *coerce=NULL;
        if (!coerce) {
            if((coerce = (c74::max::t_object*)c74::max::newinstance(c74::max::gensym("jit.coerce"),0,NULL)))
                c74::max::freeobject(coerce);
        }
        
        m_jit_coerce = (t_object*)jit_object_new(max::gensym("jit_coerce"));
        
        static c74::max::t_object *transpose=NULL;
        if (!transpose) {
            if((transpose = (c74::max::t_object*)c74::max::newinstance(c74::max::gensym("jit.transpose"),0,NULL)))
                c74::max::freeobject(transpose);
        }
        
        m_jit_transpose = (t_object*)jit_object_new(max::gensym("jit_transpose"));

    }
    
    ~mlmat_convert() {
        
        if(m_jit_scanwrap) {
            c74::max::freeobject(m_jit_scanwrap);
        }
        
        if(m_jit_coerce) {
            c74::max::freeobject(m_jit_coerce);
        }
        
        if(m_jit_transpose) {
            c74::max::freeobject(m_jit_transpose);
        }
    }
    
    
    t_jit_err jitter_transpose(t_object* in_matrix, t_object* out_matrix) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_info, minfo;
        
        jit_object_method(in_matrix, _jit_sym_getinfo, &in_info);
        
        minfo = in_info;
        
        t_object* transposed_matrix = nullptr;
        
        minfo.dimcount = 2;
        minfo.dim[0] = in_info.dim[1];
        minfo.dim[1] = in_info.dim[0];
        
        transposed_matrix = static_cast<t_object*>(max::jit_object_new(_jit_sym_jit_matrix,&minfo));
        
        err = (t_jit_err)jit_object_method((t_object*)m_jit_transpose, _jit_sym_matrix_calc,in_matrix, transposed_matrix);
        if(err) { std::cerr << err << std::endl;}
        
        err = (t_jit_err)max::object_method(out_matrix,_jit_sym_setinfo,&minfo);
        if(err) { std::cerr << err << std::endl;}
        err = (t_jit_err)max::object_method(out_matrix,max::_jit_sym_frommatrix,transposed_matrix,nullptr);
        if(err) { std::cerr << err << std::endl;}
        
        if(transposed_matrix) {
            jit_object_free(transposed_matrix);
        }
        
        
        return err;
    }
    
    t_jit_err jitter_coerce(t_object* in_matrix, t_object* out_matrix, long planecount, t_symbol *type ) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info minfo, in_info;
        t_object* tmp_matrix = nullptr;
        char *in_bp;
        
        jit_object_method(in_matrix, _jit_sym_getinfo, &in_info);
        jit_object_method(in_matrix,_jit_sym_getdata,&in_bp);
        
        minfo = in_info;
        minfo.flags = JIT_MATRIX_DATA_REFERENCE|JIT_MATRIX_DATA_FLAGS_USE;
        minfo.planecount = planecount;
        minfo.type = type;
        
        if(in_info.planecount > planecount) {
            minfo.dim[0] = in_info.planecount*in_info.dim[0];
        } else {
            minfo.dim[0] = in_info.dim[0] / planecount;
        }
        
        tmp_matrix = static_cast<t_object*>(max::jit_object_new(_jit_sym_jit_matrix,&minfo));

        jit_object_method(tmp_matrix,_jit_sym_data,in_bp);
        
        ///
        err =  jit_attr_setlong(m_jit_coerce, _jit_sym_planecount, planecount);
        if(err) { std::cerr << err << std::endl;}
        err =  jit_attr_setsym(m_jit_coerce, _jit_sym_type, type);
        if(err) { std::cerr << err << std::endl;}
        
        err = (t_jit_err)jit_object_method((t_object*)m_jit_coerce, _jit_sym_matrix_calc,in_matrix, tmp_matrix);
        if(err) { std::cerr << err << std::endl;}

        minfo.flags = 0 ;
        
        jit_object_method(out_matrix, _jit_sym_setinfo, &minfo);
        
        err = (t_jit_err)max::object_method(out_matrix,max::_jit_sym_frommatrix,tmp_matrix,nullptr);
        
        if(err) { std::cerr << err << std::endl;}

        if (tmp_matrix) jit_object_free(tmp_matrix);
        
        return err;
    }
    
    t_jit_err jitter_flatten(t_object* in_matrix, t_object* out_matrix, bool transposed = false) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_info, minfo;
        t_object* coerced_matrix = nullptr;
        t_object* scanwrapped_matrix = nullptr;
        
        jit_object_method(in_matrix, _jit_sym_getinfo, &in_info);
        
        jit_matrix_info_default(&minfo);
                          
        minfo.type = in_info.type;
        minfo.planecount = 1;
        minfo.dim[0] = in_info.planecount*in_info.dim[0];
        
        coerced_matrix = static_cast<t_object*>(max::jit_object_new(_jit_sym_jit_matrix,&minfo));

        jitter_coerce(in_matrix, coerced_matrix, 1, minfo.type);


        minfo.type = in_info.type;
        minfo.dimcount = 2;
        minfo.planecount = 1;
        minfo.flags = 0;
        minfo.dim[0] = in_info.planecount;
        minfo.dim[1] = in_info.dim[0]*in_info.dim[1];

        scanwrapped_matrix = static_cast<t_object*>(max::jit_object_new(_jit_sym_jit_matrix,&minfo));
        jit_object_method(scanwrapped_matrix, _jit_sym_getinfo, &minfo);

        long newdim[2] = {minfo.dim[0], minfo.dim[1]};

        err =  jit_attr_setlong_array(m_jit_scanwrap, _jit_sym_dim, 2, newdim);
        if(err) { std::cerr << err << std::endl;}
        err =  jit_attr_setlong(m_jit_coerce, _jit_sym_planecount, 1);
        if(err) { std::cerr << err << std::endl;}
        err =  jit_attr_setsym(m_jit_coerce, _jit_sym_type, minfo.type);
        if(err) { std::cerr << err << std::endl;}

        err = (t_jit_err)jit_object_method((t_object*)m_jit_scanwrap, _jit_sym_matrix_calc,coerced_matrix, scanwrapped_matrix);
        if(err) { std::cerr << err << std::endl;}
        
        if(!transposed) {
            err = (t_jit_err)max::object_method(out_matrix,_jit_sym_setinfo,&minfo);
            if(err) { std::cerr << err << std::endl;}
            err = (t_jit_err)max::object_method(out_matrix,max::_jit_sym_frommatrix,scanwrapped_matrix,nullptr);
            if(err) { std::cerr << err << std::endl;}
        } else {
            jitter_transpose(scanwrapped_matrix, out_matrix);
        }

        if(scanwrapped_matrix) {
            jit_object_free(scanwrapped_matrix);
        }

        
        if(coerced_matrix) {
            jit_object_free(coerced_matrix);
        }
        
        return err;
    }
    
    t_jit_err jitter_extrude(t_object* in_matrix, t_object* out_matrix, bool transposed = false) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_info, minfo;
        t_object* initial_matrix = nullptr;
        t_object* coerced_matrix = nullptr;
        
        jit_object_method(in_matrix, _jit_sym_getinfo, &in_info);
        initial_matrix = static_cast<t_object*>(max::jit_object_new(_jit_sym_jit_matrix,&in_info));
        
        if(!transposed) {
            err = (t_jit_err)max::object_method(initial_matrix,max::_jit_sym_frommatrix,in_matrix,nullptr);
            if(err) { std::cerr << err << std::endl;}
        } else {
            jitter_transpose(in_matrix, initial_matrix);
        }

        jit_object_method(initial_matrix, _jit_sym_getinfo, &in_info);
        
        minfo.type = in_info.type;
        minfo.dimcount = 1;
        minfo.planecount = in_info.dim[0];
        minfo.flags = 0;
        minfo.dim[0] = in_info.dim[1];
        
        if(minfo.planecount > 32) {
            (cerr << "can't create a matrix with more than 32 planes. " << endl);
            goto out;
        }
        
        coerced_matrix = static_cast<t_object*>(max::jit_object_new(_jit_sym_jit_matrix,&minfo));
        
        jitter_coerce(initial_matrix, coerced_matrix, minfo.planecount, minfo.type);
        
        jit_object_method(coerced_matrix, _jit_sym_getinfo, &minfo);

 
        err = (t_jit_err)jit_object_method(out_matrix,_jit_sym_setinfo,&minfo);
        if(err) { std::cerr << err << std::endl;}
        err = (t_jit_err)jit_object_method(out_matrix,max::_jit_sym_frommatrix,coerced_matrix,nullptr);
        if(err) { std::cerr << err << std::endl;}
        
        if(coerced_matrix) {
            jit_object_free(coerced_matrix);
        }
        
    out:
       
        if(initial_matrix) {
            jit_object_free(initial_matrix);
        }
        
        return err;
    }
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info out_minfo;
        //t_object* flat_matrix = nullptr;
        
        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto out_matrix = object_method(outputs, _jit_sym_getindex, 0);

        auto in_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_savelock = object_method(out_matrix, _jit_sym_lock, 1);
        err = (t_jit_err)max::object_method(out_matrix,_jit_sym_getinfo,&out_minfo);

        if(input_mode == output_mode) {
            //just pass through
            err = (t_jit_err)jit_object_method(out_matrix,_jit_sym_setinfo,&out_minfo);
            err = (t_jit_err)jit_object_method(out_matrix,_jit_sym_frommatrix,in_matrix,NULL);

            goto out;
        }
        
        if(input_mode == 0) {
            if(output_mode == 1) {
                jitter_flatten((t_object*)in_matrix, (t_object*)out_matrix, false);
            } else {
                jitter_flatten((t_object*)in_matrix, (t_object*)out_matrix, true);
            }
        } else if(input_mode == 1) {
            if(output_mode == 2) {
                jitter_transpose((t_object*)in_matrix, (t_object*)out_matrix);
            } else {
                jitter_extrude((t_object*)in_matrix, (t_object*)out_matrix, false);
            }
        } else if(input_mode == 2) {
            if(output_mode == 1) {
                jitter_transpose((t_object*)in_matrix, (t_object*)out_matrix);
            } else {
                jitter_extrude((t_object*)in_matrix, (t_object*)out_matrix, true);
            }
        }
        
    out:
       object_method(in_matrix,_jit_sym_lock,in_savelock);
       object_method(out_matrix,_jit_sym_lock,out_savelock);
       return err;
    }


private:

    
    message<> fileusage {this, "fileusage",
        MIN_FUNCTION {
            void *w { args[0] };
            c74::max::fileusage_addfilename(w, 0, "jit.scanwrap");
            c74::max::fileusage_addfilename(w, 0, "jit.coerce");
            c74::max::fileusage_addfilename(w, 0, "jit.transform");
            return {};
        }
    };
    
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 1, 1));
        
        jit_mop_single_type(mop, _jit_sym_float64);
        auto in1 = object_method(mop,_jit_sym_getinput,1);
        jit_attr_setlong(in1,_jit_sym_dimlink,0);
        jit_attr_setlong(in1,_jit_sym_typelink,0);
        jit_class_addadornment(c, mop);
        jit_class_addmethod(c, (method)mlmat_matrix_calc, "matrix_calc", A_CANT, 0);
        return {};
    }};
    
    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        max_jit_class_mop_wrap(c, this_jit_class, 0);
        max_jit_class_wrap_standard(c, this_jit_class, 0);
        class_addmethod(c, (method)mlmat_assist, "assist", A_CANT, 0);

        return {};
    }};
    
    t_object * m_jit_scanwrap;
    t_object * m_jit_coerce;
    t_object * m_jit_transpose;

};


MIN_EXTERNAL(mlmat_convert);


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_convert>* job = (minwrap<mlmat_convert>*)(x);
        err = job->m_min_object.matrix_calc(x, inputs, outputs);
    }

    return err;
}


void mlmat_assist(void* x, void* b, long io, long index, char* s) {
    switch(io) {
        case 1:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) coordinates");
                    break;
                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) results");
                    break;

                default:
                    sprintf(s, "dumpout");
                    break;
            }

    }
}
