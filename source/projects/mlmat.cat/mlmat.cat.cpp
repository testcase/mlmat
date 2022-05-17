/// @file mlmat.convert.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include "mlmat_object.hpp"

using namespace c74;
using namespace c74::min;
using namespace c74::max;

void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);


class mlmat_cat : public mlmat_object<mlmat_cat> {
public:
    MIN_DESCRIPTION	{""};
    MIN_TAGS		{"ml"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.scaling, mlmat.lookup"};
    
    mlmat_cat() {

    }
    
    ~mlmat_cat() {
        if(m_matrix_data) {
            sysmem_freeptr(m_matrix_data);
            //sysmem_freeptr(m_matrix_data);
        }
        
        if(m_matrix) {
            
            //jit_object_free(m_matrix);
        }
    }


    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_matrix_info, out_matrix_info;
        char *in_bp;
        
        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto out_matrix = object_method(outputs, _jit_sym_getindex, 0);
        
        auto in_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_savelock = object_method(out_matrix, _jit_sym_lock, 1);
        
        
        jit_object_method(in_matrix, _jit_sym_getinfo, &in_matrix_info);
        
        
        if(m_matrix == nullptr && m_matrix_data == nullptr) {
            t_jit_matrix_info minfo = in_matrix_info;
            minfo.flags = JIT_MATRIX_DATA_REFERENCE|JIT_MATRIX_DATA_FLAGS_USE;
            m_matrix = jit_object_new(_jit_sym_jit_matrix, &minfo);
            m_matrix_data = sysmem_newptr(minfo.size);
            
            //get pointer to input matrix data
            jit_object_method(in_matrix,_jit_sym_getdata,&in_bp);
            //set the external data for matrix to ptr
            jit_object_method(m_matrix,_jit_sym_data,m_matrix_data);
            //copy memory from input data to ptr
            sysmem_copyptr(in_bp, m_matrix_data, minfo.size);
        } else if(m_matrix != nullptr && m_matrix_data != nullptr) {
            //get pointer to input matrix data
            t_jit_matrix_info minfo;
            t_ptr *ptr ;
            jit_object_method(m_matrix, _jit_sym_getinfo, &minfo);
            //get pointer to input matrix data
            jit_object_method(in_matrix,_jit_sym_getdata,&in_bp);
            //increase size of data ptr
            m_matrix_data = sysmem_resizeptr(m_matrix_data, minfo.size + in_matrix_info.size);
            ptr =  &m_matrix_data;
            //if above fails can't do this.
            ptr += in_matrix_info.size;
            sysmem_copyptr(in_bp, ptr, in_matrix_info.size);
            
            minfo.size += in_matrix_info.size;
            minfo.dim[1] += in_matrix_info.dim[1];
            minfo.dimstride[1] += in_matrix_info.dimstride[1];
            minfo.flags = JIT_MATRIX_DATA_REFERENCE|JIT_MATRIX_DATA_FLAGS_USE;
            err = (t_jit_err) jit_object_method(m_matrix, _jit_sym_setinfo_ex, &minfo);
           // std::cout << err << std::endl;
            err = (t_jit_err) jit_object_method(m_matrix,_jit_sym_data,m_matrix_data);
            //std::cout << err << std::endl;
        
        } else {
            std::cerr << "oops that aint right. did not free something" << std::endl;
            goto out;
        }
        
        jit_object_method(m_matrix, _jit_sym_getinfo, &out_matrix_info);
        
        out_matrix_info.flags = 0;
        err = (t_jit_err)jit_object_method(out_matrix,_jit_sym_setinfo,&out_matrix_info);
        //std::cout << err << std::endl;
        err = (t_jit_err) jit_object_method(out_matrix,_jit_sym_frommatrix,m_matrix,NULL);
        //std::cout << err << std::endl;
       


  
        
       
    out:
       object_method(in_matrix,_jit_sym_lock,in_savelock);
       object_method(out_matrix,_jit_sym_lock,out_savelock);
       return err;
    }
    
    max::t_ptr m_matrix_data { nullptr };
    void * m_matrix;
    
    

private:

     
   
    
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 1, 1));
        
        jit_mop_single_type(mop, _jit_sym_float64);

        
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
    
    

};


MIN_EXTERNAL(mlmat_cat);


t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_cat>* job = (minwrap<mlmat_cat>*)(x);
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
