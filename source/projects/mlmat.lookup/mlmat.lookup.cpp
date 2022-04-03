/// @file mlmat.lookup.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// TODO: Other distance Metrics?
/// TODO: Serialization read/write
/// TODO: Testing

#include "c74_min.h"
#include "mlmat_operator.hpp"

using namespace c74;
using namespace c74::min;
using namespace c74::max;

void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);
void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);


class mlmat_lookup : public mlmat_operator<mlmat_lookup> {
public:
    MIN_DESCRIPTION	{"Utility for getting indexed data from a matrix."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.scaling"};

    
    inlet<>  input1	{ this, "(matrix) Matrix containing query points.", "matrix" };
    inlet<>  input2 { this, "(matrix) Matrix containing data matrix.", "matrix" };
    outlet<> output	{ this, "(matrix) Matrix with lookup.", "matrix"  };

    message<> clear { this, "clear", "clear model.",
        MIN_FUNCTION {
            m_data.reset();
            return {};
        }
        
    };

    t_jit_err process_data_matrix(t_object *matrix) {
        t_jit_err err = JIT_ERR_NONE;
        arma::Mat<double> dat;
        dat = jit_to_arma(mode, matrix, dat);
        m_data = std::make_unique<arma::Mat<double>>(std::move(dat));
        m_mode_changed = false;
        return err;
    }
    
    
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_query_info,in_data_info, out_info;
        arma::Col<arma::uword> query;
        arma::mat resulting;
        t_object* in_matrix_long = nullptr;
       
        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto data_matrix = object_method(inputs, _jit_sym_getindex, 1);
        auto out_results = object_method(outputs, _jit_sym_getindex, 0);

        auto query_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto data_matrix_savelock = object_method(data_matrix, _jit_sym_getindex, 1);
        auto out_results_savelock = object_method(out_results, _jit_sym_lock, 1);
        object_method(in_matrix, _jit_sym_getinfo, &in_query_info);
        object_method(data_matrix, _jit_sym_getinfo, &in_data_info);
        
        
        if(m_data == nullptr) {
            (cerr << "no data to lookup" << endl);
            goto out;
        }
      
        if(m_mode_changed) {
            (cerr << "mode has changed must resubmit reference set" << endl);
            goto out;
        }
        
          
        
        // need better check for diff modes
//        if(in_query_info.planecount != in_data_info.dimcount) {
//            (cerr << "wrong planecount" << endl);
//            goto out;
//        }
        
        try {
            check_mode(in_query_info, mode, "lookup");
        } catch (const std::invalid_argument& s) {
            cout << s.what() << endl;
            goto out;
        }
    
        in_matrix_long  = convert_to_long(static_cast<t_object*>(in_matrix), in_query_info);
        
        query = jit_to_arma_limit(mode, static_cast<t_object*>(in_matrix_long), query, in_data_info.dim[0], in_data_info.dim[1]);

        query.clamp(0, (m_data->n_cols-1));//this is extra safeguard. could remove

        resulting = m_data->cols(query);
        
        out_info = in_query_info;
        out_info.type = _jit_sym_float64;
    
        switch(mode.get()) {
            case 0:
                out_info.planecount = resulting.n_rows;
                out_info.dimcount = in_query_info.dimcount;
                out_info.dim[0] =   in_query_info.dim[0];
                out_info.dim[1] =   in_query_info.dim[1];
                break;
            case 1:
                out_info.planecount = 1;
                out_info.dim[0] = resulting.n_rows;
                out_info.dim[1] = resulting.n_cols;
                break;
            case 2:
                out_info.planecount = 1;
                out_info.dim[0] = resulting.n_cols;
                out_info.dim[1] = resulting.n_rows;
                break;
            default:
                out_info.planecount = 1;
                break;
                
        }

        out_results = arma_to_jit(mode, resulting, static_cast<t_object*>(out_results), out_info);


    out:
        
        if(in_matrix != in_matrix_long) { jit_object_free(in_matrix_long); }
        object_method(in_matrix,_jit_sym_lock,query_savelock);
        object_method(data_matrix,_jit_sym_lock,data_matrix_savelock);
        object_method(out_results,_jit_sym_lock,out_results_savelock);

        return err;
    }
    
private:
    // override jitclass_setup so we can have our own matrix_calc. jitclass_setup is called first (and only once when the object is loaded for the first time) during the intitialization of the object.
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];
        // add mop
        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 2, 1));
        
        // force type
        jit_mop_single_type(mop, _jit_sym_float64);
        jit_mop_input_nolink(mop, 2);
        jit_mop_output_nolink(mop, 1);
        
        auto input2 = jit_object_method(mop,_jit_sym_getinput,2);
        
        //always adapt
        jit_object_method(input2,_jit_sym_ioproc,jit_mop_ioproc_copy_adapt);
        
        jit_class_addadornment(c, mop);
        
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
        class_addmethod(c, (method)max_mlmat_jit_matrix, "jit_matrix", A_GIMME, 0);
        class_addmethod(c, (method)mlmat_assist, "assist", A_CANT, 0);

        return {};
    }};
    
    std::unique_ptr<arma::Mat<double>> m_data { nullptr };
};


MIN_EXTERNAL(mlmat_lookup);


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
    
        minwrap<mlmat_lookup>* job = (minwrap<mlmat_lookup>*)(j);
        err = job->m_min_object.process_data_matrix(static_cast<t_object*>(m));

        jit_error_code(x,err);
    }
}

t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_lookup>* job = (minwrap<mlmat_lookup>*)(x);
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

                case 1:
                    sprintf(s, "(matrix) dataset");
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
