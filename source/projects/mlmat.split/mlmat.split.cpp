/// @file mlmat.split.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.
/// TODO: Use scaling_model.hpp for serialization
/// TODO: Need to fix up error messagesß

#include "c74_min.h"

#include "mlmat_object.hpp"
#include <mlpack/core/data/split_data.hpp>


using namespace c74;
using namespace c74::min;
using namespace c74::max;
using namespace mlpack::data;

void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);
void max_mlmat_jit_matrix(max_jit_wrapper *x, t_symbol *s, short argc,t_atom *argv);
void max_jit_mlmat_mproc(max_jit_wrapper *x, void *mop);

class mlmat_split : public mlmat_object<mlmat_split> {
public:
    MIN_DESCRIPTION	{"Splits data into testing and training."};
    MIN_TAGS		{"ML"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.scaling"};
    
    attribute<int> seed { this, "seed", 0,
        description {
            "Random seed if random basis being used. 0 indicates no seed."
        }
    };
    
    attribute<bool> use_labels { this, "use_labels", false,
           description {
               "If 1 labels are expected. "
        }
    };
    
    attribute<double, threadsafe::no, limit::clamp> ratio { this, "ratio", 0.2,
        description {
            "Percentage of dataset to use for test set (between 0 and 1)."
        },
        range {0.0, 1.0}
    };

    attribute<bool> shuffle { this, "shuffle", true,
           description {
               "If true, the sample order is shuffled; otherwise, each sample is visited in linear order.  "
        }
    };

    //should make private and have setter i guess
    bool m_received_labels = false;
    
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_matrix_info, in_labels_info, out_train_data_info, out_test_data_info,out_train_labels_info, out_test_labels_info;
        arma::Mat<double> input_data;
        arma::Row<size_t> input_labels;
        arma::Mat<double> train_data;
        arma::Row<size_t> train_labels;
        arma::Mat<double> test_data;
        arma::Row<size_t> test_labels;
       
        auto in_matrix = (t_object*)object_method(inputs, _jit_sym_getindex, 0);
        auto in_labels_matrix = (t_object*)object_method(inputs, _jit_sym_getindex, 1);
        auto out_train_matrix = (t_object*)object_method(outputs, _jit_sym_getindex, 0);
        auto out_test_matrix = (t_object*)object_method(outputs, _jit_sym_getindex, 1);

    
        auto in_matrix_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto in_labels_matrix_savelock = object_method(in_labels_matrix, _jit_sym_lock, 1);
        auto out_train_matrix_savelock = object_method(out_train_matrix, _jit_sym_lock, 1);
        auto out_test_matrix_savelock = object_method(out_test_matrix, _jit_sym_lock, 1);


        
        object_method(in_matrix, _jit_sym_getinfo, &in_matrix_info);
        t_object* in_matrix64 = convert_to_float64(static_cast<t_object*>(in_matrix), in_matrix_info);
        
        input_data = jit_to_arma(mode, in_matrix64, input_data);
        
        
        if(use_labels && !m_received_labels) {
            (cerr << "Have not received labels matrix." << endl);
            goto out;
        }
        
        if(use_labels) {
            auto out_train_labels = (t_object*)object_method(outputs, _jit_sym_getindex, 2);
            auto out_test_labels = (t_object*)object_method(outputs, _jit_sym_getindex, 3);
            
            auto out_train_labels_savelock = object_method(out_train_labels, _jit_sym_lock, 1);
            auto out_test_labels_savelock = object_method(out_test_labels, _jit_sym_lock, 1);
        
            object_method(in_labels_matrix, _jit_sym_getinfo, &in_labels_info);
            
            try {
                check_mode(in_labels_info, mode, "split");
            } catch (std::invalid_argument& s) {
                cerr << s.what() << endl;
                object_method(out_train_labels,_jit_sym_lock,out_train_labels_savelock);
                object_method(out_test_labels,_jit_sym_lock,out_test_labels_savelock);
                object_method(out_train_labels,_jit_sym_lock,out_train_labels_savelock);
                object_method(out_test_labels,_jit_sym_lock,out_test_labels_savelock);
                goto out;
            }
            
            if(in_labels_info.planecount > 1) {
                (cerr << "Labels should be in a single plane matrix. Matrix received has " << in_labels_info.planecount << "." << endl);
                object_method(out_train_labels,_jit_sym_lock,out_train_labels_savelock);
                object_method(out_test_labels,_jit_sym_lock,out_test_labels_savelock);
                object_method(out_train_labels,_jit_sym_lock,out_train_labels_savelock);
                object_method(out_test_labels,_jit_sym_lock,out_test_labels_savelock);
                goto out;
            }
            
            input_labels = jit_to_arma(mode, in_labels_matrix, input_labels);
            
            if(input_labels.n_cols != input_data.n_cols) {
                (cerr << "labels wrong size. got " << input_labels.n_cols << " expected " << input_data.n_cols << "." << endl);
                object_method(out_train_labels,_jit_sym_lock,out_train_labels_savelock);
                object_method(out_test_labels,_jit_sym_lock,out_test_labels_savelock);
                object_method(out_train_labels,_jit_sym_lock,out_train_labels_savelock);
                object_method(out_test_labels,_jit_sym_lock,out_test_labels_savelock);
                goto out;
            }
            
            mlpack::data::Split(input_data, input_labels, train_data, test_data, train_labels, test_labels, ratio, shuffle);
            
            out_train_labels_info = in_labels_info;
            out_train_labels_info.dimcount = 1;
            out_train_labels_info.dim[0] = train_labels.n_cols;
            
            out_test_labels_info = in_labels_info;
            out_test_labels_info.dimcount = 1;
            out_test_labels_info.dim[0] = test_labels.n_cols;
            
            out_train_labels = arma_to_jit(mode, train_labels, out_train_labels, out_train_labels_info);
            out_test_labels = arma_to_jit(mode, test_labels, out_test_labels, out_test_labels_info);
            
        } else {
            mlpack::data::Split(input_data, train_data, test_data, ratio, shuffle);
        }
    
        out_train_data_info = in_matrix_info;
        out_train_data_info.dimcount = 1;
        out_train_data_info.dim[0] = train_data.n_cols;
        
        out_test_data_info = in_matrix_info;
        out_test_data_info.dimcount = 1;
        out_test_data_info.dim[0] = test_data.n_cols;
        
        out_train_matrix = arma_to_jit(mode, train_data, out_train_matrix, out_train_data_info);
        out_test_matrix = arma_to_jit(mode, test_data, out_test_matrix, out_test_data_info);

         
    out:
        if(in_matrix != in_matrix64) { jit_object_free(in_matrix64); }
        object_method(in_matrix,_jit_sym_lock,in_matrix_savelock);
        object_method(out_train_matrix,_jit_sym_lock,out_train_matrix_savelock);
        object_method(out_test_matrix,_jit_sym_lock,out_test_matrix_savelock);
        object_method(in_labels_matrix,_jit_sym_lock,in_labels_matrix_savelock );
        

        return err;
        
    }
    

private:
    // override jitclass_setup so we can have our own matrix_calc. jitclass_setup is called first (and only once when the object is loaded for the first time) during the intitialization of the object.
    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        t_class* c = args[0];

        t_atom exact_types[1];
            
        atom_setsym(exact_types, _jit_sym_long);
        

        // add mop
        t_object* mop = static_cast<t_object*>(jit_object_new(_jit_sym_jit_mop, 2, 4));
        
        jit_mop_single_type(mop, _jit_sym_float64);
        

        auto in2 = object_method(mop,_jit_sym_getinput,2);
        auto out3 = object_method(mop,_jit_sym_getoutput,3);
        auto out4 = object_method(mop,_jit_sym_getoutput,4);
        
        
        object_method_typed(in2, _jit_sym_types, 1, exact_types, NULL);
        object_method_typed(out3, _jit_sym_types, 1, exact_types, NULL);
        object_method_typed(out4, _jit_sym_types, 1, exact_types, NULL);
        
        
        jit_attr_setlong(in2,_jit_sym_dimlink,0);
        jit_attr_setlong(in2,_jit_sym_typelink,0);


        
        //always adapt
        jit_object_method(in2,_jit_sym_ioproc,jit_mop_ioproc_copy_adapt);
               
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
        class_addmethod(c, (method)mlmat_assist, "assist", A_CANT, 0);
        max_jit_classex_mop_mproc(c,this_jit_class,(void*)max_jit_mlmat_mproc);
        class_addmethod(c, (method)max_mlmat_jit_matrix, "jit_matrix", A_GIMME, 0);
        return {};
    }};

};


MIN_EXTERNAL(mlmat_split);


void max_jit_mlmat_mproc(max_jit_wrapper *x, void *mop)
{
    t_jit_err err =JIT_ERR_NONE;
    void *o,*p, *j;
    t_atom a;
    long outputmode = max_jit_mop_getoutputmode(x);

    if (outputmode==1) {


        // send in link list that includes only outputs 2 3 4
        t_linklist * op =  static_cast<t_linklist*>(object_method(mop,_jit_sym_getoutputlist));
        j = max_jit_obex_jitob_get(x);
        minwrap<mlmat_split>* job = (minwrap<mlmat_split>*)(j);

        bool use_labels = job->m_min_object.use_labels;


        t_linklist *outputlist = linklist_new();
        linklist_append(outputlist, linklist_getindex(op, 0));
        linklist_append(outputlist, linklist_getindex(op, 1));

        if(use_labels) {
            linklist_append(outputlist, linklist_getindex(op, 2));
            linklist_append(outputlist, linklist_getindex(op, 3));
        }

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

            if ((p=object_method(mop,_jit_sym_getoutput,2)) &&
                (o=max_jit_mop_io_getoutlet(p)))
            {
                atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
                outlet_anything(o,_jit_sym_jit_matrix,1,&a);
            }

            if(use_labels) {
                if ((p=object_method(mop,_jit_sym_getoutput,3)) &&
                        (o=max_jit_mop_io_getoutlet(p)))
                {
                    atom_setsym(&a,object_attr_getsym(p,_jit_sym_matrixname));
                    outlet_anything(o,_jit_sym_jit_matrix,1,&a);
                }

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

t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
    t_jit_err err = JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_split>* job = (minwrap<mlmat_split>*)(x);
        err = job->m_min_object.matrix_calc(x, inputs, outputs);
    }

    return err;
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
        
        void *p, *mop;
        
        if (!(mop=max_jit_obex_adornment_get(x,_jit_sym_jit_mop)))
            jit_error_code(x,err);
        
        p = object_method((t_object*)mop,_jit_sym_getinput,2);
        j = max_jit_obex_jitob_get(x);
        
        minwrap<mlmat_split>* job = (minwrap<mlmat_split>*)(j);
        job->m_min_object.m_received_labels = true;
        
       
    }
    
}



void mlmat_assist(void* x, void* b, long io, long index, char* s) {
    switch(io) {
        case 1:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) training data");
                    break;

                case 1:
                    sprintf(s, "(matrix) training labels");
                    break;

                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    sprintf(s, "(matrix) training data");
                    break;

                case 1:
                    sprintf(s, "(matrix) testing data");
                    break;
                    
                case 2:
                    sprintf(s, "(matrix) training labels");
                    break;
                    
                case 3:
                    sprintf(s, "(matrix) testing labels");
                    break;
                    
                default:
                    sprintf(s, "dumpout");
                    break;
            }

    }
}



