/// @file mlmat.convert.cpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.


#include "mlmat.hpp"

#include <cstring>

using namespace c74;
using namespace c74::min;
using namespace c74::max;


void mlmat_assist(void* x, void* b, long m, long a, char* s) ;
t_jit_err mlmat_matrix_calc(t_object* x, t_object* inputs, t_object* outputs);


class mlmat_convert : public mlmat_object<mlmat_convert> {
public:
    MIN_DESCRIPTION	{"Convert matrices between mlmat modes. Converts a matrix to between different mlmat modes. <at>input</at> specifies the expected input mode and <at>output</at> indicates the mode to convert to. Note that this will fail if attempting to convert from mode 1 or 2 to mode 0 if the resulting matrix would require more than 32 planes."};
    MIN_TAGS		{"ml"};
    MIN_AUTHOR		{"Todd Ingalls"};
    MIN_RELATED		{"mlmat.scaling, mlmat.lookup"};

    attribute<int, threadsafe::no, limit::clamp, allow_repetitions::no> input_mode{ this, "input", 0,
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

    // The conversions are pure memory reinterpretations (the same data in a
    // different shape), so the previous implementation chained jit.coerce,
    // jit.scanwrap and jit.transpose objects through up to four temporary
    // matrices per call. The mappings are done directly here:
    //
    //   mode 0: each cell is a data point, each plane one feature
    //   mode 1: each row is a data point, each column in the row a feature
    //   mode 2: each column is a data point, each row in the column a feature
    //
    // with P = number of features and N = number of data points, and mode 0
    // cell n = (x, y) with x = n mod w, y = n div w:
    //
    //   0 -> 1: out(p, n)            = in plane p of cell n      (linear identity)
    //   1 -> 0: out plane p of cell n = in(p, n)                 (linear identity)
    //   0 -> 2: out(n, p)            = in plane p of cell n      (transpose)
    //   2 -> 0: out plane p of cell n = in(n, p)                 (transpose)
    //   1 -> 2: out(n, p)            = in(p, n)                  (transpose)
    //   2 -> 1: out(p, n)            = in(n, p)                 (transpose)
    t_jit_err matrix_calc(t_object* x, t_object* inputs, t_object* outputs) {
        t_jit_err err = JIT_ERR_NONE;
        t_jit_matrix_info in_info, out_info, actual_info;

        // all locals declared up front, so the gotos below do not cross
        // any initializations
        long es = 0;
        long P = 0, N = 0;
        long w = 0, h = 1;
        c74::max::uchar* idataptr = nullptr;
        c74::max::uchar* odataptr = nullptr;
        size_t s0i = 0, s1i = 0, s0o = 0, s1o = 0, esz = 0;
        bool in_packed = false;

        auto in_matrix = object_method(inputs, _jit_sym_getindex, 0);
        auto out_matrix = object_method(outputs, _jit_sym_getindex, 0);

        auto in_savelock = object_method(in_matrix, _jit_sym_lock, 1);
        auto out_savelock = object_method(out_matrix, _jit_sym_lock, 1);
        err = (t_jit_err)max::object_method(out_matrix,_jit_sym_getinfo,&out_info);

        if(input_mode == output_mode) {
            //just pass through
            err = (t_jit_err)jit_object_method(out_matrix,_jit_sym_setinfo,&out_info);
            err = (t_jit_err)jit_object_method(out_matrix,_jit_sym_frommatrix,in_matrix,NULL);

            goto out;
        }

        jit_object_method(in_matrix, _jit_sym_getinfo, &in_info);

        es = jit_elem_size(in_info.type);
        if(es <= 0) {
            (cerr << "unsupported matrix type." << endl);
            goto out;
        }

        // features (P) and data points (N) in the input
        if(input_mode == 0) {
            P = in_info.planecount;
            w = in_info.dim[0];
            h = (in_info.dimcount > 1) ? in_info.dim[1] : 1;
            N = w * h;
        } else if(input_mode == 1) {
            P = in_info.dim[0];
            N = (in_info.dimcount > 1) ? in_info.dim[1] : 1;
        } else {
            N = in_info.dim[0];
            P = (in_info.dimcount > 1) ? in_info.dim[1] : 1;
        }

        if(output_mode == 0 && P > 32) {
            (cerr << "can't create a matrix with more than 32 planes. " << endl);
            goto out;
        }

        // output shape (type preserved, as before)
        memset(&out_info, 0, sizeof(out_info));
        out_info.type = in_info.type;
        out_info.flags = 0;
        if(output_mode == 0) {
            out_info.planecount = P;
            out_info.dimcount = 1;
            out_info.dim[0] = N;
        } else if(output_mode == 1) {
            out_info.planecount = 1;
            out_info.dimcount = 2;
            out_info.dim[0] = P;
            out_info.dim[1] = N;
        } else {
            out_info.planecount = 1;
            out_info.dimcount = 2;
            out_info.dim[0] = N;
            out_info.dim[1] = P;
        }

        err = (t_jit_err)jit_object_method(out_matrix,_jit_sym_setinfo,&out_info);
        if(err) { std::cerr << err << std::endl; }
        jit_object_method(out_matrix, _jit_sym_getinfo, &actual_info);

        jit_object_method(in_matrix, _jit_sym_getdata, &idataptr);
        jit_object_method(out_matrix, _jit_sym_getdata, &odataptr);

        if(idataptr == nullptr || odataptr == nullptr) {
            goto out;
        }

        s0i = size_t(in_info.dimstride[0]);
        s1i = (in_info.dimcount > 1) ? size_t(in_info.dimstride[1]) : 0;
        s0o = size_t(actual_info.dimstride[0]);
        s1o = (actual_info.dimcount > 1) ? size_t(actual_info.dimstride[1]) : 0;
        esz = size_t(es);
        in_packed = mlmat::jit_is_packed(in_info, esz);

        if(input_mode == 0 && output_mode == 1) {
            // out(p, n) = in plane p of cell n -- linear identity when packed
            if(in_packed) {
                memcpy(odataptr, idataptr, size_t(P) * size_t(N) * esz);
            } else {
                for(long n = 0; n < N; ++n) {
                    const long x = n % w;
                    const long y = n / w;
                    const c74::max::uchar* src = idataptr + size_t(y) * s1i + size_t(x) * s0i;
                    c74::max::uchar* dst = odataptr + size_t(n) * s1o;
                    for(long p = 0; p < P; ++p) {
                        memcpy(dst + size_t(p) * s0o, src + size_t(p) * esz, esz);
                    }
                }
            }
        } else if(input_mode == 0 && output_mode == 2) {
            // out(n, p) = in plane p of cell n
            for(long n = 0; n < N; ++n) {
                const long x = n % w;
                const long y = n / w;
                const c74::max::uchar* src = idataptr + size_t(y) * s1i + size_t(x) * s0i;
                for(long p = 0; p < P; ++p) {
                    memcpy(odataptr + size_t(p) * s1o + size_t(n) * s0o, src + size_t(p) * esz, esz);
                }
            }
        } else if(input_mode == 1 && output_mode == 0) {
            // out plane p of cell n = in(p, n) -- linear identity when packed
            if(in_packed) {
                memcpy(odataptr, idataptr, size_t(P) * size_t(N) * esz);
            } else {
                for(long n = 0; n < N; ++n) {
                    for(long p = 0; p < P; ++p) {
                        memcpy(odataptr + size_t(n) * s0o + size_t(p) * esz,
                               idataptr + size_t(n) * s1i + size_t(p) * s0i, esz);
                    }
                }
            }
        } else if(input_mode == 2 && output_mode == 0) {
            // out plane p of cell n = in(n, p)
            for(long n = 0; n < N; ++n) {
                for(long p = 0; p < P; ++p) {
                    memcpy(odataptr + size_t(n) * s0o + size_t(p) * esz,
                           idataptr + size_t(p) * s1i + size_t(n) * s0i, esz);
                }
            }
        } else if(input_mode == 1 && output_mode == 2) {
            // out(n, p) = in(p, n)
            for(long n = 0; n < N; ++n) {
                for(long p = 0; p < P; ++p) {
                    memcpy(odataptr + size_t(p) * s1o + size_t(n) * s0o,
                           idataptr + size_t(n) * s1i + size_t(p) * s0i, esz);
                }
            }
        } else if(input_mode == 2 && output_mode == 1) {
            // out(p, n) = in(n, p)
            for(long n = 0; n < N; ++n) {
                for(long p = 0; p < P; ++p) {
                    memcpy(odataptr + size_t(n) * s1o + size_t(p) * s0o,
                           idataptr + size_t(p) * s1i + size_t(n) * s0i, esz);
                }
            }
        }

    out:
       object_method(in_matrix,_jit_sym_lock,in_savelock);
       object_method(out_matrix,_jit_sym_lock,out_savelock);
       return err;
    }

private:

    // size of one element for the jit matrix types
    static long jit_elem_size(t_symbol* type) {
        if(type == _jit_sym_char) {
            return 1;
        } else if(type == _jit_sym_long) {
            return 4;
        } else if(type == _jit_sym_float32) {
            return 4;
        } else if(type == _jit_sym_float64) {
            return 8;
        }
        return 0;
    }

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
                    snprintf(s, 64, "(matrix) coordinates");
                    break;
                default:
                    break;
            }
            break;
        case 2:
            switch(index) {
                case 0:
                    snprintf(s, 64, "(matrix) results");
                    break;

                default:
                    snprintf(s, 64, "dumpout");
                    break;
            }

    }
}
