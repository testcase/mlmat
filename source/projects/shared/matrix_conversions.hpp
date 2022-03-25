/// @file
/// @ingroup
/// @copyright Copyright 2018 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.

#pragma once


#include "c74_min.h"
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include "matrix_conversions.hpp"
using namespace c74;
using namespace c74::min;
using namespace c74::max;

using max::t_object;
using max::t_symbol;
using max::t_atom;
using max::t_class;
using max::gensym;
using max::t_jit_matrix_info;
using max::t_jit_err;
using max::t_int32;
using max::JIT_ERR_NONE;
using max::_jit_sym_float64;
using max::_jit_sym_float32;
using max::_jit_sym_long;
using max::_jit_sym_char;
using max::_jit_sym_jit_matrix;
using max::_jit_sym_getinfo;
using max::_jit_sym_getdata;
using max::_jit_sym_setinfo;
using max::object_method_imp;
using max::_jit_sym_frommatrix;
using max::t_jit_matrix_info;
using max::t_jit_op_info;


t_object* convert_to_float64(t_object *matrix, t_jit_matrix_info& minfo) {
    t_object *m; // destination matrix
    if(minfo.type == _jit_sym_float64) {
        return matrix;
    }
    t_jit_matrix_info dest_info = minfo;
    
    dest_info.type = _jit_sym_float64;
    m = static_cast<t_object*>(max::jit_object_new(_jit_sym_jit_matrix,&dest_info));
    max::object_method(m,_jit_sym_frommatrix,matrix,NULL);
    return m;
}


t_object* convert_to_long(t_object *matrix, t_jit_matrix_info& minfo) {
    t_object *m; // destination matrix
   
    if(minfo.type == c74::max::_jit_sym_long) {
        return matrix;
    }
    t_jit_matrix_info dest_info = minfo;
    dest_info.type = _jit_sym_long;
    m = static_cast<t_object*>(max::jit_object_new(_jit_sym_jit_matrix,&dest_info));
    max::object_method(m,_jit_sym_frommatrix,matrix,NULL);
    
    return m;
}

//feel like this template is a little hacky with the coords stuff but keeps other things cleaner
template <typename M, typename A>
t_jit_err fill_jit_matrix(t_object* jitter_matrix, A arma, int mode, bool is_coords = false, long x = 0) {
    uchar *dataptr = nullptr;
    t_jit_err err = JIT_ERR_NONE;
    size_t stepsize = sizeof(M);
    t_jit_matrix_info minfo;
    uchar *p = nullptr;
    uchar *p2 = nullptr;
    uchar *p1 = nullptr;
    long aelem = 0;
    
    err = (t_jit_err)max::object_method(jitter_matrix,_jit_sym_getinfo,&minfo);
    err = (t_jit_err)max::object_method(jitter_matrix, _jit_sym_getdata, &dataptr);
    
    switch (mode) {
        case 0:
            //if 2 planes and is suppose to contain 2d coords, do the conversion
            if(is_coords && (minfo.planecount == 2)) {
                M pos = 0;
                for(auto jslice=0;jslice<minfo.dim[2];jslice++) {
                    p = dataptr + (jslice*minfo.dimstride[2]);
                    for(auto jcol=0;jcol<minfo.dim[0];jcol++) {
                        p2 = p + (jcol*minfo.dimstride[0]);
                        for(auto jrow=0;jrow<minfo.dim[1];jrow++) {
                            p1 = p2 + (jrow*minfo.dimstride[1]);
                            pos = arma(aelem++);
                            *(M*)p1 = (long)pos % x;
                            p1 += stepsize;
                            *(M*)p1 = (long)pos / x;
                        }
                    }
                }
                
            } else {
                for(auto jslice=0;jslice<minfo.dim[2];jslice++) {
                    p = dataptr + (jslice*minfo.dimstride[2]);
                    // std::cout << "slice " << jslice << " " << (p - dataptr) << std::endl;
                    for(auto jcol=0;jcol<minfo.dim[0];jcol++) {
                        p2 = p +   (jcol*minfo.dimstride[0]);
                        // std::cout << "  col " << jcol << " " << (p2 - dataptr) << std::endl;
                        for(auto jrow=0;jrow<minfo.dim[1];jrow++) {
                            p1 = p2 + (jrow*minfo.dimstride[1]);
                            // std::cout << "    row " << jrow << " " << (p1 - dataptr) << std::endl;
                            for(auto jplane=0;jplane<minfo.planecount;jplane++) {
                                *(M*)p1 = arma(aelem++);
                                p1 += stepsize;
                            }
                        }
                    }
                }
            }
            break;
            
        case 1:
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    *(M*)p = arma(aelem++);
                    p += stepsize;
                }
            }
            break;
            
        case 2:
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    *(M*)p = arma(aelem++);
                    p += stepsize;
                }
            }
            break;
            
        default:
            (std::cerr << "could not create matrix" << endl);
            break;
    }
    return err;
}


/*
 converts a double arma row to a jitter matrix
 mode 0: returns a 1 cell matrix with each entry in the row being a plane
 possible it could hold coordinates that need to be converted to 2d coords. Would it even make sense
 to use 2d ?
 mode 1: returns a 1d matrix along dim[0]
 mode 2: returns a 1d matrix along dim[1]
 
 */

t_object* arma_to_jit(const int mode,
                      const arma::Row<double>& arma,
                      t_object *jitter_matrix,
                      t_jit_matrix_info& target_info,
                      const bool is_coords = false,
                      const long x = 0) {
    t_jit_err err = JIT_ERR_NONE;
    t_jit_matrix_info minfo;
    t_object* tmp_matrix = nullptr;
    
    minfo = target_info;
    minfo.type = _jit_sym_float64;
    minfo.flags = 0;
    
    switch(mode) {
        case 0:
            if(target_info.dimcount == 1) {
                minfo.dimcount = 1;
                minfo.dim[0] = target_info.dim[0];
                minfo.dim[1] = 1;
            } else {
                minfo.dimcount = 2;
                minfo.dim[0] = target_info.dim[0];
                minfo.dim[1] = target_info.dim[1];;
                //(std::cerr << "invalid dimcount requested" << std::endl);
            }
            break;
            
        case 1:
            minfo.planecount = 1;
            minfo.dimcount = 1;
            minfo.dim[0] = 1;
            minfo.dim[1] = arma.n_elem;
            break;
            
        case 2:
            minfo.planecount = 1;
            minfo.dimcount = 2;
            minfo.dim[0] = arma.n_elem;
            minfo.dim[1] = 1;
            break;
            
        default:
            (std::cerr << "invalid mode for object" << std::endl);
            break;
    }
    
    tmp_matrix = static_cast<t_object*>(max::jit_object_new(_jit_sym_jit_matrix,&minfo));
    if(!tmp_matrix) {
        (std::cerr << "could not create matrix" << std::endl);
        return jitter_matrix;
    }
    
    err = fill_jit_matrix<double, arma::Row<double>>(tmp_matrix, arma, mode, is_coords, x);
    
    err = (t_jit_err)max::object_method(jitter_matrix,_jit_sym_setinfo,&minfo);
    err = (t_jit_err)max::object_method(jitter_matrix,max::_jit_sym_frommatrix,tmp_matrix,nullptr);
    jit_object_free(tmp_matrix);
    return jitter_matrix;
}


t_object* arma_to_jit(const int mode,
                      const arma::Row<size_t>& arma,
                      t_object *jitter_matrix,
                      t_jit_matrix_info& target_info,
                      const bool is_coords = false,
                      const long x = 0) {
    t_jit_err err = JIT_ERR_NONE;
    t_jit_matrix_info minfo;
    t_object* tmp_matrix = nullptr;
    
    minfo.type = max::_jit_sym_long;
    minfo.flags = 0;
    
    
    switch(mode) {
        case 0:
            if(target_info.dimcount == 1) {
                minfo.dimcount = 1;
                minfo.dim[0] = target_info.dim[0];
                minfo.dim[1] = 1;
                minfo.planecount = target_info.planecount;
            } else if(target_info.dimcount == 2) {
                minfo.dimcount = 2;
                minfo.dim[0] = target_info.dim[0];
                minfo.dim[1] = target_info.dim[1];
                minfo.planecount = target_info.planecount;
            } else {
                //should not happen
                //cerr << "invalid dimcount requested" << endl;
            }
            break;
            
        case 1:
            minfo.planecount = 1;
            minfo.dimcount = 2;
            minfo.dim[0] = 1;
            minfo.dim[1] = arma.n_elem;
            break;
            
        case 2:
            minfo.planecount = 1;
            minfo.dimcount = 1;
            minfo.dim[0] = arma.n_elem;
            minfo.dim[1] = 1;
            break;
            
        default:
            break;
            //should not happen
            //cerr << "invalid mode for object" << endl;
    }
    
    //create temporary matrix
    tmp_matrix = static_cast<t_object*>(max::jit_object_new(_jit_sym_jit_matrix,&minfo));
    if(!tmp_matrix) {
        //cerr << "could not create matrix" << endl;
        return jitter_matrix;
    }
    
    err = fill_jit_matrix<t_int32, arma::Row<size_t>>(tmp_matrix, arma, mode,is_coords, x);
    
    err = (t_jit_err)max::object_method(jitter_matrix,_jit_sym_setinfo,&minfo);
    err = (t_jit_err)max::object_method(jitter_matrix,max::_jit_sym_frommatrix,tmp_matrix,nullptr);
    
    jit_object_free(tmp_matrix);
    
    
    return jitter_matrix;
    
}

t_object* arma_to_jit(const int mode,
                      arma::Mat<double>& arma,
                      t_object *jitter_matrix,
                      t_jit_matrix_info& target_info,
                      const bool is_coords = false,
                      const long x = 0) {
    t_jit_err err = JIT_ERR_NONE;
    t_jit_matrix_info minfo;
    t_object* tmp_matrix = nullptr;
    
    minfo.type = _jit_sym_float64;
    minfo.flags = 0;

    minfo.dimcount = target_info.dimcount;
    switch(mode) {
        case 0:
            if(target_info.dimcount == 1) {
                minfo.dim[0] = target_info.dim[0];
                minfo.dim[1] = 1;
                minfo.dim[2] = 0;
                minfo.planecount = target_info.planecount;
            } else if(target_info.dimcount == 2) {
                minfo.dim[0] = target_info.dim[0];
                minfo.dim[1] = target_info.dim[1];
                minfo.dim[2] = 0;
                minfo.planecount = target_info.planecount;
            } else if(target_info.dimcount == 3) {
                minfo.dim[0] = target_info.dim[0];
                minfo.dim[1] = target_info.dim[1];
                minfo.dim[2] = target_info.dim[2];
                minfo.planecount = target_info.planecount;
            } else {
                
            }
            break;
        case 1:
            minfo.planecount = 1;
            minfo.dimcount = 2;
            minfo.dim[0] = arma.n_rows;
            minfo.dim[1] = arma.n_cols;
            break;
        case 2:
            minfo.planecount = 1;
            minfo.dimcount = 2;
            minfo.dim[0] = arma.n_cols;
            minfo.dim[1] = arma.n_rows;
            break;
            
        default:
            break;
    }
    
    //create temporary matrix
    tmp_matrix = static_cast<t_object*>(max::jit_object_new(_jit_sym_jit_matrix,&minfo));
    if(!tmp_matrix) {
        //cerr << "could not create matrix" << endl;
        return jitter_matrix;
    }
    
    err = fill_jit_matrix<double, arma::Mat<double>>(tmp_matrix, arma, mode, is_coords, x);
    
    err = (t_jit_err)max::object_method(jitter_matrix,_jit_sym_setinfo,&minfo);
    err = (t_jit_err)max::object_method(jitter_matrix,max::_jit_sym_frommatrix,tmp_matrix,nullptr);
    jit_object_free(tmp_matrix);
    return jitter_matrix;
}

t_object* arma_to_jit(const int mode,
                      const arma::Mat<size_t>& arma,
                      t_object *jitter_matrix,
                      t_jit_matrix_info& target_info,
                      const bool is_coords = false,
                      const long x = 0) {
    t_jit_err err = JIT_ERR_NONE;
    t_jit_matrix_info minfo;
    t_object* tmp_matrix = nullptr;
    
    
    minfo.type = max::_jit_sym_long;
    minfo.flags = 0;
    
    minfo.dimcount = target_info.dimcount;
    
    switch(mode) {
        case 0:
            
            if(target_info.dimcount == 1) {
                minfo.dim[0] = target_info.dim[0];
                minfo.dim[1] = 1;
                minfo.dim[2] = 0;//could be 1
                minfo.planecount = target_info.planecount;
            } else if(target_info.dimcount == 2) {
                minfo.dim[0] = target_info.dim[0];
                minfo.dim[1] = target_info.dim[1];
                minfo.dim[2] = 0;
                minfo.planecount = target_info.planecount;
            } else if(target_info.dimcount == 3) {
                minfo.dim[0] = target_info.dim[0];
                minfo.dim[1] = target_info.dim[1];
                minfo.dim[2] = target_info.dim[2];
                minfo.planecount = target_info.planecount;
            } else {
                
            }
            break;
        case 1:
            minfo.planecount = target_info.planecount;
            minfo.dimcount = 2;
            minfo.dim[0] = arma.n_rows;
            minfo.dim[1] = arma.n_cols;
            break;
        case 2:
            minfo.planecount = target_info.planecount;
            minfo.dimcount = 2;
            minfo.dim[0] = arma.n_cols;
            minfo.dim[1] = arma.n_rows;
            break;
            
        default:
            break;
    }
    
    //create temporary matrix
    tmp_matrix = static_cast<t_object*>(max::jit_object_new(_jit_sym_jit_matrix,&minfo));
    if(!tmp_matrix) {
        //cerr << "could not create matrix" << endl;
        return jitter_matrix;
    }
    
    err = fill_jit_matrix<t_int32, arma::Mat<size_t>>(tmp_matrix, arma, mode, is_coords, x);
    
    err = (t_jit_err)max::object_method(jitter_matrix,_jit_sym_setinfo,&minfo);
    err = (t_jit_err)max::object_method(jitter_matrix,max::_jit_sym_frommatrix,tmp_matrix,nullptr);
    jit_object_free(tmp_matrix);
    return jitter_matrix;
}


arma::mat& jit_to_arma(const int mode,
                       const t_object *jitter_matrix,
                       arma::Mat<double>& arma_matrix ) {
    /* this is not optimized for speed */
    t_jit_matrix_info minfo;
    t_jit_err err = JIT_ERR_NONE;
    uchar *dataptr = nullptr;
    uchar *p = nullptr;
    uchar *p1 = nullptr;
    object_method(jitter_matrix, _jit_sym_getinfo, &minfo);
    err = (t_jit_err)max::object_method(jitter_matrix, _jit_sym_getdata, &dataptr);
    
    if(minfo.type != _jit_sym_float64) {
        //ERROR FOR NOW
        return arma_matrix;
    }
    
    if(minfo.dimcount == 1) { minfo.dim[1] = 1;} //for loops
    
    arma_matrix.set_size(minfo.planecount , minfo.dim[0]*minfo.dim[1]);
    
    long alem = 0;
    
    switch(mode) {
        case 0:
            for(auto jcol=0;jcol<minfo.dim[0];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[0]);
                for(auto jrow=0;jrow<minfo.dim[1];jrow++) {
                    p1 = p + (jrow*minfo.dimstride[1]);
                    for(auto jplane=0;jplane<minfo.planecount;jplane++) {
                        arma_matrix(alem++) = *(double*)p1;
                        p1 += sizeof(double);
                    }
                }
            }
            break;
            
        case 1:
            arma_matrix.set_size(minfo.dim[0], minfo.dim[1]);
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    arma_matrix(jrow, jcol) = *(double*)p;
                    p += sizeof(double);
                }
            }
            break;
            
        case 2:
            //TODO: compare doing same as in mode 1 and then
            //transposing the arma mat
            // Will catch earlier if not a 1d or 2d matrix and not 1 plane
            arma_matrix.set_size(minfo.dim[1],  minfo.dim[0]);
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    arma_matrix(jcol, jrow) = *(double*)p;
                    p += sizeof(double);
                }
            }
        default:
            break;
            
    }
    return arma_matrix;
}

arma::Mat<size_t>& jit_to_arma(const int mode,
                               const t_object *jitter_matrix,
                               arma::Mat<size_t>& arma_matrix ) {
    /* this is not optimized for speed */
    t_jit_matrix_info minfo;
    t_jit_err err = JIT_ERR_NONE;
    uchar *dataptr = nullptr;
    uchar *p = nullptr;
    object_method((t_object*)jitter_matrix, _jit_sym_getinfo, &minfo);
    err = (t_jit_err)max::object_method(jitter_matrix, _jit_sym_getdata, &dataptr);
    
    if(minfo.type != max::_jit_sym_long) {
        //ERROR FOR NOW
        return arma_matrix;
    }
    
    if(minfo.dimcount == 1) { minfo.dim[1] = 1;} //for loops
    
    arma_matrix.set_size(minfo.planecount , minfo.dim[0]*minfo.dim[1]);
    
    long acol = 0;
    
    
    switch(mode) {
        case 0:
            arma_matrix.resize(minfo.planecount,  minfo.dim[0]*minfo.dim[1]);
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    size_t *colptr = arma_matrix.colptr(acol++);
                    for(auto jplane=0;jplane<minfo.planecount;jplane++) {
                        *colptr++ = *(t_int32*)p;
                        p += sizeof(t_int32);
                    }
                }
            }
            break;
            
        case 1:
            arma_matrix.resize(minfo.dim[0], minfo.dim[1]);
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    arma_matrix(jrow, jcol) = *(t_int32*)p;
                    p += sizeof(t_int32);
                }
            }
            break;
            
        case 2:
            //TODO: compare doing same as in mode 1 and then
            //transposing the arma mat
            // Will catch earlier if not a 1d or 2d matrix and not 1 plane
            arma_matrix.resize(minfo.dim[1],  minfo.dim[0]);
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    arma_matrix(jcol, jrow) = *(t_int32*)p;
                    p += sizeof(t_int32);
                }
            }
        default:
            break;
            
    }
    return arma_matrix;
}

arma::Row<size_t>& jit_to_arma(const int mode,
                               t_object *jitter_matrix,
                               arma::Row<size_t>& arma_row ) {
    /* this is not optimized for speed */
    t_jit_matrix_info minfo;
    t_jit_err err = JIT_ERR_NONE;
    uchar *dataptr = nullptr;
    uchar *p = nullptr;
    object_method(jitter_matrix, _jit_sym_getinfo, &minfo);
    err = (t_jit_err)max::object_method(jitter_matrix,max:: _jit_sym_getdata, &dataptr);
    
    if(minfo.type != max::_jit_sym_long) {
        //ERROR FOR NOW
        return arma_row;
    }
    
    //think that the plane should be limited to 1
    
    if(minfo.dimcount == 1) { minfo.dim[1] = 1;} //for loops
    
    arma_row.set_size(minfo.dim[0]*minfo.dim[1]);
    
    long arow = 0;
    
    switch(mode) {
        case 0:
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    for(auto jplane=0;jplane<minfo.planecount;jplane++) {
                        arma_row(arow++) = *(t_int32*)p;
                        p += sizeof(t_int32);
                    }
                }
            }
            break;
            
        case 1:
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    arma_row(arow++) = *(t_int32*)p;
                    p += sizeof(t_int32);
                }
            }
            break;
            
        case 2:
            //TODO: compare doing same as in mode 1 and then
            //transposing the arma mat
            // Will catch earlier if not a 1d or 2d matrix and not 1 plane
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    arma_row(arow++) = *(t_int32*)p;
                    p += sizeof(t_int32);
                }
            }
        default:
            break;
            
    }
    return arma_row;
}

/// TODO: The jit matrix iter is wrong here!!!
arma::Col<arma::uword>& jit_to_arma(const int mode,
                                    const t_object *jitter_matrix,
                                    arma::Col<arma::uword>& arma_col) {
    /* this is not optimized for speed */
    t_jit_matrix_info minfo;
    t_jit_err err = JIT_ERR_NONE;
    uchar *dataptr = nullptr;
    uchar *p = nullptr;
    uchar *p1 = nullptr;
    object_method(jitter_matrix, _jit_sym_getinfo, &minfo);
    err = (t_jit_err)max::object_method(jitter_matrix,max::_jit_sym_getdata, &dataptr);
    
    if(minfo.type != max::_jit_sym_long) {
        //ERROR FOR NOW
        return arma_col;
    }
    
    if(minfo.dimcount == 1) { minfo.dim[1] = 1;}
    
    arma_col.set_size(minfo.dim[0]*minfo.dim[1]);
    
    long acol = 0;
    long alem = 0;
    // t_int32 pos = 0;
    switch(mode) {
        case 0:
            
            for(auto jcol=0;jcol<minfo.dim[0];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[0]);
                for(auto jrow=0;jrow<minfo.dim[1];jrow++) {
                    p1 = p + (jrow*minfo.dimstride[1]);
                    for(auto jplane=0;jplane<minfo.planecount;jplane++) {
                        arma_col(alem++) = *(t_int32*)p1;
                        p1 += sizeof(t_int32);
                    }
                }
            }
            
            break;
            
        case 1:
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    arma_col(acol++) = *(t_int32*)p;
                    p += sizeof(t_int32);
                }
            }
            break;
            
        case 2:
            //TODO: compare doing same as in mode 1 and then
            //transposing the arma mat
            // Will catch earlier if not a 1d or 2d matrix and not 1 plane
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    arma_col(acol++) = *(t_int32*)p;
                    p += sizeof(t_int32);
                }
            }
        default:
            break;
            
    }
    return arma_col;
}




arma::Col<arma::uword>& jit_to_arma_limit(const int mode,
                                          const t_object *jitter_matrix,
                                          arma::Col<arma::uword>& arma_col,
                                          t_int32 max_x,
                                          t_int32 max_y) {
    /* this is not optimized for speed */
    t_jit_matrix_info minfo;
    t_jit_err err = JIT_ERR_NONE;
    uchar *dataptr = nullptr;
    uchar *p = nullptr;
    uchar *p1 = nullptr;
    object_method(jitter_matrix, _jit_sym_getinfo, &minfo);
    err = (t_jit_err)max::object_method(jitter_matrix,max::_jit_sym_getdata, &dataptr);
    
    if(minfo.type != max::_jit_sym_long) {
        //ERROR FOR NOW
        return arma_col;
    }
    
    if(minfo.dimcount == 1) { minfo.dim[1] = 1;}
    
    arma_col.set_size(minfo.dim[0]*minfo.dim[1]);
    
    long acol = 0;
    long alem = 0;
    // t_int32 pos = 0;
    switch(mode) {
        case 0:
            if(minfo.planecount == 1) {
                t_int32 m = (max_x * max_y)-1;
                for(auto jcol=0;jcol<minfo.dim[0];jcol++) {
                    p = dataptr + (jcol*minfo.dimstride[0]);
                    for(auto jrow=0;jrow<minfo.dim[1];jrow++) {
                        p1 = p + (jrow*minfo.dimstride[1]);
                        for(auto jplane=0;jplane<minfo.planecount;jplane++) {
                            t_int32 d = *(t_int32*)p1;
                            d = MIN_CLAMP(d, 0, m);
                            arma_col(alem++) = d;
                            p1 += sizeof(t_int32);
                        }
                    }
                }
            } else if (minfo.planecount == 2) {
                for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                    p = dataptr + (jcol*minfo.dimstride[1]);
                    for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                        
                        t_int32 xpos = *(t_int32*)p;
                        xpos = MIN_CLAMP(xpos, 0, max_x-1);
                        
                        p += sizeof(t_int32);
                        t_int32 ypos = *(t_int32*)p;
                        ypos = MIN_CLAMP(ypos, 0, max_y-1);
                        p += sizeof(t_int32);
                        
                        ypos *= max_x;//
                        arma_col(acol++) =  xpos + ypos;
                    }
                }
            } else {
                //ERROR
            }
            break;
            
        case 1:
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    arma_col(acol++) = MIN_CLAMP(*(t_int32*)p,0, (max_y)-1);
                    p += sizeof(t_int32);
                }
            }
            break;
            
        case 2:
            //TODO: compare doing same as in mode 1 and then
            //transposing the arma mat
            // Will catch earlier if not a 1d or 2d matrix and not 1 plane
            for(auto jcol=0;jcol<minfo.dim[1];jcol++) {
                p = dataptr + (jcol*minfo.dimstride[1]);
                for(auto jrow=0;jrow<minfo.dim[0];jrow++) {
                    arma_col(acol++) = MIN_CLAMP(*(t_int32*)p,0, (max_x)-1);
                    p += sizeof(t_int32);
                }
            }
        default:
            break;
            
    }
    return arma_col;
}

