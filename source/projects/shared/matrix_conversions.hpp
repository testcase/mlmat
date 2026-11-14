/// @file
/// @ingroup
/// @copyright Copyright 2018 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.

#pragma once
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>

#include "c74_min.h"

#include <algorithm>

// ---------------------------------------------------------------------------
// conversion notes
//
// The functions below move data between armadillo matrices and jit matrices.
// jit matrices are normally allocated in a standard "packed" layout
// (plane-interleaved cells, each dimstride the product of the dimensions
// below it). When the layout is packed the copies are done with a single
// linear pass (memcpy-able); for any other dimstride the element-by-element
// fallback loops are used, which accept arbitrary strides.
//
// Loop counters, dimensions and byte offsets are held in size_t locals
// (the jit API fields are long, which is only 32 bits on Windows).
// ---------------------------------------------------------------------------


namespace mlmat {

/// Is the matrix data in Jitter's standard packed layout?
/// (dimstride[0] == planecount * elem_size, and each higher dimstride the
/// product of the dimensions below it.)
inline bool jit_is_packed(const c74::max::t_jit_matrix_info& minfo, size_t elem_size)
{
    if (minfo.dimcount < 1 || minfo.planecount < 1)
        return false;

    size_t stride = size_t(minfo.planecount) * elem_size;
    if (size_t(minfo.dimstride[0]) != stride)
        return false;

    for (long i = 1; i < minfo.dimcount; ++i) {
        stride *= size_t(minfo.dim[i - 1]);
        if (size_t(minfo.dimstride[i]) != stride)
            return false;
    }
    return true;
}

/// Copy n elements from src to dst. Falls back to a converting loop when
/// the element sizes differ (e.g. jit int32 <-> armadillo size_t).
template <typename SrcElem, typename DstElem>
inline void copy_elements(const SrcElem* src, DstElem* dst, size_t n)
{
    if constexpr (sizeof(SrcElem) == sizeof(DstElem)) {
        std::copy_n(src, n, dst);
    } else {
        for (size_t i = 0; i < n; ++i)
            dst[i] = DstElem(src[i]);
    }
}

inline size_t trailing_zeros(size_t v)
{
    size_t s = 0;
    while ((v & 1) == 0) {
        v >>= 1;
        ++s;
    }
    return s;
}

/// Core of fill_jit_matrix: writes arma_matrix into a jit matrix data
/// buffer according to minfo and the mlmat mode. See the object
/// documentation for the meaning of the modes.
template <typename M, typename A>
c74::max::t_jit_err fill_matrix_data(const c74::max::t_jit_matrix_info& minfo,
                                     c74::max::uchar* dataptr,
                                     const A& arma_matrix,
                                     int mode,
                                     bool is_coords,
                                     size_t x)
{
    if (dataptr == nullptr)
        return c74::max::JIT_ERR_INVALID_PTR;

    const size_t stepsize = sizeof(M);
    const size_t d0 = size_t(minfo.dim[0]);
    const size_t d1 = size_t(minfo.dim[1]);
    const size_t d2 = size_t(minfo.dim[2]);
    const size_t pc = size_t(minfo.planecount);
    const size_t s0 = size_t(minfo.dimstride[0]);
    const size_t s1 = size_t(minfo.dimstride[1]);
    const size_t s2 = size_t(minfo.dimstride[2]);

    // for the coordinate conversions: divide/modulo by x, using shift/mask
    // when x is a power of two and the value is non-negative
    const bool x_is_pow2 = (x != 0) && ((x & (x - 1)) == 0);
    const size_t x_shift = x_is_pow2 ? trailing_zeros(x) : 0;
    const long xl = long(x);

    switch (mode) {
        case 0: {
            if (is_coords && (minfo.dimcount == 1 || minfo.dimcount == 2)) {
                // linear index -> (x, y) coordinate pairs written to planes 0/1
                const auto* src = arma_matrix.memptr();
                size_t aelem = 0;
                for (size_t jslice = 0; jslice < d2; ++jslice) {
                    c74::max::uchar* p = dataptr + jslice * s2;
                    for (size_t jcol = 0; jcol < d1; ++jcol) {
                        c74::max::uchar* p2 = p + jcol * s1;
                        for (size_t jrow = 0; jrow < d0; ++jrow) {
                            c74::max::uchar* p1 = p2 + jrow * s0;
                            const M pos = M(src[aelem++]);
                            const long ip = long(pos);
                            long mx, my;
                            if (x_is_pow2 && ip >= 0) {
                                mx = ip & (xl - 1);
                                my = ip >> x_shift;
                            } else {
                                mx = ip % xl;
                                my = ip / xl;
                            }
                            *(M*)p1 = M(mx);
                            p1 += stepsize;
                            *(M*)p1 = M(my);
                        }
                    }
                }
            } else if (is_coords && (minfo.dimcount == 3)) {
                // per-slice source rows, converted to (x, y) coordinate pairs
                for (size_t jslice = 0; jslice < d2; ++jslice) {
                    c74::max::uchar* p = dataptr + jslice * s2;
                    size_t aelem = 0;
                    for (size_t jcol = 0; jcol < d1; ++jcol) {
                        c74::max::uchar* p2 = p + jcol * s1;
                        for (size_t jrow = 0; jrow < d0; ++jrow) {
                            c74::max::uchar* p1 = p2 + jrow * s0;
                            const M pos = M(arma_matrix(jslice, aelem++));
                            const long ip = long(pos);
                            long mx, my;
                            if (x_is_pow2 && ip >= 0) {
                                mx = ip & (xl - 1);
                                my = ip >> x_shift;
                            } else {
                                mx = ip % xl;
                                my = ip / xl;
                            }
                            *(M*)p1 = M(mx);
                            p1 += stepsize;
                            *(M*)p1 = M(my);
                        }
                    }
                }
            } else {
                if (jit_is_packed(minfo, stepsize)) {
                    // packed cells interleave the planes exactly like the
                    // column-major (planecount x cells) armadillo layout,
                    // so the whole slice is one linear copy. (the original
                    // loop restarts the source for every slice, so do the
                    // same for 3d matrices)
                    const size_t per_slice = std::min(pc * d0 * d1, size_t(arma_matrix.n_elem));
                    const auto* src = arma_matrix.memptr();
                    for (size_t jslice = 0; jslice < d2; ++jslice) {
                        copy_elements(src, (M*)(dataptr + jslice * s2), per_slice);
                    }
                } else {
                    const auto* src = arma_matrix.memptr();
                    for (size_t jslice = 0; jslice < d2; ++jslice) {
                        c74::max::uchar* p = dataptr + jslice * s2;
                        size_t aelem = 0;
                        for (size_t jcol = 0; jcol < d1; ++jcol) {
                            c74::max::uchar* p2 = p + jcol * s1;
                            for (size_t jrow = 0; jrow < d0; ++jrow) {
                                c74::max::uchar* p1 = p2 + jrow * s0;
                                for (size_t jplane = 0; jplane < pc; ++jplane) {
                                    *(M*)p1 = M(src[aelem++]);
                                    p1 += stepsize;
                                }
                            }
                        }
                    }
                }
            }
            break;
        }

        case 1: {
            // (the walk below assumes one plane per cell, so the linear fast
            // path only applies to single-plane matrices)
            if (pc == 1 && jit_is_packed(minfo, stepsize)) {
                const size_t total = std::min(d0 * d1, size_t(arma_matrix.n_elem));
                copy_elements(arma_matrix.memptr(), (M*)dataptr, total);
            } else {
                size_t aelem = 0;
                for (size_t jcol = 0; jcol < d1; ++jcol) {
                    c74::max::uchar* p = dataptr + jcol * s1;
                    for (size_t jrow = 0; jrow < d0; ++jrow) {
                        *(M*)p = M(arma_matrix(aelem++));
                        p += stepsize;
                    }
                }
            }
            break;
        }

        case 2: {
            // (single-plane matrices only, as above)
            if (pc == 1 && jit_is_packed(minfo, stepsize)) {
                // in the packed layout the jit matrix holds the transpose
                // of the sequential arma data; armadillo's blocked transpose
                // plus a linear copy beats strided element writes
                arma::Mat<typename A::elem_type> tmp = arma_matrix.t();
                const size_t total = std::min(d0 * d1, size_t(tmp.n_elem));
                copy_elements(tmp.memptr(), (M*)dataptr, total);
            } else {
                size_t aelem = 0;
                for (size_t jrow = 0; jrow < d0; ++jrow) {
                    c74::max::uchar* p1 = dataptr + jrow * s0;
                    for (size_t jcol = 0; jcol < d1; ++jcol) {
                        *(M*)p1 = M(arma_matrix(aelem++));
                        p1 += s1;
                    }
                }
            }
            break;
        }

        default:
            (std::cerr << "could not create matrix" << std::endl);
            break;
    }
    return c74::max::JIT_ERR_NONE;
}

/// Core of jit_to_arma for arma::Mat: reads a jit matrix data buffer into a
/// column-major armadillo matrix according to the mlmat mode.
/// JT is the jit element type (double or t_int32).
template <typename JT, typename E>
void read_matrix_data(const c74::max::t_jit_matrix_info& minfo,
                      const c74::max::uchar* dataptr,
                      int mode,
                      arma::Mat<E>& arma_matrix)
{
    if (dataptr == nullptr)
        return;

    const size_t es = sizeof(JT);
    const size_t d0 = size_t(minfo.dim[0]);
    const size_t d1 = (minfo.dimcount == 1) ? 1 : size_t(minfo.dim[1]);
    const size_t pc = size_t(minfo.planecount);
    const size_t s0 = size_t(minfo.dimstride[0]);
    const size_t s1 = size_t(minfo.dimstride[1]);

    switch (mode) {
        case 0: {
            arma_matrix.set_size(pc, d0 * d1);
            if (jit_is_packed(minfo, es)) {
                // packed plane-interleaved cells are exactly the column-major
                // (planecount x cells) armadillo layout
                copy_elements((const JT*)dataptr, arma_matrix.memptr(), pc * d0 * d1);
            } else {
                size_t alem = 0;
                for (size_t jcol = 0; jcol < d1; ++jcol) {
                    const c74::max::uchar* pcol = dataptr + jcol * s1;
                    for (size_t jrow = 0; jrow < d0; ++jrow) {
                        const c74::max::uchar* p1 = pcol + jrow * s0;
                        for (size_t jplane = 0; jplane < pc; ++jplane) {
                            arma_matrix(alem++) = E(*(const JT*)(p1 + jplane * es));
                        }
                    }
                }
            }
            break;
        }

        case 1: {
            arma_matrix.set_size(d0, d1);
            // (the fallback walk assumes one plane per cell)
            if (pc == 1 && jit_is_packed(minfo, es)) {
                copy_elements((const JT*)dataptr, arma_matrix.memptr(), d0 * d1);
            } else {
                for (size_t jcol = 0; jcol < d1; ++jcol) {
                    const JT* p = (const JT*)(dataptr + jcol * s1);
                    for (size_t jrow = 0; jrow < d0; ++jrow) {
                        arma_matrix(jrow, jcol) = E(p[jrow]);
                    }
                }
            }
            break;
        }

        case 2: {
            // (single-plane matrices only, as above)
            if (pc == 1 && jit_is_packed(minfo, es)) {
                // read linearly (mode 1 layout), then transpose once
                arma::Mat<E> tmp(d0, d1, arma::fill::none);
                copy_elements((const JT*)dataptr, tmp.memptr(), d0 * d1);
                arma_matrix = tmp.t();
            } else {
                arma_matrix.set_size(d1, d0);
                for (size_t jcol = 0; jcol < d1; ++jcol) {
                    const JT* p = (const JT*)(dataptr + jcol * s1);
                    for (size_t jrow = 0; jrow < d0; ++jrow) {
                        arma_matrix(jcol, jrow) = E(p[jrow]);
                    }
                }
            }
            break;
        }

        default:
            break;
    }
}

/// Core of jit_to_arma for arma::Row: reads a jit matrix data buffer into a
/// row vector, flattening in the same order as the original implementation.
template <typename JT, typename E>
void read_row_data(const c74::max::t_jit_matrix_info& minfo,
                   const c74::max::uchar* dataptr,
                   int mode,
                   arma::Row<E>& arma_row)
{
    if (dataptr == nullptr)
        return;

    const size_t es = sizeof(JT);
    const size_t d0 = size_t(minfo.dim[0]);
    const size_t d1 = (minfo.dimcount == 1) ? 1 : size_t(minfo.dim[1]);
    const size_t pc = size_t(minfo.planecount);
    const size_t s0 = size_t(minfo.dimstride[0]);
    const size_t s1 = size_t(minfo.dimstride[1]);

    switch (mode) {
        case 0: {
            // one entry per plane of each cell
            arma_row.set_size(pc * d0 * d1);
            if (jit_is_packed(minfo, es)) {
                copy_elements((const JT*)dataptr, arma_row.memptr(), pc * d0 * d1);
            } else {
                size_t arow = 0;
                for (size_t jcol = 0; jcol < d1; ++jcol) {
                    const c74::max::uchar* p = dataptr + jcol * s1;
                    for (size_t jrow = 0; jrow < d0; ++jrow) {
                        const c74::max::uchar* p1 = p + jrow * s0;
                        for (size_t jplane = 0; jplane < pc; ++jplane) {
                            arma_row(arow++) = E(*(const JT*)(p1 + jplane * es));
                        }
                    }
                }
            }
            break;
        }

        case 1:
        case 2: {
            arma_row.set_size(d0 * d1);
            // (the fallback walk assumes one plane per cell)
            if (pc == 1 && jit_is_packed(minfo, es)) {
                copy_elements((const JT*)dataptr, arma_row.memptr(), d0 * d1);
            } else {
                size_t arow = 0;
                for (size_t jcol = 0; jcol < d1; ++jcol) {
                    const c74::max::uchar* p = dataptr + jcol * s1;
                    for (size_t jrow = 0; jrow < d0; ++jrow) {
                        arma_row(arow++) = E(*(const JT*)p);
                        p += es;
                    }
                }
            }
            break;
        }

        default:
            break;
    }
}

} // namespace mlmat


// ---------------------------------------------------------------------------
// public conversion API
// ---------------------------------------------------------------------------

c74::max::t_object* convert_to_float64(c74::max::t_object *matrix, c74::max::t_jit_matrix_info& minfo) {
    c74::max::t_object *m; // destination matrix
    if(minfo.type == c74::max::_jit_sym_float64) {
        return matrix;
    }
    c74::max::t_jit_matrix_info dest_info = minfo;

    dest_info.type = c74::max::_jit_sym_float64;
    m = static_cast<c74::max::t_object*>(c74::max::jit_object_new(c74::max::_jit_sym_jit_matrix,&dest_info));
    c74::max::object_method(m, c74::max::_jit_sym_frommatrix,matrix,NULL);
    return m;
}


c74::max::t_object* convert_to_long(c74::max::t_object *matrix, c74::max::t_jit_matrix_info& minfo) {
    c74::max::t_object *m; // destination matrix

    if(minfo.type == c74::max::_jit_sym_long) {
        return matrix;
    }
    c74::max::t_jit_matrix_info dest_info = minfo;
    dest_info.type = c74::max::_jit_sym_long;
    m = static_cast<c74::max::t_object*>(c74::max::jit_object_new(c74::max::_jit_sym_jit_matrix,&dest_info));
    c74::max::object_method(m, c74::max::_jit_sym_frommatrix,matrix,NULL);

    return m;
}

/// Fills a jit matrix from an armadillo matrix according to the mlmat
/// mode. (The matrix is assumed to be locked by the caller.)
template <typename M, typename A>
c74::max::t_jit_err fill_jit_matrix(c74::max::t_object* jitter_matrix, const A& arma_matrix, int mode, bool is_coords = false, long x = 0) {
    c74::max::t_jit_matrix_info minfo;
    c74::max::uchar *dataptr = nullptr;

    c74::max::object_method(jitter_matrix, c74::max::_jit_sym_getinfo, &minfo);
    c74::max::object_method(jitter_matrix, c74::max::_jit_sym_getdata, &dataptr);

    return mlmat::fill_matrix_data<M>(minfo, dataptr, arma_matrix, mode, is_coords, size_t(x));
}

/// Resizes the destination jit matrix to minfo and fills it directly from
/// the armadillo matrix (no temporary matrix, single copy).
template <typename M, typename A>
c74::max::t_object* arma_to_jit_fill(const A& arma_matrix,
                                     c74::max::t_object* jitter_matrix,
                                     c74::max::t_jit_matrix_info& minfo,
                                     int mode,
                                     bool is_coords,
                                     long x)
{
    c74::max::object_method(jitter_matrix, c74::max::_jit_sym_setinfo, &minfo);

    // fill according to the matrix info as the jit system actually
    // realized it (it normalizes dimcount/dims on setinfo)
    c74::max::t_jit_matrix_info actual_info;
    c74::max::uchar* dataptr = nullptr;
    c74::max::object_method(jitter_matrix, c74::max::_jit_sym_getinfo, &actual_info);
    c74::max::object_method(jitter_matrix, c74::max::_jit_sym_getdata, &dataptr);

    if (dataptr != nullptr) {
        mlmat::fill_matrix_data<M>(actual_info, dataptr, arma_matrix, mode, is_coords, size_t(x));
    }
    return jitter_matrix;
}


/*
 converts a double arma row to a jitter matrix
 mode 0: returns a 1 cell matrix with each entry in the row being a plane
 possible it could hold coordinates that need to be converted to 2d coords. Would it even make sense
 to use 2d ?
 mode 1: returns a 1d matrix along dim[0]
 mode 2: returns a 1d matrix along dim[1]

 */

c74::max::t_object* arma_to_jit(const int mode,
                      const arma::Row<double>& arma,
                        c74::max::t_object *jitter_matrix,
                        c74::max::t_jit_matrix_info& target_info,
                      const bool is_coords = false,
                      const long x = 0) {
    c74::max::t_jit_matrix_info minfo;

    minfo = target_info;
    minfo.type = c74::max::_jit_sym_float64;
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
                minfo.dim[1] = target_info.dim[1];
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

    return arma_to_jit_fill<double>(arma, jitter_matrix, minfo, mode, is_coords, x);
}


c74::max::t_object* arma_to_jit(const int mode,
                                const arma::Row<size_t>& arma,
                                c74::max::t_object *jitter_matrix,
                                c74::max::t_jit_matrix_info& target_info,
                                const bool is_coords = false,
                                const long x = 0) {

    c74::max::t_jit_matrix_info minfo;

    minfo.type = c74::max::_jit_sym_long;
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
    }

    return arma_to_jit_fill<c74::max::t_int32>(arma, jitter_matrix, minfo, mode, is_coords, x);
}

c74::max::t_object* arma_to_jit(const int mode,
                      arma::Mat<double>& arma,
    c74::max::t_object *jitter_matrix,
    c74::max::t_jit_matrix_info& target_info,
                      const bool is_coords = false,
                      const long x = 0) {
    c74::max::t_jit_matrix_info minfo;

    minfo.type = c74::max::_jit_sym_float64;
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

    return arma_to_jit_fill<double>(arma, jitter_matrix, minfo, mode, is_coords, x);
}

c74::max::t_object* arma_to_jit(const int mode,
                      const arma::Mat<size_t>& arma,
    c74::max::t_object *jitter_matrix,
    c74::max::t_jit_matrix_info& target_info,
                      const bool is_coords = false,
                      const long x = 0) {
    c74::max::t_jit_matrix_info minfo;


    minfo.type = c74::max::_jit_sym_long;
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

    return arma_to_jit_fill<c74::max::t_int32>(arma, jitter_matrix, minfo, mode, is_coords, x);
}


arma::mat& jit_to_arma(const int mode,
                       const c74::max::t_object *jitter_matrix,
                       arma::Mat<double>& arma_matrix ) {
    c74::max::t_jit_matrix_info minfo;
    c74::max::uchar *dataptr = nullptr;

    c74::max::object_method((c74::max::t_object*)jitter_matrix, c74::max::_jit_sym_getinfo, &minfo);
    c74::max::object_method((c74::max::t_object*)jitter_matrix, c74::max::_jit_sym_getdata, &dataptr);

    if(minfo.type != c74::max::_jit_sym_float64) {
        //ERROR FOR NOW
        return arma_matrix;
    }

    mlmat::read_matrix_data<double>(minfo, dataptr, mode, arma_matrix);
    return arma_matrix;
}

arma::Mat<size_t>& jit_to_arma(const int mode,
                               const c74::max::t_object *jitter_matrix,
                               arma::Mat<size_t>& arma_matrix ) {
    c74::max::t_jit_matrix_info minfo;
    c74::max::uchar *dataptr = nullptr;

    c74::max::object_method((c74::max::t_object*)jitter_matrix, c74::max::_jit_sym_getinfo, &minfo);
    c74::max::object_method((c74::max::t_object*)jitter_matrix, c74::max::_jit_sym_getdata, &dataptr);

    if(minfo.type != c74::max::_jit_sym_long) {
        //ERROR FOR NOW
        return arma_matrix;
    }

    mlmat::read_matrix_data<c74::max::t_int32>(minfo, dataptr, mode, arma_matrix);
    return arma_matrix;
}

arma::Row<size_t>& jit_to_arma(const int mode,
                                 c74::max::t_object *jitter_matrix,
                               arma::Row<size_t>& arma_row ) {
    c74::max::t_jit_matrix_info minfo;
    c74::max::uchar *dataptr = nullptr;

    c74::max::object_method(jitter_matrix, c74::max::_jit_sym_getinfo, &minfo);
    c74::max::object_method(jitter_matrix, c74::max:: _jit_sym_getdata, &dataptr);

    if(minfo.type != c74::max::_jit_sym_long) {
        //ERROR FOR NOW
        return arma_row;
    }

    mlmat::read_row_data<c74::max::t_int32>(minfo, dataptr, mode, arma_row);
    return arma_row;
}

namespace mlmat {

/// Core of jit_to_arma_limit. JT is the jit element type (t_int32 or
/// double); values are truncated to long and clamped into the reference
/// set bounds before being stored as indices.
template <typename JT>
void read_limited_indices(const c74::max::t_jit_matrix_info& minfo,
                          const c74::max::uchar* dataptr,
                          int mode,
                          arma::Col<arma::uword>& arma_col,
                          long max_x,
                          long max_y)
{
    if (dataptr == nullptr)
        return;

    const size_t es = sizeof(JT);
    const size_t d0 = size_t(minfo.dim[0]);
    const size_t d1 = (minfo.dimcount == 1) ? 1 : size_t(minfo.dim[1]);
    const size_t pc = size_t(minfo.planecount);
    const size_t s0 = size_t(minfo.dimstride[0]);
    const size_t s1 = size_t(minfo.dimstride[1]);
    const long mx_max = max_x - 1;
    const long my_max = max_y - 1;
    const long m_max = max_x * max_y - 1;

    switch(mode) {
        case 0:
            if(pc == 1) {
                arma_col.set_size(pc * d0 * d1);
                if (mlmat::jit_is_packed(minfo, es)) {
                    const JT* src = (const JT*)dataptr;
                    const size_t total = pc * d0 * d1;
                    for (size_t i = 0; i < total; ++i) {
                        arma_col(i) = arma::uword(std::clamp(long(src[i]), long(0), m_max));
                    }
                } else {
                    size_t alem = 0;
                    for (size_t jcol = 0; jcol < d1; ++jcol) {
                        const c74::max::uchar* p = dataptr + jcol * s1;
                        for (size_t jrow = 0; jrow < d0; ++jrow) {
                            const c74::max::uchar* p1 = p + jrow * s0;
                            for (size_t jplane = 0; jplane < pc; ++jplane) {
                                const long d = long(*(const JT*)(p1 + jplane * es));
                                arma_col(alem++) = arma::uword(std::clamp(d, long(0), m_max));
                                p1 += es;
                            }
                        }
                    }
                }
            } else if (pc == 2) {
                // 2d coordinates clamped into the reference set bounds.
                // (as before, this assumes fully packed data)
                arma_col.set_size(d0 * d1);
                size_t acol = 0;
                const JT* p = (const JT*)dataptr;
                for (size_t i = 0; i < d0 * d1; ++i) {
                    const long xpos = std::clamp(long(p[2 * i]), long(0), mx_max);
                    const long ypos = std::clamp(long(p[2 * i + 1]), long(0), my_max);
                    arma_col(acol++) = arma::uword(xpos) + arma::uword(ypos) * arma::uword(max_x);
                }
            } else {
                //ERROR
            }
            break;

        case 1: {
            arma_col.set_size(d0 * d1);
            if (mlmat::jit_is_packed(minfo, es)) {
                const JT* src = (const JT*)dataptr;
                const size_t total = d0 * d1;
                for (size_t i = 0; i < total; ++i) {
                    arma_col(i) = arma::uword(std::clamp(long(src[i]), long(0), my_max));
                }
            } else {
                size_t acol = 0;
                for (size_t jcol = 0; jcol < d1; ++jcol) {
                    const c74::max::uchar* p = dataptr + jcol * s1;
                    for (size_t jrow = 0; jrow < d0; ++jrow) {
                        const long tmp = long(*(const JT*)p);
                        arma_col(acol++) = arma::uword(std::clamp(tmp, long(0), my_max));
                        p += es;
                    }
                }
            }
            break;
        }

        case 2: {
            arma_col.set_size(d0 * d1);
            if (mlmat::jit_is_packed(minfo, es)) {
                const JT* src = (const JT*)dataptr;
                const size_t total = d0 * d1;
                for (size_t i = 0; i < total; ++i) {
                    arma_col(i) = arma::uword(std::clamp(long(src[i]), long(0), mx_max));
                }
            } else {
                size_t acol = 0;
                for (size_t jcol = 0; jcol < d1; ++jcol) {
                    const c74::max::uchar* p = dataptr + jcol * s1;
                    for (size_t jrow = 0; jrow < d0; ++jrow) {
                        const long tmp = long(*(const JT*)p);
                        arma_col(acol++) = arma::uword(std::clamp(tmp, long(0), mx_max));
                        p += es;
                    }
                }
            }
            break;
        }

        default:
            break;
    }
}

} // namespace mlmat


arma::Col<arma::uword>& jit_to_arma_limit(const int mode,
                                          const c74::max::t_object *jitter_matrix,
                                          arma::Col<arma::uword>& arma_col,
                                 c74::max::t_int32 max_x,
                                    c74::max::t_int32 max_y) {
    c74::max::t_jit_matrix_info minfo;
    c74::max::uchar *dataptr = nullptr;

    c74::max::object_method((c74::max::t_object*)jitter_matrix, c74::max::_jit_sym_getinfo, &minfo);
    c74::max::object_method((c74::max::t_object*)jitter_matrix, c74::max::_jit_sym_getdata, &dataptr);

    if(minfo.type == c74::max::_jit_sym_long) {
        mlmat::read_limited_indices<c74::max::t_int32>(minfo, dataptr, mode, arma_col, long(max_x), long(max_y));
    } else if(minfo.type == c74::max::_jit_sym_float64) {
        // float64 coordinates (e.g. from a single-type float64 mop) are
        // read directly, avoiding a conversion matrix. values are
        // truncated, so integral coordinates behave identically to the
        // long path
        mlmat::read_limited_indices<double>(minfo, dataptr, mode, arma_col, long(max_x), long(max_y));
    } else {
        //ERROR FOR NOW
        return arma_col;
    }

    return arma_col;
}
