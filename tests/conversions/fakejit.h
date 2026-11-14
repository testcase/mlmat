// Fake Jitter implementation for the conversion parity harness.
#pragma once
#include "c74_min.h"
#include <cstring>
#include <vector>

namespace fakejit {

// In the stub a "t_object*" is actually a FakeMatrix*. The stub mimics how
// jit_matrix realizes an info struct: dims beyond dimcount are normalized
// to 1, strides are computed for the packed layout, and the buffer holds
// planecount * prod(dims) elements.
struct FakeMatrix {
    c74::max::t_jit_matrix_info info;
    std::vector<c74::max::uchar> data;

    size_t elem_size() const
    {
        return (info.type == c74::max::jit_sym_float64) ? 8 : 4;
    }

    size_t total_bytes() const
    {
        size_t total = size_t(info.planecount) * elem_size();
        for (long i = 0; i < info.dimcount; ++i)
            total *= size_t(info.dim[i]);
        return total;
    }

    void realize()
    {
        for (long i = info.dimcount; i < 32; ++i)
            info.dim[i] = 1;
        long stride = info.planecount * long(elem_size());
        info.dimstride[0] = stride;
        for (long i = 1; i < 32; ++i) {
            stride *= info.dim[i - 1];
            info.dimstride[i] = stride;
        }
        info.size = long(total_bytes());
        data.assign(total_bytes(), 0xCD); // poison, to catch unwritten cells
    }

    // place a value at (jrow, jcol[, jslice], plane) honoring the current strides
    void set(size_t jrow, size_t jcol, size_t jslice, size_t jplane, double v)
    {
        size_t es = elem_size();
        size_t offset = jslice * size_t(info.dimstride[2]) +
                        jcol * size_t(info.dimstride[1]) +
                        jrow * size_t(info.dimstride[0]) + jplane * es;
        if (es == 8)
            memcpy(data.data() + offset, &v, 8);
        else {
            c74::max::t_int32 iv = c74::max::t_int32(v);
            memcpy(data.data() + offset, &iv, 4);
        }
    }
};

} // namespace fakejit
