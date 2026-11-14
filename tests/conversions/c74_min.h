// Stub of the Max/Jitter API surface used by matrix_conversions.hpp so the
// conversion code can be tested outside of Max. The fake jit matrices are
// implemented in fakejit.h/.cpp.
#pragma once
#include <armadillo>
#include <cstddef>
#include <cstdint>

namespace c74 { namespace max {

using uchar = unsigned char;
using t_int32 = int32_t;
using t_jit_err = long;
struct t_object;      // opaque; actually a fakejit::FakeMatrix in the stub
struct t_symbol { const char* s; };

enum {
    JIT_ERR_NONE = 0,
    JIT_ERR_INVALID_PTR = -1
};

struct t_jit_matrix_info {
    long size;
    t_symbol* type;
    long flags;
    long dimcount;
    long dim[32];
    long dimstride[32];
    long planecount;
};

// symbols (defined in fakejit.cpp)
extern t_symbol* jit_sym_getinfo;
extern t_symbol* jit_sym_getdata;
extern t_symbol* jit_sym_setinfo;
extern t_symbol* jit_sym_frommatrix;
extern t_symbol* jit_sym_lock;
extern t_symbol* jit_sym_jit_matrix;
extern t_symbol* jit_sym_float64;
extern t_symbol* jit_sym_long;

void* object_method(const void* x, t_symbol* s, ...);
void* jit_object_new(t_symbol* classname, ...);
void jit_object_free(void* x);

}} // namespace c74::max

// the real headers expose the symbols via these macros; always used with an
// explicit c74::max:: prefix in the conversion code
#define _jit_sym_getinfo    jit_sym_getinfo
#define _jit_sym_getdata    jit_sym_getdata
#define _jit_sym_setinfo    jit_sym_setinfo
#define _jit_sym_frommatrix jit_sym_frommatrix
#define _jit_sym_lock       jit_sym_lock
#define _jit_sym_jit_matrix jit_sym_jit_matrix
#define _jit_sym_float64    jit_sym_float64
#define _jit_sym_long       jit_sym_long

// the original header calls jit_object_free unqualified in one place
using c74::max::jit_object_free;
