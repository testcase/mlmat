// Fake Max/Jitter API function implementations.
#include "fakejit.h"
#include <cstdarg>

namespace c74 { namespace max {

static t_symbol sym_getinfo{"getinfo"};
static t_symbol sym_getdata{"getdata"};
static t_symbol sym_setinfo{"setinfo"};
static t_symbol sym_frommatrix{"frommatrix"};
static t_symbol sym_lock{"lock"};
static t_symbol sym_jit_matrix{"jit_matrix"};
static t_symbol sym_float64{"float64"};
static t_symbol sym_long{"long"};

t_symbol* jit_sym_getinfo = &sym_getinfo;
t_symbol* jit_sym_getdata = &sym_getdata;
t_symbol* jit_sym_setinfo = &sym_setinfo;
t_symbol* jit_sym_frommatrix = &sym_frommatrix;
t_symbol* jit_sym_lock = &sym_lock;
t_symbol* jit_sym_jit_matrix = &sym_jit_matrix;
t_symbol* jit_sym_float64 = &sym_float64;
t_symbol* jit_sym_long = &sym_long;

void* object_method(const void* x, t_symbol* s, ...)
{
    if (x == nullptr)
        return nullptr;
    auto* m = const_cast<fakejit::FakeMatrix*>(reinterpret_cast<const fakejit::FakeMatrix*>(x));
    va_list ap;
    va_start(ap, s);
    void* result = nullptr;
    if (s == jit_sym_getinfo) {
        auto* info = va_arg(ap, t_jit_matrix_info*);
        *info = m->info;
    } else if (s == jit_sym_getdata) {
        auto** d = va_arg(ap, uchar**);
        *d = m->data.data();
        result = *d;
    } else if (s == jit_sym_setinfo) {
        auto* info = va_arg(ap, t_jit_matrix_info*);
        m->info = *info;
        m->realize();
    } else if (s == jit_sym_frommatrix) {
        auto* src = va_arg(ap, void*);
        auto* sm = reinterpret_cast<fakejit::FakeMatrix*>(src);
        m->info = sm->info;
        m->data = sm->data;
    } else if (s == jit_sym_lock) {
        result = (void*)1L;
    }
    va_end(ap);
    return result;
}

void* jit_object_new(t_symbol* classname, ...)
{
    if (classname != jit_sym_jit_matrix)
        return nullptr;
    va_list ap;
    va_start(ap, classname);
    auto* info = va_arg(ap, t_jit_matrix_info*);
    auto* m = new fakejit::FakeMatrix();
    m->info = *info;
    m->realize();
    va_end(ap);
    return m;
}

void jit_object_free(void* x)
{
    delete reinterpret_cast<fakejit::FakeMatrix*>(x);
}

}} // namespace c74::max
