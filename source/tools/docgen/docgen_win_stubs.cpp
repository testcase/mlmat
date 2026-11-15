/// @file
/// Windows-only linker stubs for the reference page generators.
///
/// The docgens construct min objects headlessly against the min-api mock
/// kernel: the compiled-in object/wrap code references many kernel, jit
/// and MSP symbols that are never actually called while generating a
/// reference page. On macOS those resolve weakly against the SDK's
/// JitterAPI/MaxAudioAPI frameworks. On Windows they would have to come
/// from the SDK import libraries (MaxAPI.lib, jitlib.lib, MaxAudio.lib),
/// but import libraries create hard DLL dependencies at process start,
/// and the docgen executables run on machines without Max installed.
/// Instead the mock kernel provides what it implements, and everything
/// it does not (determined by nm over the mac docgen builds) is stubbed
/// here. This TU is empty on other platforms.
///
/// Note: jit.symbols.h declares the _jit_sym_* variables with
/// __declspec(dllimport) on Windows, so the compiled references go
/// through __imp__-prefixed pointers rather than the plain names; those
/// (and only those) are defined here.

#ifdef _WIN32

extern "C" {


// --- Max kernel (normally MaxAPI.lib) ---
extern "C" void* linklist_new() { return nullptr; }
extern "C" void linklist_chuck() { }
extern "C" void* linklist_getindex() { return nullptr; }
extern "C" long linklist_append() { return 0; }
extern "C" long object_method_typed() { return 0; }
extern "C" long object_obex_lookup() { return 0; }
extern "C" void* object_attr_getsym() { return nullptr; }
extern "C" long path_getdefault() { return 0; }
extern "C" void* quittask_install() { return nullptr; }
extern "C" void quittask_remove2() { }
extern "C" long open_dialog() { return 0; }
extern "C" long saveas_dialog() { return 0; }
extern "C" char* strncpy_zero() { return nullptr; }

// --- MSP buffer API (normally MaxAudio.lib) ---
extern "C" long buffer_getchannelcount() { return 0; }
extern "C" long buffer_getframecount() { return 0; }
extern "C" double buffer_getsamplerate() { return 0; }
extern "C" void* buffer_ref_new() { return nullptr; }
extern "C" long buffer_ref_notify() { return 0; }
extern "C" void buffer_ref_set() { }
extern "C" void buffer_setdirty() { }
extern "C" void buffer_view() { }

// --- jit / mop glue (normally jitlib.lib) ---
extern "C" long jit_attr_setlong() { return 0; }
extern "C" long jit_error_code() { return 0; }
extern "C" long jit_mop_input_nolink() { return 0; }
extern "C" long jit_mop_ioproc_copy_adapt() { return 0; }
extern "C" long jit_mop_output_nolink() { return 0; }
extern "C" long jit_mop_single_type() { return 0; }
extern "C" void* jit_object_method_imp() { return nullptr; }
extern "C" long max_jit_classex_mop_mproc() { return 0; }
extern "C" long max_jit_mop_inputs() { return 0; }
extern "C" void* max_jit_mop_io_getoutlet() { return nullptr; }
extern "C" long max_jit_mop_jit_matrix() { return 0; }
extern "C" long max_jit_mop_matrix_args() { return 0; }
extern "C" long max_jit_mop_outputs() { return 0; }
extern "C" long max_jit_mop_setup() { return 0; }
extern "C" void* max_jit_obex_dumpout_get() { return nullptr; }
extern "C" void max_jit_obex_dumpout_set() { }
extern "C" long max_jit_obex_inletnumber_get() { return 0; }
extern "C" void max_jit_obex_jitob_set() { }

// --- jit cached symbols: dllimport data, so __imp__ pointers (normally
//     jitlib.dll). Never dereferenced by the docgen.
void* __imp__jit_sym_char = nullptr;
void* __imp__jit_sym_dimlink = nullptr;
void* __imp__jit_sym_float32 = nullptr;
void* __imp__jit_sym_float64 = nullptr;
void* __imp__jit_sym_frommatrix = nullptr;
void* __imp__jit_sym_getdata = nullptr;
void* __imp__jit_sym_getindex = nullptr;
void* __imp__jit_sym_getinfo = nullptr;
void* __imp__jit_sym_getinput = nullptr;
void* __imp__jit_sym_getinputlist = nullptr;
void* __imp__jit_sym_getmatrix = nullptr;
void* __imp__jit_sym_getoutput = nullptr;
void* __imp__jit_sym_getoutputlist = nullptr;
void* __imp__jit_sym_ioproc = nullptr;
void* __imp__jit_sym_jit_matrix = nullptr;
void* __imp__jit_sym_jit_mop = nullptr;
void* __imp__jit_sym_lock = nullptr;
void* __imp__jit_sym_long = nullptr;
void* __imp__jit_sym_matrix_calc = nullptr;
void* __imp__jit_sym_matrixname = nullptr;
void* __imp__jit_sym_setinfo = nullptr;
void* __imp__jit_sym_typelink = nullptr;
void* __imp__jit_sym_types = nullptr;

} // extern "C"

#endif // _WIN32
