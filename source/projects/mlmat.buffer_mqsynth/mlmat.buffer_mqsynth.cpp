/// @file
/// mlmat.buffer_mqsynth: McAulay-Quatieri sinusoidal synthesis -- a
/// tracked partial matrix (mlmat.spectral_peaks with @track 1:
/// amplitudes, frequencies and phases per partial per frame) becomes
/// a bank of continuous, gliding sinusoids written to a buffer~.
///
/// @license    Use of this source code is governed by the MIT License found in the License.md file.

#include "c74_min.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>
#include "mlmat_stft.hpp"
using namespace c74;
using namespace c74::min;

max::t_object* maxob_from_jitob(max::t_object* job);
void mlmat_mqsynth_assist(void* x, void* b, long io, long index, char* s);
max::t_jit_err mlmat_mqsynth_matrix_calc(max::t_object* x, max::t_object* inputs, max::t_object* outputs);
void mlmat_mqsynth_notify(max::t_object* x, max::t_symbol* s, max::t_symbol* msg, void* sender, void* data);

class mlmat_buffer_mqsynth : public object<mlmat_buffer_mqsynth>, public matrix_operator<>
{
public:
    MIN_DESCRIPTION    { "Synthesize a tracked partial matrix (mlmat.spectral_peaks with @track 1: amplitudes, frequencies, phases) back into audio with McAulay-Quatieri sinusoidal interpolation, written to a buffer~." };
    MIN_TAGS        { "audio, analysis" };
    MIN_AUTHOR        { "Todd Ingalls" };
    MIN_RELATED        { "mlmat.buffer_stft, mlmat.spectral_peaks, mlmat.buffer_istft" };
    MIN_DISCUSSION  {"Sinusoidal-model resynthesis: each tracked partial becomes one continuous sinusoid, interpolated frame to frame with linear amplitude and a cubic phase polynomial that matches the measured phase and the frequency (its own derivative) at both frame boundaries -- the construction that makes partials glide smoothly instead of clicking at every frame boundary. Feed the tracked output of <o>mlmat.spectral_peaks</o> (with <at>track</at> set) into the inlet: plane 0 holds the partial amplitudes, plane 1 the frequencies in Hz, plane 2 the analysis phases in radians that anchor the interpolation. Set <at>fftsize</at> and <at>overlap</at> to match the analysis (the hop between synthesis points is fftsize / overlap, and the output length is the matrix's frame count times the hop). Births fade in over the frame before the partial, deaths fade out over the frame after, and <o>mlmat.spectral_peaks</o>'s coasting frames (zero amplitude, frequency held) fade down and back within one partial. Because the phases travel inside the matrix, edits between the objects are phase-safe: transform the matrix -- drop, transpose or rescale partials, repeat frames for time stretching -- and this object resynthesizes the result without detaching partials from their phase data. Untracked input (the default per-frame amplitude ranking) synthesizes unrelated glissandi; use tracked output. The partials carry amplitudes, so set mlmat.spectral_peaks' <at>output</at> to amplitudes for true-level resynthesis. Frequencies are computed against <at>samplerate</at> if set, otherwise the current MSP sample rate (which matches the buffer_stft chain)."};

    max::t_buffer_ref *m_buffer_reference {nullptr};

    attribute<int, threadsafe::no, limit::clamp> channel {this, "channel", 1,
        description {"Channel to write in the buffer~. The channel number uses 1-based counting."},
        range {1, buffer_reference::k_max_channels}
    };

    attribute<int, threadsafe::no, limit::clamp> fftsize {this, "fftsize", 1024,
        description {"Size of the analysis DFT. Must match mlmat.buffer_stft's fftsize."},
        range {16, 65536}
    };

    attribute<int, threadsafe::no, limit::clamp> overlap {this, "overlap", 4,
        description {"Overlap of the analysis windows. Must match mlmat.buffer_stft's overlap: the synthesis hop is fftsize / overlap."},
        range {1, 65536}
    };

    attribute<double> samplerate { this, "samplerate", 0.0,
        description {
            "Sample rate in Hz the partial frequencies refer to. If 0 (the default), the current MSP sample rate is used, which matches the mlmat.buffer_stft chain when its resampling is on."
        },
        range {0.0, 192000.0}
    };

    attribute<min::symbol> initial_buffer { this, "initial_buffer", true,
        visibility(visibility::hide)
    };

    // I find I need this to get valid buffer when instatiated in a loaded patch
    queue<> initial_buffer_deferrer { this,
        MIN_FUNCTION {
            set(initial_buffer.get());
            return {};
        }};

    queue<> update_buffer { this,
        MIN_FUNCTION {
            buffer_update();
            return {};
    }};

    // Using this to get first arg for buffer
    min::message<> maxob_setup {this, "maxob_setup",
        MIN_FUNCTION {
            max::t_object* mob = maxob_from_jitob(min::object_base::maxobj());
            m_dumpoutlet = max_jit_obex_dumpout_get(mob);
            if(!args.empty()) {
                initial_buffer.set(args);
                initial_buffer_deferrer.set();
            }
            return {};
    }};

    min::message<> dblclick {this, "dblclick",
        MIN_FUNCTION {
            max::buffer_view(max::buffer_ref_getobject(m_buffer_reference));
            return {};
    }};

    min::message<> set {this, "set",
        MIN_FUNCTION {
            if (!m_buffer_reference)
                m_buffer_reference = c74::max::buffer_ref_new(min::object_base::maxobj(), args[0]);
            else
                max::buffer_ref_set(m_buffer_reference, args[0]);
            return {};
    }};

    c74::max::t_jit_err matrix_calc(c74::max::t_object* x, c74::max::t_object* inputs, c74::max::t_object* outputs) {
        max::t_jit_err err = max::JIT_ERR_NONE;
        max::t_jit_matrix_info pk_minfo;
        max::uchar* pk_data = nullptr;

        auto in_matrix = max::object_method(inputs, max::_jit_sym_getindex, 0);
        if (!in_matrix)
            return max::JIT_ERR_INVALID_PTR;
        long pk_savelock = (long) max::object_method(in_matrix, max::_jit_sym_lock, 1);

        max::object_method(in_matrix, max::_jit_sym_getinfo, &pk_minfo);

        {
            const bool pk_is64 = (pk_minfo.type == max::_jit_sym_float64);
            if (!pk_is64 && pk_minfo.type != max::_jit_sym_float32) {
                cerr << "mlmat.buffer_mqsynth: input matrix must be float32 or float64." << endl;
                err = max::JIT_ERR_INVALID_INPUT;
                goto out;
            }
            if (pk_minfo.dimcount != 2) {
                cerr << "mlmat.buffer_mqsynth: input matrix must be 2d (frames x partials)." << endl;
                err = max::JIT_ERR_INVALID_INPUT;
                goto out;
            }
            if (pk_minfo.planecount != 3) {
                cerr << "mlmat.buffer_mqsynth: input matrix must have 3 planes (amplitudes, frequencies, phases -- mlmat.spectral_peaks @track 1 output)." << endl;
                err = max::JIT_ERR_INVALID_INPUT;
                goto out;
            }
            if (pk_minfo.dim[0] <= 0 || pk_minfo.dim[1] <= 0) {
                err = max::JIT_ERR_INVALID_INPUT;
                goto out;
            }
            if (!mlmat::fftsize_is_valid(fftsize)) {
                cerr << "mlmat.buffer_mqsynth: fftsize " << fftsize << " is not a power of two." << endl;
                err = max::JIT_ERR_INVALID_INPUT;
                goto out;
            }

            const long frames = pk_minfo.dim[0];
            const size_t slots = static_cast<size_t>(pk_minfo.dim[1]);
            const size_t hop = size_t(fftsize) / size_t(overlap);
            if (hop == 0) {
                cerr << "mlmat.buffer_mqsynth: overlap " << overlap << " exceeds fftsize "
                     << fftsize << "; the hop would be zero." << endl;
                err = max::JIT_ERR_INVALID_INPUT;
                goto out;
            }
            const double sr = (samplerate > 0.0) ? static_cast<double>(samplerate)
                                                 : static_cast<double>(max::sys_getsr());
            const size_t output_length = static_cast<size_t>(frames) * hop;

            max::t_buffer_obj *buffer = max::buffer_ref_getobject(m_buffer_reference);
            if (!(buffer && m_buffer_reference)) {
                cerr << "mlmat.buffer_mqsynth: buffer specified is not valid" << endl;
                err = max::JIT_ERR_INVALID_PTR;
                goto out;
            }
            max::object_method(static_cast<max::t_object*>(buffer), max::gensym("sizeinsamps"), (void*)static_cast<long>(output_length), 0);

            max::object_method(in_matrix, max::_jit_sym_getdata, &pk_data);
            if (pk_data == nullptr) {
                err = max::JIT_ERR_INVALID_INPUT;
                goto out;
            }

            // unpack (members: a steady-state call allocates nothing)
            m_peak_amps.assign(static_cast<size_t>(frames) * slots, 0.0f);
            m_peak_freqs.assign(static_cast<size_t>(frames) * slots, 0.0f);
            m_peak_phases.assign(static_cast<size_t>(frames) * slots, 0.0f);

            for (long f = 0; f < frames; ++f) {
                max::uchar* prow = pk_data + static_cast<size_t>(f) * pk_minfo.dimstride[0];
                for (size_t s = 0; s < slots; ++s) {
                    max::uchar* ip = prow + s * pk_minfo.dimstride[1];
                    if (pk_is64) {
                        m_peak_amps[static_cast<size_t>(f) * slots + s] = static_cast<float>(reinterpret_cast<const double*>(ip)[0]);
                        m_peak_freqs[static_cast<size_t>(f) * slots + s] = static_cast<float>(reinterpret_cast<const double*>(ip)[1]);
                        m_peak_phases[static_cast<size_t>(f) * slots + s] = static_cast<float>(reinterpret_cast<const double*>(ip)[2]);
                    } else {
                        m_peak_amps[static_cast<size_t>(f) * slots + s] = reinterpret_cast<const float*>(ip)[0];
                        m_peak_freqs[static_cast<size_t>(f) * slots + s] = reinterpret_cast<const float*>(ip)[1];
                        m_peak_phases[static_cast<size_t>(f) * slots + s] = reinterpret_cast<const float*>(ip)[2];
                    }
                }
            }

            m_mqsynth.process(m_peak_amps, m_peak_freqs, m_peak_phases,
                              static_cast<size_t>(frames), slots,
                              fftsize, overlap, sr,
                              m_samples);

            max::buffer_setdirty(buffer);
            buffer_update();
        }

    out:
        max::object_method(in_matrix, c74::max::_jit_sym_lock, pk_savelock);
        return err;
    }

    max::t_max_err notify(max::t_object* x, max::t_symbol* s, max::t_symbol* msg, void* sender, void* data) {
        return max::buffer_ref_notify(m_buffer_reference, s, msg, sender, data);
    }

    template<typename matrix_type>
    matrix_type calc_cell(matrix_type input, const min::matrix_info& info, min::matrix_coord& position) {
        matrix_type output;
        return output;
    }

    ~mlmat_buffer_mqsynth() {
        if(m_buffer_reference) {
            max::object_free(m_buffer_reference);
        }
    }

protected:
    void* m_dumpoutlet { nullptr };

private:
    message<> buffer_update {this, "buffer_update",
        MIN_FUNCTION {
            max::t_buffer_obj *buffer = max::buffer_ref_getobject(m_buffer_reference);
            if (!(buffer && m_buffer_reference)) {
                return {};
            }
            float *tab = buffer_locksamples(buffer);
            size_t b_frame_count = max::buffer_getframecount(buffer);
            max::t_atom_long b_channel_count = max::buffer_getchannelcount(buffer);

            if (buffer && tab) {
                auto chan = std::min<size_t>(channel - 1, b_channel_count);
                for(auto i = 0;i<b_frame_count;i++) {
                    tab[(i*b_channel_count) + chan] = m_samples[i];
                }
                max::buffer_setdirty(buffer);
            }
            max::buffer_unlocksamples(buffer);
            return {};
    }};

    message<> maxclass_setup {this, "maxclass_setup", MIN_FUNCTION {
        c74::max::t_class* c = args[0];
        max_jit_class_mop_wrap(c, this_jit_class, 0);
        max_jit_class_wrap_standard(c, this_jit_class, 0);
        class_addmethod(c, (c74::max::method)mlmat_mqsynth_assist, "assist", c74::max::A_CANT, 0);
        return {};
    }};

    message<> jitclass_setup {this, "jitclass_setup", MIN_FUNCTION {
        max::t_class* c = args[0];
        // add mop: one matrix input (the tracked partials), no matrix
        // outputs (the result goes to the buffer~)
        max::t_object* mop = static_cast<max::t_object*>(max::jit_object_new(max::_jit_sym_jit_mop, 1, 0));
        jit_mop_single_type(mop, max::_jit_sym_float32);
        jit_class_addadornment(c, mop);
        jit_class_addmethod(c, (max::method)mlmat_mqsynth_matrix_calc, "matrix_calc", max::A_CANT, 0);
        jit_class_addmethod(c, (max::method)mlmat_mqsynth_notify, "notify", max::A_CANT, 0); //cant get notify to work unless i do this
        return {};
    }};

    // making custom mop so max_jit_mop_matrix_args does not get called as don't need these
    message<> mop_setup {this, "mop_setup", MIN_FUNCTION {
        max::t_jit_err err = max::JIT_ERR_NONE;

        // max_jit_mop_new() appends the max wrapper pointer as the final
        // argument of this message, and inside the jit object maxobj() is
        // the jit object itself. Link the two first: this is the canonical
        // custom-mop sequence (see max.jit.mop.h), and it is also what
        // stores the "maxwrapper" key that maxob_from_jitob() relies on
        // elsewhere (e.g. in maxob_setup). Without it the wrapper and the
        // jit object are never connected: no outlets are created, no
        // attribute getting/setting is forwarded from the box, and
        // jit_error_code() during creation crashes on the null doc.
        max::t_object* x = args.empty()
            ? maxob_from_jitob(min::object_base::maxobj())
            : static_cast<max::t_object*>(args.back());
        max::t_object* o = min::object_base::maxobj();
        max::max_jit_obex_jitob_set(x, o);

        max::max_jit_obex_dumpout_set(x,outlet_new(x,NULL));
        err = max::max_jit_mop_setup(x);
        if(err != max::JIT_ERR_NONE) {
            (cerr << "mop_setup: max_jit_mop_setup error " << err << endl);
        }

        err = max::max_jit_mop_inputs(x);
        if(err != max::JIT_ERR_NONE) {
            (cerr << "mop_setup: max_jit_mop_inputs error " << err << endl);
        }

        err = max::max_jit_mop_outputs(x);
        if(err != max::JIT_ERR_NONE) {
            (cerr << "mop_setup: max_jit_mop_outputs error " << err << endl);
        }
        return {};
    }};

    std::vector<float> m_samples;
    std::vector<float> m_peak_amps;
    std::vector<float> m_peak_freqs;
    std::vector<float> m_peak_phases;
    MQSynth m_mqsynth;
};

MIN_EXTERNAL(mlmat_buffer_mqsynth);

void mlmat_mqsynth_notify(c74::max::t_object* x, c74::max::t_symbol* s, c74::max::t_symbol* msg, void* sender, void* data) {
    minwrap<mlmat_buffer_mqsynth>* job = (minwrap<mlmat_buffer_mqsynth>*)(x);
    job->m_min_object.notify(x, s, msg, sender, data);
}

c74::max::t_jit_err mlmat_mqsynth_matrix_calc(c74::max::t_object* x, c74::max::t_object* inputs, c74::max::t_object* outputs) {
    c74::max::t_jit_err err = c74::max::JIT_ERR_NONE;
    if (!x || !inputs || !outputs)
        return c74::max::JIT_ERR_INVALID_PTR;
    else {
        minwrap<mlmat_buffer_mqsynth>* job = (minwrap<mlmat_buffer_mqsynth>*)(x);
        err = job->m_min_object.matrix_calc(x, inputs, outputs);
    }
    return err;
}

c74::max::t_object* maxob_from_jitob(c74::max::t_object* job) {
    c74::max::t_object* mwrap = NULL;
    c74::max::object_obex_lookup(job, c74::max::gensym("maxwrapper"), &mwrap);

    return mwrap;
}

void mlmat_mqsynth_assist(void* x, void* b, long io, long index, char* s) {
   switch(io) {
       case 1:
           switch(index) {
               case 0:
                   snprintf(s, 64, "(matrix) tracked partials: amplitudes, frequencies, phases");
                   break;
               default:
                   break;
           }
           break;
       case 2:
           switch(index) {
               case 0:
                   snprintf(s, 64, "(matrix) data");
                   break;
               default:
                   snprintf(s, 64, "dumpout");
                   break;
           }
   }
}
