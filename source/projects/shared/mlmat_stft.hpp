/// @file mlmat.hpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_min.h"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>
using namespace c74;
using namespace c74::min;


class STFT
{
    
public:
    STFT() {
        
    }
    
    void process(std::vector<float>& input,
                 std::vector<float>& out_magnitudes,
                 std::vector<float>& out_phases,
                 int fftsize,
                 int overlap,
                 const string window_string,
                 bool output_polar = true,
                 bool full_spectrum = false)
    {
        
        size_t step = fftsize/overlap;
        
        vDSP_DFT_Setup tmp_setup = m_setup;

        m_setup = vDSP_DFT_zrop_CreateSetup(tmp_setup, fftsize, vDSP_DFT_FORWARD );
        if(m_setup == 0) {
            std::stringstream ss;
            ss << "fftsize " << fftsize << " is not valid.";
            throw std::invalid_argument( ss.str()  );
        }
        
        if(tmp_setup != NULL) {
            vDSP_DFT_DestroySetup(tmp_setup);
        }
        size_t fftsize_2 = fftsize / 2;
        size_t frame_count = input.size();
        
        // throw error if out vectors not correct length to hold data?


        
        // with resize and same sizes as last run should not incur much overhead. right?
        
        m_in_samples.resize(fftsize);
        m_in_real.resize(fftsize_2);
        m_in_imag.resize(fftsize_2, 0.0f);
        m_out_real.resize(fftsize_2);
        m_out_imag.resize(fftsize_2);
        m_window_vector.resize(fftsize);
        m_magnitudes.resize(fftsize_2);
        m_phases.resize(fftsize_2);
        

        /* window */
    
        
        if(window_string == "triangle") {
            float tristart = 0.0f;
            float tritop = 1.0f;
            float up = 1.0f/fftsize_2;
            float down = -1.0f/fftsize_2;
            vDSP_vramp(&tristart, &up,m_window_vector.data(), 1, fftsize/2);
            vDSP_vramp(&tritop, &down,m_window_vector.data()+((fftsize/2)-1), 1, fftsize/2);
        } else if (window_string == "hanning") {
            vDSP_hann_window(m_window_vector.data(), fftsize, vDSP_HANN_NORM );
        } else if (window_string == "hamming") {
            vDSP_hamm_window(m_window_vector.data(), fftsize, 0);
        } else if (window_string == "blackman") {
            vDSP_blkman_window(m_window_vector.data(), fftsize, 0);
        }
        
        DSPSplitComplex complex_input = { m_in_real.data(), m_in_imag.data()};
        DSPSplitComplex complex_output = { m_out_real.data(), m_out_imag.data()};
        int current_pos = 0;
        
        for (auto i = 0; i < frame_count; i+=step) {
            //get next N samples
            if(i+fftsize > frame_count) {
                for(auto j=0;j<fftsize;j++) {
                    if((i+j) >= frame_count) {
                        m_in_samples[j] = 0.0f;
                    } else {
                        m_in_samples[j] = input[(i+j)];
                    }
                }
            }
            else {
                std::copy_n(input.begin()+i, fftsize, m_in_samples.begin());
                
            }
            //window
            if (window_string != "square") {
                vDSP_vmul(m_in_samples.data(), 1, m_window_vector.data(), 1, m_in_samples.data(), 1, fftsize);
            }
            
            //put into SplitComplex
            vDSP_ctoz((DSPComplex *)m_in_samples.data(), 2, &complex_input, 1, fftsize_2);

            // dft
            vDSP_DFT_Execute(m_setup,complex_input.realp, complex_input.imagp,complex_output.realp,complex_output.imagp);
        

            float scale = .5;
        
            vDSP_vsmul(complex_output.realp, 1, &scale, complex_output.realp, 1, fftsize_2);
            vDSP_vsmul(complex_output.imagp, 1, &scale, complex_output.imagp, 1, fftsize_2);
     
            complex_output.imagp[0] = 0.0;
            
            if(output_polar) {
                //this gets a little convoluted but after profiling seems faster
                vDSP_zvabs(&complex_output, 1, m_magnitudes.data(), 1, fftsize_2);  // Compute magnitude
                vDSP_zvphas(&complex_output, 1, m_phases.data(), 1, fftsize_2);  // Compute phase (radians)
                std::copy(m_magnitudes.begin(), m_magnitudes.end(), out_magnitudes.begin()+current_pos);
                std::copy(m_phases.begin(), m_phases.end(), out_phases.begin()+current_pos);
                current_pos += fftsize_2;
                
                if(full_spectrum) {
                    std::copy(m_magnitudes.rbegin(), m_magnitudes.rend(), out_magnitudes.begin()+current_pos);
                    std::copy(m_phases.rbegin(), m_phases.rend(), out_phases.begin()+current_pos);
                    current_pos += fftsize_2;
                }
            
            } else {
                // output rectangular data. vectors are magnitudes and phases in this case but that is misleading
                std::copy(m_out_real.begin(), m_out_real.end(), out_magnitudes.begin()+current_pos);
                std::copy(m_out_imag.begin(), m_out_imag.end(), out_phases.begin()+current_pos);
                current_pos += fftsize_2;
                if(full_spectrum) {
                    std::copy(m_out_real.rbegin(), m_out_real.rend(), out_magnitudes.begin()+current_pos);
                    std::copy(m_out_imag.rbegin(), m_out_imag.rend(), out_phases.begin()+current_pos);
                    current_pos += fftsize_2;
                }
            }
        }
        
    }
    
    ~STFT() {
        if(m_setup != nullptr) {
            vDSP_DFT_DestroySetup(m_setup);
        }
    }
private:
    vDSP_DFT_Setup m_setup = nullptr;
    std::vector<float> m_in_samples;
    std::vector<float> m_in_real;
    std::vector<float> m_in_imag;
    std::vector<float> m_out_real;
    std::vector<float> m_out_imag;
    std::vector<float> m_window_vector;
    std::vector<float> m_magnitudes;
    std::vector<float> m_phases;
};



class ISTFT {
public:
    ISTFT() {
    }
    
    void process(std::vector<float>& in_real,
                 std::vector<float>& in_imag,
                 std::vector<float>& samples,
                 int fftsize,
                 int overlap,
                 const string window_string,
                 bool input_polar = true,
                 bool full_spectrum = false)
    {
        
        if(input_polar) {
            float r = 0.0;
            float theta = 0.0;
            for(auto i=0;i<in_real.size();++i) {
                r = in_real[i];
                theta = in_imag[i];
                in_real[i] = r * cos(theta);
                in_imag[i] = r * sin(theta);
            }
        }
        

        size_t step = fftsize / overlap;
        size_t fftsize_2 = fftsize / 2;
        
        m_frame_real.resize(fftsize_2, 0.0f);
        m_frame_imag.resize(fftsize_2, 0.0f);
        
        vDSP_vclr(samples.data(), 1, samples.size());
        
        
        m_signal.resize(fftsize);
        m_window_vector.resize(fftsize);

        //size_t num_bins = fftsize_2;
        
        /* window */
        if(window_string == "triangle") {
            float tristart = 0.0f;
            float tritop = 1.0f;
            float up = 1.0f/fftsize_2;
            float down = -1.0f/fftsize_2;
            vDSP_vramp(&tristart, &up,m_window_vector.data(), 1, fftsize_2);
            vDSP_vramp(&tritop, &down,m_window_vector.data()+((fftsize/2)-1), 1, fftsize_2);
        } else if (window_string == "hanning") {
            vDSP_hann_window(m_window_vector.data(), fftsize, vDSP_HANN_NORM );
        } else if (window_string == "hamming") {
            vDSP_hamm_window(m_window_vector.data(), fftsize, 0);
        } else if (window_string == "blackman") {
            vDSP_blkman_window(m_window_vector.data(), fftsize, 0);
        }
        
        vDSP_DFT_Setup tmp_setup = m_setup;
        
        m_setup = vDSP_DFT_zrop_CreateSetup(tmp_setup, fftsize, vDSP_DFT_INVERSE );
        
        if(m_setup == 0) {
            std::stringstream ss;
            ss << "fftsize " << fftsize << " is not valid.";
            throw std::invalid_argument( ss.str()  );
        }
        
        if(tmp_setup != NULL) {
            vDSP_DFT_DestroySetup(tmp_setup);
        }
        
        DSPSplitComplex complex_input = { in_real.data(), in_imag.data()};
        DSPSplitComplex complex_output = { m_frame_real.data(), m_frame_imag.data()};

        
        float scaling = 1.0f / fftsize / overlap;
        
        for (auto i = 0, j=0; i < (samples.size()); i+=fftsize_2, j+=step) {
            
            vDSP_DFT_Execute(m_setup,complex_input.realp+i, complex_input.imagp+i,complex_output.realp,complex_output.imagp);

            vDSP_ztoc(&complex_output, 1, ( DSPComplex *)m_signal.data(), 2,fftsize_2);
            
            vDSP_vsmul(m_signal.data(), 1, &scaling, m_signal.data(), 1, fftsize);
            
            if (window_string != "square") {
                vDSP_vmul(m_signal.data(), 1, m_window_vector.data(), 1, m_signal.data(), 1, fftsize);
            }
            
            vDSP_vadd(m_signal.data(), 1, samples.data()+j, 1, samples.data()+j, 1, fftsize);
        }
    }
    
    
    
    ~ISTFT() {
        if(m_setup != nullptr) {
            vDSP_DFT_DestroySetup(m_setup);
        }
    }
    
private:
    vDSP_DFT_Setup m_setup = nullptr;
    
    std::vector<float> m_frame_real;
    std::vector<float> m_frame_imag;
    std::vector<float> m_signal;
    std::vector<float> m_window_vector;
        
    
};
