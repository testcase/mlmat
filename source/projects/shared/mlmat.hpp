/// @file mlmat.hpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.


#pragma once

//if code has not been patched for better documentation generation at least
// define this macro so can build without errors

#ifndef MIN_DISCUSSION
//#warning "source code for min-api has not been patched to generate better doc. this will not impact functionality of object. please see README for patching instructions"

#define MIN_DISCUSSION static constexpr const char* class_discussion
#endif

#include "c74_min.h"
#ifdef WIN_VERSION
#define	hypot hypot //this overrides a define which is causing conflicts with mlpack
#endif

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/methods/preprocess/scaling_model.hpp>

using namespace c74;
using namespace c74::min;
using namespace c74::max;
using mlpack::data::ScalingModel;

// does test that matrix is suitable for mode
inline void check_mode(const t_jit_matrix_info& minfo,
                       const int mode,
                       const std::string& callerDescription,
                       const std::string& addInfo = "matrix")
{
    if(mode > 0) {
        if(minfo.planecount > 1) {
            std::ostringstream oss;
            oss << callerDescription << ": number of planes (" << minfo.planecount << ") in matrix "
            << "is not valid in mode "  << mode << ". Expecting 1 plane!";
            throw std::invalid_argument(oss.str());
        }
    }
}


// retrieve maxob from jitter obj
t_object* maxob_from_jitob(c74::max::t_object* job) {
    c74::max::t_object* mwrap = NULL;
    object_obex_lookup(job, gensym("maxwrapper"), &mwrap);
    return mwrap;
}




template<typename ModelType>
inline void save_model_file(const atoms& args,
                            const ModelType& model,
                            const std::string& model_description) {
    if(model.model) {
        short path = 0;
        char filename[max::MAX_FILENAME_CHARS] = {0};
        char fullpath[max::MAX_PATH_CHARS] = {0};
        char native_path[max::MAX_PATH_CHARS]={0};
        
        if(!args.empty()) {
            string name = string(args[0]);
            path = max::path_getdefault();
            strncpy_zero(filename, name.c_str(), max::MAX_FILENAME_CHARS);
            max::path_toabsolutesystempath(path, filename, fullpath);
            max::path_nameconform(fullpath, native_path, max::PATH_STYLE_NATIVE, max::PATH_TYPE_PATH);
            try {
                mlpack::data::Save(string(fullpath), model_description, model, true);
            } catch (const std::runtime_error& s) {
                std::throw_with_nested(std::runtime_error("Error writing model file to disk."));
            }
        } else {
            if(!saveas_dialog(filename,&path, NULL)) {
                path_toabsolutesystempath(path, filename, fullpath);
                path_nameconform(fullpath, native_path, max::PATH_STYLE_NATIVE, max::PATH_TYPE_PATH);
                try {
                    mlpack::data::Save(string(fullpath), model_description, model, true);
                } catch (const std::runtime_error& s) {
                    std::throw_with_nested(std::runtime_error("Error writing model file to disk."));
                }
            }
        }
    } else {
        throw std::runtime_error("No trained model to save!");
    }
}


template<class class_type>
class mlmat_serializable_model
{
public:
    std::unique_ptr<class_type> model { nullptr };
    //bool model_trained = false;
    std::unique_ptr<ScalingModel> scaler { nullptr };
    bool scaler_changed = false;
    long dim0 = 0;//this is lame but cant see how else to do this. only needed for mode 0 with 2d reference matrix
    long dimcount = 1; //the horror
    bool autoscale = false;
    template<typename Archive>
    
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(CEREAL_NVP(model));
        ar(CEREAL_NVP(autoscale));
        ar(CEREAL_NVP(scaler));
        ar(CEREAL_NVP(scaler));
        ar(CEREAL_NVP(scaler_changed));
        ar(CEREAL_NVP(dim0));
        ar(CEREAL_NVP(dimcount));
    }
};
