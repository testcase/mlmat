/// @file mlmat.hpp
/// @ingroup mlmat
/// @copyright Copyright 2021 Todd Ingalls. All rights reserved.
/// @license  Use of this source code is governed by the MIT License found in the License.md file.

#pragma once

#include "c74_min.h"



#ifdef WIN_VERSION
#define	hypot hypot //this overrides a define which is causing conflicts with mlpack
#endif

#include "mlmat_object.hpp"


//if code has not been patched for better documentation generation at least
// define this macro so can build without errors

#ifndef MIN_DISCUSSION

#ifdef WIN_VERSION
#pragma message "source code for min-api has not been patched to generate better doc. this will not impact functionality of object. please see README for patching instructions"
#else
#warning "source code for min-api has not been patched to generate better doc. this will not impact functionality of object. please see README for patching instructions"
#endif

#define MIN_DISCUSSION static constexpr const char* class_discussion
#endif






// does test that matrix is suitable for mode
inline void check_mode(const c74::max::t_jit_matrix_info& minfo,
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






