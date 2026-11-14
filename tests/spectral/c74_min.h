// Stub for the c74 min headers outside of Max: the spectral code only uses
// c74::min::string, which is layout-compatible with std::string here.
#pragma once
#include <string>
#include <sstream>
namespace c74 { namespace min { using string = std::string; } }
using namespace c74;
using namespace c74::min;
using namespace std;
