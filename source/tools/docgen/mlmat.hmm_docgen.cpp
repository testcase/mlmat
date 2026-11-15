/// @file
/// Reference-page generator for mlmat.hmm.
///
/// Runs outside of Max against the min-api mock kernel (the object unit
/// test pattern): constructs the object, calls stock
/// c74::min::doc_generate() to write the base page, then enhances it
/// (attribute defaults, enum lists, list sizes, class discussion -- see
/// mlmat_docgen.hpp) without any modification to min-api. Wired as a
/// POST_BUILD step of the external so the refpage always ends up newer
/// than the .mxo: doc_update inside Max then leaves the enhanced page
/// alone.

#include "c74_min.h"
#include "mlmat_docgen.hpp"
#include "../../projects/mlmat.hmm/mlmat.hmm.cpp"

#include <filesystem>

int main(int argc, char** argv)
{
    std::string refpage = argc > 1 ? argv[1] : "docs/mlmat.hmm.maxref.xml";

    // a fresh checkout has no docs/ (it is generated, not tracked)
    if (const size_t slash = refpage.rfind('/'); slash != std::string::npos) {
        std::error_code ec;
        std::filesystem::create_directories(refpage.substr(0, slash), ec);
    }

    mlmat_hmm instance;

    // stock generation writes the base page ...
    c74::min::doc_generate(instance, refpage, "mlmat.hmm", "mlmat_hmm");

    // ... and this adds the parts the old min-api patch used to emit
    mlmat::doc_enhance(instance, refpage);

    return 0;
}
