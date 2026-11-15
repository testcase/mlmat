/// @file
/// Reference-page generator for mlmat.mlp_classifier.
///
/// Runs outside of Max against the min-api mock kernel (the object unit
/// test pattern): constructs the object, calls stock
/// c74::min::doc_generate() to write the base page, then enhances it
/// (attribute defaults, enum lists, list sizes, class discussion -- see
/// mlmat_docgen.hpp) without any modification to min-api. Wired as a
/// POST_BUILD step of the external so the refpage always ends up newer
/// than the .mxo: doc_update inside Max then leaves the enhanced page
/// alone.

#define MIN_TEST
#include "c74_min.h"
#include "mlmat_docgen.hpp"
#include "../../projects/mlmat.mlp_classifier/mlmat.mlp_classifier.cpp"

int main(int argc, char** argv)
{
    const std::string refpage = argc > 1 ? argv[1] : "docs/mlmat.mlp_classifier.maxref.xml";

    mlmat_mlp_classifier instance;

    // stock generation writes the base page ...
    c74::min::doc_generate(instance, refpage, "mlmat.mlp_classifier", "mlmat_mlp_classifier");

    // ... and this adds the parts the old min-api patch used to emit
    mlmat::doc_enhance(instance, refpage);

    return 0;
}
