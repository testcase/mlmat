/// @file
/// Enhancement pass for min-api generated reference pages, without
/// modifying min-api.
///
/// The flow (see the mlmat.<object>_docgen tools): stock
/// c74::min::doc_generate() writes the base page, then doc_enhance()
/// adds what the old "better docs" min-api patch used to emit from
/// inside the generator:
///   * the default value appended to each attribute's description
///   * a correct size for list attributes (stock always writes 1)
///   * an <attributelist> with the enum values (enumvals), label and
///     style for symbol attributes with a range -- this is what makes
///     the Inspector show a popup menu for the attribute
///   * the class discussion (the MIN_DISCUSSION macro in
///     shared/mlmat.hpp) as a <discussion> element
///
/// Everything here consumes public min-api APIs only:
/// instance.attributes(), attribute_base::default_string(),
/// range_string(), label_string(), datatype(), visible().

#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "c74_min.h"

namespace mlmat {

// ---------------------------------------------------------------------------
// class discussion (companion of the MIN_DISCUSSION macro in mlmat.hpp)
// ---------------------------------------------------------------------------

template <class T, class = void>
struct has_class_discussion : std::false_type {};

template <class T>
struct has_class_discussion<T, std::void_t<decltype(&T::class_discussion)>> : std::true_type {};

template <class min_class_type>
std::string doc_get_discussion()
{
    if constexpr (has_class_discussion<min_class_type>::value) {
        return c74::min::doc_format(min_class_type::class_discussion);
    }
    return {};
}

// ---------------------------------------------------------------------------
// string helpers
// ---------------------------------------------------------------------------

inline std::vector<std::string> split_tokens(const std::string& s, bool strip_quotes)
{
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string word;
    while (iss >> word) {
        if (strip_quotes && word.size() >= 2 && word.front() == '"' && word.back() == '"') {
            word = word.substr(1, word.size() - 2);
        }
        if (!word.empty()) {
            tokens.push_back(word);
        }
    }
    return tokens;
}

// Locate the <attribute name='NAME' ...> ... </attribute> block of the
// ATTRIBUTES section (the page may contain nested <attribute> elements
// in attributelists -- those never directly follow "name='NAME'" from
// the top level, and matching the quoted name keeps this unambiguous).
inline bool find_attribute_block(const std::string& page, const std::string& name, size_t& begin, size_t& end)
{
    const std::string open = "<attribute name='" + name + "'";
    begin = page.find(open);
    if (begin == std::string::npos) {
        return false;
    }
    end = page.find("</attribute>", begin);
    if (end == std::string::npos) {
        return false;
    }
    end += std::string("</attribute>").size();
    return true;
}

// ---------------------------------------------------------------------------
// the enhancement pass
// ---------------------------------------------------------------------------

// Format a single atom as text. min::atom is layout-compatible with
// max::t_atom (min itself casts &atom[0] to t_atom* throughout).
inline std::string atom_to_text(const c74::min::atom& a)
{
    const auto* ta = reinterpret_cast<const c74::max::t_atom*>(&a);
    switch (c74::max::atom_gettype(ta)) {
        case c74::max::A_LONG:
            return std::to_string(c74::max::atom_getlong(ta));
        case c74::max::A_FLOAT: {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(c74::max::atom_getfloat(ta)));
            return buf;
        }
        case c74::max::A_SYM:
            return c74::max::atom_getsym(ta)->s_name;
        default:
            return {};
    }
}

// The declared default of an attribute, as text. The object is freshly
// constructed when this runs, so the current values ARE the defaults.
// (The stock default_string() goes through atom_gettext, which the
// mock kernel stubs to the empty string.)
inline std::string default_value_string(const c74::min::attribute_base& attr)
{
    std::string s;
    for (const auto& a : attr.get_atoms()) {
        const std::string t = atom_to_text(a);
        if (t.empty()) {
            continue;
        }
        if (!s.empty()) {
            s += " ";
        }
        s += t;
    }
    return s;
}

template <class min_class_type>
void doc_enhance(const min_class_type& instance, const std::string& refpage_fullpath)
{
    std::ifstream in(refpage_fullpath);
    if (!in) {
        return;
    }
    std::stringstream buf;
    buf << in.rdbuf();
    in.close();
    std::string page = buf.str();
    if (page.empty()) {
        return;
    }

    for (const auto& [name, attr_ptr] : instance.attributes()) {
        const auto& attr = *attr_ptr;
        if (attr.visible() == c74::min::visibility::hide) {
            continue;
        }

        size_t begin;
        size_t end;
        if (!find_attribute_block(page, name, begin, end)) {
            continue;
        }
        std::string block = page.substr(begin, end - begin);

        const std::string default_str = default_value_string(attr);

        // list size from the default's atom count (stock always writes 1)
        const size_t default_count = attr.get_atoms().size();
        if (default_count > 1) {
            const size_t size_pos = block.find("size='1'");
            if (size_pos != std::string::npos) {
                block.replace(size_pos, std::string("size='1'").size(),
                              "size='" + std::to_string(default_count) + "'");
            }
        }

        // append the default value to the description
        if (!default_str.empty() && block.find("default =") == std::string::npos) {
            const size_t desc_end = block.rfind("</description>");
            if (desc_end != std::string::npos) {
                block.insert(desc_end, " (default = " + default_str + ")");
            }
        }

        // enum values for symbol attributes with a range: drives the
        // Inspector's popup menu
        const std::string attr_type = attr.datatype();
        const std::string range_str = attr.range_string();
        if (attr_type == "symbol" && !range_str.empty()) {
            const auto tokens = split_tokens(range_str, true);
            if (!tokens.empty()) {
                std::stringstream enums;
                enums << "\t\t\t<attributelist>" << std::endl;
                enums << "\t\t\t\t<attribute name='enumvals' get='1' set='1' type='atom' size='"
                      << tokens.size() << "' >" << std::endl;
                enums << "\t\t\t\t\t<enumlist>" << std::endl;
                for (const auto& t : tokens) {
                    enums << "\t\t\t\t\t\t<enum name='" << t << "' />" << std::endl;
                }
                enums << "\t\t\t\t\t</enumlist>" << std::endl;
                enums << "\t\t\t\t</attribute>" << std::endl;
                enums << "\t\t\t\t<attribute name='label' get='1' set='1' type='symbol' size='1' value='"
                      << attr.label_string() << "' />" << std::endl;
                enums << "\t\t\t\t<attribute name='style' get='1' set='1' type='symbol' size='1' value='enum' />" << std::endl;
                enums << "\t\t\t</attributelist>" << std::endl;
                const size_t close = block.rfind("</attribute>");
                if (close != std::string::npos) {
                    block.insert(close, enums.str());
                }
            }
        }

        page.replace(begin, end - begin, block);
    }

    // class discussion
    const std::string discussion = doc_get_discussion<min_class_type>();
    if (!discussion.empty()) {
        const size_t footer = page.rfind("</c74object>");
        if (footer != std::string::npos) {
            std::stringstream d;
            d << "\t<!--DISCUSSION-->" << std::endl << std::endl;
            d << "\t<discussion>" << discussion << "</discussion>" << std::endl << std::endl;
            page.insert(footer, d.str());
        }
    }

    std::ofstream out(refpage_fullpath);
    out << page;
}

} // namespace mlmat
