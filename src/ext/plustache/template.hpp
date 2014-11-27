/**
 * @file template.hpp
 * @brief header file for plustache template
 * @author Daniel Schauenberg <d@unwiredcouch.com>
 */
#ifndef PLUSTACHE_TEMPLATE_H
#define PLUSTACHE_TEMPLATE_H
#include <iostream>
#include <fstream>
#include <streambuf>
#include <boost/algorithm/string/trim.hpp>
#include <boost/regex.hpp>

#include <ext/plustache/plustache_types.hpp>
#include <ext/plustache/context.hpp>

namespace Plustache {
    class template_t {
      typedef PlustacheTypes::ObjectType ObjectType;
      typedef PlustacheTypes::CollectionType CollectionType;
    public:
        template_t ();
        template_t (std::string& tmpl_path);
        ~template_t ();
        std::string render(const std::string& tmplate, const Context& ctx);
        std::string render(const std::string& tmplate, const ObjectType& ctx);

    private:
        std::string template_path;
        /* opening and closing tags */
        std::string otag;
        std::string ctag;
        /* regex */
        boost::regex tag;
        boost::regex section;
        boost::regex escape_chars;
        /* lut for HTML escape chars */
        std::map<std::string, std::string> escape_lut;
        /* render and helper methods */
        std::string render_tags(const std::string& tmplate,
                                const Context& ctx);
        std::string render_sections(const std::string& tmplate,
                                    const Context& ctx);
        std::string html_escape(const std::string& s);
        std::string get_partial(const std::string& partial) const;
        void change_delimiter(const std::string& opentag,
                              const std::string& closetag);
        void compile_data();
        std::string get_template(const std::string& tmpl);
    };
} // namespace Plustache
#endif
