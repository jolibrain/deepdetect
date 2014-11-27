/**
 * @file template.hpp
 * @brief header file for plustache template
 * @author Daniel Schauenberg <d@unwiredcouch.com>
 */
#ifndef PLUSTACHE_CONTEXT_H
#define PLUSTACHE_CONTEXT_H

#include <iostream>
#include <ext/plustache/plustache_types.hpp>

namespace Plustache {
	class Context {
	public:
	    Context ();
	    ~Context ();
	    int add(const std::string& key, const std::string& value);
	    int add(const std::string& key, PlustacheTypes::CollectionType& c);
	    int add(const std::string& key, const PlustacheTypes::ObjectType& o);
	    int add(const PlustacheTypes::ObjectType& o);
	    PlustacheTypes::CollectionType get(const std::string& key) const;

	  private:
	    /* data */
	    std::map<std::string, PlustacheTypes::CollectionType> ctx;
	};
} // namespace Plustache
#endif
