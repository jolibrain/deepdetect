/**
 * @file template.cpp
 * @brief plustache template
 * @author Daniel Schauenberg <d@unwiredcouch.com>
 */

#include <ext/plustache/context.hpp>
#include <ext/plustache/plustache_types.hpp>

using namespace Plustache;

Context::Context()
{

}

Context::~Context()
{

}

/**
 * @brief method to add a simple key/value to the Context
 *
 * @param key
 * @param value
 *
 * @return 0 on success
 */
int Context::add(const std::string& key, const std::string& value)
{
    PlustacheTypes::ObjectType obj;
    obj[key] = value;
    ctx[key].push_back(obj);
    return 0;
}

/**
 * @brief method to add a collection to a specific key in the Context
 *
 * @param key to store the data
 * @param c Collection to add
 *
 * @return 0 on success
 */
int Context::add(const std::string& key, PlustacheTypes::CollectionType& c)
{
    if (ctx.find(key) == ctx.end())
    {
        ctx[key] = c;
    }
    else
    {
        for(PlustacheTypes::CollectionType::iterator it = c.begin();
            it != c.end(); ++it)
        {
            (*this).add(key, (*it));
        }
    }
    return 0;
}

/**
 * @brief method to add an additional object to a collection in the Context
 *
 * @param key for the collection
 * @param o object to add
 *
 * @return 0
 */
int Context::add(const std::string& key, const PlustacheTypes::ObjectType& o)
{
    if (ctx.find(key) == ctx.end())
    {
      PlustacheTypes::CollectionType c;
      c.push_back(o);
      ctx[key] = c;
    }
    else
    {
        ctx[key].push_back(o);
    }

    return 0;
}

/**
 * @brief method to add fields of an ObjectType directly to the Context
 *
 * @param o ObjectType with fields
 *
 * @return 0
 */
int Context::add(const PlustacheTypes::ObjectType& o)
{
    for(PlustacheTypes::ObjectType::const_iterator it = o.begin();
        it != o.end(); it++)
    {
        (*this).add(it->first, it->second);
    }
    return 0;
}

/**
 * @brief method to get a value from the Context
 *
 * This is a generic getter which always returns a collection
 * (vector of maps) for a keyword. If the return value is a collection, the
 * collection is returned. If it is only a single value, a vector
 * with length 1 is returned. If the keyword wasn't found, a vector with
 * length 1 and an empty bucket for the keyword is returned.
 *
 * @param key
 *
 * @return collection for the keyword
 */
PlustacheTypes::CollectionType Context::get(const std::string& key) const
{
  PlustacheTypes::CollectionType ret;
  std::map<std::string, PlustacheTypes::CollectionType> :: const_iterator it;
  it = ctx.find(key);
  if (it == ctx.end())
  {
    PlustacheTypes::ObjectType o;
    o[key] = "";
    ret.push_back(o);
  }
  else
  {
    ret = it->second;
  }
  return ret;
}
