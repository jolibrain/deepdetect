/**
 * DeepDetect
 * Copyright (c) 2014-2015 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
 *
 * This file is part of deepdetect.
 *
 * deepdetect is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deepdetect is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DD_UTILS
#define DD_UTILS

#include <fstream>
#include <vector>
#include <algorithm>

#include <boost/lexical_cast.hpp>
#include <rapidjson/allocators.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/reader.h>
#include <rapidjson/writer.h>

#include "apidata.h"
#include "dd_types.h"

namespace dd
{
  namespace dd_utils
  {
    // from
    // https://stackoverflow.com/questions/47496358/c-lambdas-how-to-capture-variadic-parameter-pack-from-the-upper-scope
    // see also:
    // https://en.cppreference.com/w/cpp/experimental/apply
    namespace detail
    {
      template <class F, class Tuple, std::size_t... I>
      inline constexpr decltype(auto) apply_impl(F &&f, Tuple &&t,
                                                 std::index_sequence<I...>)
      {
        // return std::invoke(std::forward<F>(f),
        //                   std::get<I>(std::forward<Tuple>(t))...);
        // Note: std::invoke is a C++17 feature
        return std::forward<F>(f)(std::get<I>(static_cast<Tuple &&>(t))...);
      }
    } // namespace detail

    template <class F, class Tuple>
    inline constexpr decltype(auto) apply(F &&f, Tuple &&t)
    {
      return detail::apply_impl(
          std::forward<F>(f), std::forward<Tuple>(t),
          std::make_index_sequence<
              std::tuple_size<std::decay_t<Tuple>>::value>{});
    }

    inline std::vector<std::string> split(const std::string &s, char delim)
    {
      std::vector<std::string> elems;
      std::stringstream ss(s);
      std::string item;
      while (std::getline(ss, item, delim))
        {
          if (!item.empty())
            elems.push_back(item);
        }
      return elems;
    }

    inline std::string trim_spaces(const std::string &item)
    {
      const std::string WHITESPACE = " \n\r\t\f\v";
      size_t start = item.find_first_not_of(WHITESPACE);
      size_t end = item.find_last_not_of(WHITESPACE);

      return start == std::string::npos || end == std::string::npos
                 ? ""
                 : item.substr(start, end - start + 1);
    }

    inline std::string jrender(const JDoc &jst)
    {
      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>,
                        rapidjson::UTF8<>, rapidjson::CrtAllocator,
                        rapidjson::kWriteNanAndInfFlag>
          writer(buffer);
      bool done = jst.Accept(writer);
      if (!done)
        throw DataConversionException("JSON rendering failed");
      return buffer.GetString();
    }

    inline std::string jrender(const JVal &jval)
    {
      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer, rapidjson::UTF8<>,
                        rapidjson::UTF8<>, rapidjson::CrtAllocator,
                        rapidjson::kWriteNanAndInfFlag>
          writer(buffer);
      bool done = jval.Accept(writer);
      if (!done)
        throw DataConversionException("JSON rendering failed");
      return buffer.GetString();
    }

    inline bool iequals(const std::string &a, const std::string &b)
    {
      unsigned int sz = a.size();
      if (b.size() != sz)
        return false;
      for (unsigned int i = 0; i < sz; ++i)
        if (std::tolower(a[i]) != std::tolower(b[i]))
          return false;
      return true;
    }

    inline bool unique(int64_t val, std::vector<int64_t> vec)
    {
      size_t count = 0;
      for (int64_t v : vec)
        {
          if (v == val)
            count++;
        }
      return count == 1;
    }

    /** boost::lexical_cast<bool> but accept "true" and "false" */
    inline bool parse_bool(const std::string &str)
    {
      try
        {
          return boost::lexical_cast<bool>(str);
        }
      catch (boost::bad_lexical_cast &)
        {
          if (str == "true")
            return true;
          else if (str == "false")
            return false;
          else
            throw;
        }
      return false;
    }

#ifdef WIN32
    inline int my_hardware_concurrency()
    {
      SYSTEM_INFO si;
      GetSystemInfo(&si);
      return si.dwNumberOfProcessors;
    }
#else
    inline int my_hardware_concurrency()
    {
      std::ifstream cpuinfo("/proc/cpuinfo");

      return std::count(std::istream_iterator<std::string>(cpuinfo),
                        std::istream_iterator<std::string>(),
                        std::string("processor"));
    }
#endif
  };
}

#endif
