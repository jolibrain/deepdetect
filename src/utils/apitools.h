/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Julien Chicha
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

#ifndef APITOOLS_H
#define APITOOLS_H

#include "apidata.h"

namespace dd
{
  namespace apitools
  {

    template <typename T>
    inline bool get_variant(const ad_variant_type &adv,
                            const std::function<void(const T &)> &f)
    {
      if (!adv.is<T>())
        {
          return false;
        }
      f(adv.get<T>());
      return true;
    }

    // Simple way to retrieve a value from an APIData when the type is
    // uncertain
    template <typename T1, typename T2>
    inline void test_variants(const APIData &ad, const std::string &name,
                              const std::function<void(const T1 &)> &f1,
                              const std::function<void(const T2 &)> &f2)
    {
      const ad_variant_type &adv = ad.get(name);
      if (get_variant<T1>(adv, f1))
        ;
      else if (get_variant<T2>(adv, f2))
        ;
      else
        throw std::runtime_error("Invalid type for '" + std::string(name)
                                 + "'");
    }

    // Try to cast a single value
    template <typename T1, typename T2, typename Out>
    inline void get_with_cast(const APIData &ad, const std::string &name,
                              Out &out)
    {
      test_variants<T1, T2>(
          ad, name, [&](const T1 &v) { out = static_cast<Out>(v); },
          [&](const T2 &v) { out = static_cast<Out>(v); });
    }

    // Try to cast a vector
    template <typename T1, typename T2, typename Out>
    inline void get_with_cast(const APIData &ad, const std::string &name,
                              std::vector<Out> &out)
    {
      test_variants<std::vector<T1>, std::vector<T2>>(
          ad, name,
          [&](const std::vector<T1> &v) { out.assign(v.begin(), v.end()); },
          [&](const std::vector<T2> &v) { out.assign(v.begin(), v.end()); });
    }

    // Floats can be fetched from doubles and integers
    inline void get_float(const APIData &ad, const std::string &name, float &v)
    {
      get_with_cast<double, int, float>(ad, name, v);
    }
    inline void get_floats(const APIData &ad, const std::string &name,
                           std::vector<float> &v)
    {
      get_with_cast<double, int, float>(ad, name, v);
    }

    // Try as one or several values
    template <typename Out>
    inline void get_vector(const APIData &ad, const std::string &name,
                           std::vector<Out> &out)
    {
      test_variants<Out, std::vector<Out>>(
          ad, name, [&](const Out &v) { out = { v }; },
          [&](const std::vector<Out> &v) { out = v; });
    }

  }
}

#endif
