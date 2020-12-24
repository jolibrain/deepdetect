/**
 * DeepDetect
 * Copyright (c) 2014 Emmanuel Benazera
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

#ifndef DDTYPES_H
#define DDTYPES_H

#include <exception>
class RapidjsonException : public std::exception
{
public:
  RapidjsonException(const char *s) : _s(s)
  {
  }
  ~RapidjsonException()
  {
  }

  const char *what() const noexcept
  {
    return _s;
  }

private:
  const char *_s;
};

#undef RAPIDJSON_ASSERT
#define RAPIDJSON_ASSERT(x)                                                   \
  if (!(x))                                                                   \
  throw RapidjsonException(RAPIDJSON_STRINGIFY(x))

//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wterminate"
//#pragma GCC diagnostic ignored "-Wsign-compare"
//#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <rapidjson/document.h>
//#pragma GCC diagnostic pop

typedef rapidjson::Document JDoc;
typedef rapidjson::Value JVal;

#endif
