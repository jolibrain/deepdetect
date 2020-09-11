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

#include "deepdetect.h"
#ifdef USE_COMMAND_LINE
#ifdef USE_CAFFE
#include "commandlineapi.h"
#endif // USE_CAFFE
#ifdef USE_JSON_API
#include "commandlinejsonapi.h"
#endif // USE_JSON_API
#endif // USE_COMMAND_LINE
#if defined(USE_HTTP_SERVER) && defined(USE_JSON_API)
#include "httpjsonapi.h"
#endif // USE_HTTP_SERVER && USE_JSON_API
#if defined(USE_JSON_API) && !defined(USE_HTTP_SERVER)                        \
    && !defined(USE_COMMAND_LINE)
#include "jsonapi.h"
#endif
#include "dd_config.h"
#include "githash.h"

namespace dd
{
  template <class TAPIStrategy>
  std::string DeepDetect<TAPIStrategy>::_commit_version = GIT_COMMIT_HASH;

  template <class TAPIStrategy> DeepDetect<TAPIStrategy>::DeepDetect()
  {
    std::cout << "DeepDetect [ commit " << DeepDetect::_commit_version
              << " ]\n";
  }

  template <class TAPIStrategy> DeepDetect<TAPIStrategy>::~DeepDetect()
  {
  }

#ifdef USE_CAFFE
#ifdef USE_COMMAND_LINE
  template class DeepDetect<CommandLineAPI>;
#endif
#endif

#ifdef USE_JSON_API
#ifdef USE_COMMAND_LINE
  template class DeepDetect<CommandLineJsonAPI>;
#endif
#ifdef USE_HTTP_SERVER
  template class DeepDetect<HttpJsonAPI>;
#endif
#if !defined(USE_COMMAND_LINE) && !defined(USE_HTTP)
  template class DeepDetect<JsonAPI>;
#endif
#endif

}
