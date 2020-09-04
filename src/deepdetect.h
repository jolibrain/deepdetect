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

#ifndef DEEPDETECT_H
#define DEEPDETECT_H

#include "apistrategy.h"
#include <vector>

namespace dd
{

  /**
   * \brief main deepdetect class, deriving API
   */
  template <class TAPIStrategy> class DeepDetect : public TAPIStrategy
  {
  public:
    DeepDetect();
    ~DeepDetect();

    static std::string
        _commit_version; /**< stores the current commit version */
  };

}

#endif
