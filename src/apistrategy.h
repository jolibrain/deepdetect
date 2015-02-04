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

#ifndef APISTRATEGY_H
#define APISTRATEGY_H

#include "dd_types.h"
#include "services.h"

namespace dd
{

  /**
   * \brief main API class, built on top of Services
   */
  class APIStrategy : public Services
    {
    public:
      APIStrategy() {};
      ~APIStrategy() {};
      
      /**
       * \brief handling of command line parameters
       */
      int boot(int argc, char *argv[]);
    };
}

#endif
