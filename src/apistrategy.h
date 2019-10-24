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
#include <spdlog/spdlog.h>

#ifdef USE_DD_SYSLOG
#include <spdlog/sinks/syslog_sink.h>
#else
#include <spdlog/sinks/stdout_sinks.h>
#endif

namespace dd
{

  /**
   * \brief main API class, built on top of Services
   */
  class APIStrategy : public Services
    {
    public:
      APIStrategy()
	{
#ifdef USE_DD_SYSLOG
	  _logger = spdlog::syslog_logger("api");
#else
	  _logger = spdlog::stdout_logger_mt("api");
#endif
	};
      ~APIStrategy()
	{
	  spdlog::drop("api");
	}
      
      /**
       * \brief handling of command line parameters
       */
      int boot(int argc, char *argv[]);
      std::shared_ptr<spdlog::logger> _logger; /**< api logger. */
    };
}

#endif
