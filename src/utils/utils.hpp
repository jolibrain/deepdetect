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

namespace dd
{
  class dd_utils
  {
  public:
    static std::vector<std::string> split(const std::string &s, char delim)
    {
      std::vector<std::string> elems;
      std::stringstream ss(s);
      std::string item;
      while (std::getline(ss, item, delim)) {
	if (!item.empty())
	  elems.push_back(item);
      }
      return elems;
    }
    
    static bool iequals(const std::string& a, const std::string& b)
    {
      unsigned int sz = a.size();
      if (b.size() != sz)
	return false;
      for (unsigned int i = 0; i < sz; ++i)
	if (std::tolower(a[i]) != std::tolower(b[i]))
	  return false;
      return true;
    }

#ifdef WIN32
    static int my_hardware_concurrency()
    {
      SYSTEM_INFO si;
      GetSystemInfo(&si);
      return si.dwNumberOfProcessors;
    }
#else
    static int my_hardware_concurrency()
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
