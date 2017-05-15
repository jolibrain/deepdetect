/**
 * DeepDetect
 * Copyright (c) 2017 Emmanuel Benazera
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

#include "tsneinputconns.h"

namespace dd
{
  void CSVTSNEInputFileConn::transform(const APIData &ad)
  {
    try
      {
	CSVInputFileConn::transform(ad);
      }
    catch (std::exception &e)
      {
	throw;
      }
    _N = _csvdata.size();
    _D = _csvdata.at(0)._v.size();
    _X = dMatR::Zero(_N,_D);
    for (int i=0;i<_N;i++)
      {
	for (int j=0;j<_D;j++)
	  {
	    _X(i,j) = _csvdata[i]._v[j];
	  }
      }
    _csvdata.clear();
  }
}
