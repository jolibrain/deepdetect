/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
 * Author:  Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#ifndef TTYPES_H
#define TTYPES_H

namespace dd
{
  enum PEType
  {
    naive,  // simple (normalized) index
    sincos, // original sin/cos freq encoding
    none
  };

  enum PEAggregation
  {
    sum,
    cat
  };

  enum Activation
  {
    relu,
    gelu,
    siren
  };

  enum EmbedType
  {
    step,
    serie,
    all
  };
}
#endif
