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

#ifndef GRAPH_H
#define GRAPH_H

#include <boost/graph/adjacency_list.hpp>

#include "caffeinput.h"
#ifdef USE_TORCH
#include "backends/torch/torchgraphbackend.h"
#endif

namespace dd::graph
{

  /**
   * this class is s simple holder for having a basegraph with
   * some traits. in our case mainly, from_proto and forward
   */

  template <class TInput, class TBackend>
  class Graph : public TInput, public TBackend
  {
  };

  template <class TBackend>
  class Graph<CaffeInput, TBackend> : public CaffeInput, public TBackend
  {
  public:
    Graph(std::string protofilename);
  };
#ifdef USE_TORCH
  template <>
  class Graph<CaffeInput, dd::TorchGraphBackend> : public CaffeInput,
                                                   public TorchGraphBackend
  {
  public:
    Graph(std::string protofilename, std::vector<int> inputdim)
        : CaffeInput(protofilename)
    {
      finalize(inputdim);
    }
    Graph(std::string protofilename) : CaffeInput(protofilename)
    {
      // we should no finalize in case input dims has not been specified
      //	  finalize();
    }
  };
}
template class dd::graph::Graph<dd::graph::CaffeInput, dd::TorchGraphBackend>;
typedef dd::graph::Graph<dd::graph::CaffeInput, dd::TorchGraphBackend>
    CaffeToTorch;
#endif
#endif
