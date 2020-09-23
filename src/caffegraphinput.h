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

#ifndef CAFFE_GRAPH_INPUT_H
#define CAFFE_GRAPH_INPUT_H

#include <google/protobuf/text_format.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "src/caffe.pb.h"
#pragma GCC diagnostic pop

#include "basegraph.h"

namespace dd
{

  /**
   *  this class add from_proto trait to basegraph, ie allows to build a base
   * graph from a prototxt file
   */
  class CaffeGraphInput : public virtual BaseGraph
  {
  public:
    /**
     *  simple contructor to build a basegraph from a prototxt
     */
    CaffeGraphInput(std::string filename)
    {
      from_proto(filename);
    }

  private:
    /**
     * create basegraph from proto
     */
    int from_proto(std::string filename);

    /**
     * read protofile
     */
    bool read_proto(std::string filename, google::protobuf::Message *proto);

    /**
     * check if we are in all permute / ssplit / concat stuff needed by caffe
     * before lstm
     * @return
     */
    bool lstm_preparation(caffe::NetParameter &net, int i);

    /**
     * check if protofile is an lstm definition created by dede
     */
    bool is_simple_lstm(caffe::NetParameter &net);

    /**
     * create basegraph from lstm protofile created by dede
     */
    bool parse_simple_lstm(caffe::NetParameter &net);
  };
}
#endif
