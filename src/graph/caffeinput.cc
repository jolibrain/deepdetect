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

#include "caffeinput.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "mllibstrategy.h"

using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;

namespace dd::graph
{

  bool CaffeInput::read_proto(std::string filename,
                              google::protobuf::Message *proto)
  {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1)
      return false;
    FileInputStream *input = new FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, proto);
    delete input;
    close(fd);
    return success;
  }

  bool CaffeInput::lstm_preparation(caffe::NetParameter &net, int i)
  {
    caffe::LayerParameter lparam = net.layer(i);
    int ninput = 1;
    if (net.layer(1).type() == "MemoryData")
      ninput = 2;
    if (i == ninput && lparam.type() == "Permute")
      return true;
    if (i == (ninput + 1) && lparam.type() == "Slice")
      return true;
    if (i == (ninput + 2) && lparam.type() == "Flatten")
      return true;
    return false;
  }

  bool CaffeInput::is_simple_lstm(caffe::NetParameter &net)
  {
    // check if we are processing a simple lstm from dd_generators
    // ie (LSTM [;affine] ) * n
    int ninput = 1;
    int firstl = 0;
    if (net.layer(1).type() == "MemoryData")
      ninput = 2;
    if (net.layer(ninput).type() != "Permute"
        || net.layer(ninput + 1).type() != "Slice"
        || net.layer(ninput + 2).type() != "Flatten")
      return false;
    if (net.layer(ninput + 3).type() == "Flatten")
      firstl = ninput + 4;
    else
      firstl = ninput + 3;

    if (net.layer(firstl).type() == "DummyData")
      firstl++;

    caffe::LayerParameter lparam = net.layer(firstl);
    if (lparam.type() != "LSTM" && lparam.type() != "RNN"
        && lparam.type() != "InnerProduct")
      return false;
    return true;
  }

  bool CaffeInput::parse_simple_lstm(caffe::NetParameter &net)
  {
    if (!is_simple_lstm(net))
      return false;

    bool first_lstm = true;
    int nlayers = net.layer_size();

    for (int i = 0; i < nlayers; ++i)
      {
        caffe::LayerParameter lparam = net.layer(i);

        if (lparam.include_size() != 0
            && lparam.include(0).phase() == ::caffe::TRAIN)
          continue;
        if (lparam.type() == "MemoryData")
          {
            std::vector<int> dims;
            dims.push_back(lparam.memory_data_param().batch_size());
            dims.push_back(lparam.memory_data_param().channels());
            dims.push_back(lparam.memory_data_param().height());
            set_input(lparam.top(0), dims);
          }
        else if (lstm_preparation(net, i))
          {
            continue;
          }
        else if (lparam.type() == "LSTM" || lparam.type() == "RNN")
          {
            Vertex v = add_layer(lparam.name(), lparam.type());
            _graph[v].num_output = lparam.recurrent_param().num_output();
            std::vector<std::string> inputs;
            if (first_lstm)
              {
                first_lstm = false;
                inputs.push_back(_inputname);
              }
            else
              inputs.push_back(lparam.bottom(0));
            add_inputs(v, inputs);

            std::vector<std::string> outputs;
            for (int i = 0; i < lparam.top_size(); ++i)
              outputs.push_back(lparam.top(i));
            add_outputs(v, outputs);
            set_output_name(lparam.top(0));
          }
        else if (lparam.type() == "InnerProduct")
          {
            Vertex v = add_layer(lparam.name(), lparam.type());
            _graph[v].num_output = lparam.inner_product_param().num_output();
            _graph[v].axis = lparam.inner_product_param().axis();

            std::vector<std::string> inputs;
            if (first_lstm)
              {
                first_lstm = false;
                inputs.push_back(_inputname);
              }
            else
              inputs.push_back(lparam.bottom(0));
            add_inputs(v, inputs);

            std::vector<std::string> outputs;
            outputs.push_back(lparam.top(0));
            add_outputs(v, outputs);
            set_output_name(lparam.top(0));
          }
        else if (lparam.type() == "Tile")
          {
            Vertex v = add_layer(lparam.name(), lparam.type());
            std::vector<std::string> inputs;
            inputs.push_back(lparam.bottom(0));
            add_inputs(v, inputs);
            std::vector<std::string> outputs;
            outputs.push_back(lparam.top(0));
            add_outputs(v, outputs);
            set_output_name(lparam.top(0));
            _graph[v].axis = lparam.tile_param().axis();
            _graph[v].tiles = lparam.tile_param().tiles();
          }
        else if (lparam.type() == "ReLU")
          {
            Vertex v = add_layer(lparam.name(), lparam.type());
            std::vector<std::string> inputs;
            inputs.push_back(lparam.bottom(0));
            add_inputs(v, inputs);
            std::vector<std::string> outputs;
            outputs.push_back(lparam.top(0));
            add_outputs(v, outputs);
            set_output_name(lparam.top(0));
          }
      }

    return true;
  }

  void CaffeInput::from_proto(std::string filename)
  {
    caffe::NetParameter net;
    if (!read_proto(filename, &net))
      throw MLLibBadParamException("unable to parse protofile");

    bool simple_lstm = is_simple_lstm(net);
    if (simple_lstm)
      {
        parse_simple_lstm(net);
        return;
      }
    throw MLLibBadParamException(
        "proto file do not contain a proper LSTM/autoencoder");
  }
}
