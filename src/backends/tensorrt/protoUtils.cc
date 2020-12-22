
// protoUtils.cc ---

// Copyright (C) 2019 Jolibrain http://www.jolibrain.com

// Author: Guillaume Infantes <guillaume.infantes@jolibrain.com>

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "protoUtils.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "src/caffe.pb.h"
#pragma GCC diagnostic pop

using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;

namespace dd
{

  bool findInputDimensions(const std::string &source, int &width, int &height)
  {
    caffe::NetParameter net;
    if (!TRTReadProtoFromTextFile(source.c_str(), &net))
      return false;

    for (int i = 0; i < net.layer_size(); ++i)
      {
        caffe::LayerParameter lparam = net.layer(i);
        if (lparam.type() == "MemoryData")
          {
            width = lparam.memory_data_param().width();
            height = lparam.memory_data_param().height();
          }
        break; // don't go further than first layer.
      }

    return true;
  }

  int findNClasses(const std::string source, bool bbox)
  {
    caffe::NetParameter net;
    if (!TRTReadProtoFromTextFile(source.c_str(), &net))
      return -1;
    int nlayers = net.layer_size();
    if (bbox)
      {
        for (int i = nlayers - 1; i >= 0; --i)
          {
            caffe::LayerParameter lparam = net.layer(i);
            if (lparam.type() == "DetectionOutput")
              return lparam.detection_output_param().num_classes();
          }
      }
    for (int i = nlayers - 1; i >= 0; --i)
      {
        caffe::LayerParameter lparam = net.layer(i);
        if (lparam.type() == "InnerProduct")
          return lparam.inner_product_param().num_output();

        if (net.layer(0).name() == "squeezenet"
            && lparam.type() == "Convolution")
          return lparam.convolution_param().num_output();
      }
    return -1;
  }

  int findTopK(const std::string source)
  {
    caffe::NetParameter net;
    if (!TRTReadProtoFromTextFile(source.c_str(), &net))
      return -1;
    int nlayers = net.layer_size();
    for (int i = nlayers - 1; i >= 0; --i)
      {
        caffe::LayerParameter lparam = net.layer(i);
        if (lparam.type() == "DetectionOutput")
          return lparam.detection_output_param().nms_param().top_k();
      }
    return -1;
  }

  int fixProto(const std::string dest, const std::string source)
  {
    caffe::NetParameter source_net;
    caffe::NetParameter dest_net;
    if (!TRTReadProtoFromTextFile(source.c_str(), &source_net))
      return 1;

    dest_net.set_name(source_net.name());
    int nlayers = source_net.layer_size();

    for (int i = 0; i < nlayers; ++i)
      {
        caffe::LayerParameter lparam = source_net.layer(i);
        if (lparam.type() == "MemoryData")
          {
            dest_net.add_input(lparam.top(0));
            caffe::BlobShape *is = dest_net.add_input_shape();
            is->add_dim(lparam.memory_data_param().batch_size());
            is->add_dim(lparam.memory_data_param().channels());
            is->add_dim(lparam.memory_data_param().height());
            is->add_dim(lparam.memory_data_param().width());
          }
        else if (lparam.type() == "Flatten")
          {
            caffe::LayerParameter *rparam = dest_net.add_layer();
            rparam->set_name(lparam.name());
            rparam->set_type("Reshape");
            rparam->add_bottom(lparam.bottom(0));
            rparam->add_top(lparam.top(0));
            int faxis = lparam.flatten_param().axis();
            caffe::ReshapeParameter *rp = rparam->mutable_reshape_param();
            caffe::BlobShape *bs = rp->mutable_shape();
            for (int i = 0; i < faxis; ++i)
              bs->add_dim(0);
            bs->add_dim(-1);
            for (int i = faxis + 1; i < 4; ++i)
              bs->add_dim(1);
          }
        else if (lparam.type() == "DetectionOutput")
          {
            caffe::LayerParameter *dlparam = dest_net.add_layer();
            caffe::NonMaximumSuppressionParameter *nmsp
                = lparam.mutable_detection_output_param()->mutable_nms_param();
            nmsp->clear_soft_nms();
            nmsp->clear_theta();
            *dlparam = lparam;
            dlparam->add_top("keep_count");
          }
        else
          {
            caffe::LayerParameter *dlparam = dest_net.add_layer();
            *dlparam = lparam;
          }
      }

    if (!TRTWriteProtoToTextFile(dest_net, dest.c_str()))
      return 2;
    return 0;
  }

  bool TRTReadProtoFromTextFile(const char *filename,
                                google::protobuf::Message *proto)
  {
    int fd = open(filename, O_RDONLY);
    if (fd == -1)
      return false;
    FileInputStream *input = new FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, proto);
    delete input;
    close(fd);
    return success;
  }

  bool TRTWriteProtoToTextFile(const google::protobuf::Message &proto,
                               const char *filename)
  {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1)
      return false;
    FileOutputStream *output = new FileOutputStream(fd);
    bool success = google::protobuf::TextFormat::Print(proto, output);
    delete output;
    close(fd);
    return success;
  }

}
