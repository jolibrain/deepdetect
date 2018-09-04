/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Julien Chicha
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

#include "backends/caffe2/nettools.h"

int main(int argc, char **argv) {

  std::string usage = *argv;
  usage += " net svg\n\n"
    "Transform a net protobuf into a graph.\n\n"
    "net   Path of the net\n"
    "svg   Path of the SVG";
  auto error = [&usage](){
    std::cerr << usage << std::endl;
    exit(1);
  };

  if (argc != 3) error();

  caffe2::NetDef net;
  if (!caffe2::ReadProtoFromFile(argv[1], &net)) {
    std::cerr << "Could not read : " << argv[1] << std::endl;
    exit(1);
  }
  dd::Caffe2NetTools::net_to_svg(net, argv[2]);
  return 0;
}
