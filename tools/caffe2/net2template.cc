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
  usage += " input_dir output_dir\n\n"
    "Reset a model repository to it's \"template\".\n\n"
    "input_dir    Where the 'predict_net.pb' and 'init_net.pb' files are placed\n"
    "output_dir   Where the 'predict_net.pbtxt' and 'init_net.pbtxt' files must be created";
  auto error = [&usage](){
    std::cerr << usage << std::endl;
    exit(1);
  };

  if (argc != 3) error();
  dd::Caffe2NetTools::untrain_model(argv[1], argv[2]);
  return 0;
}
