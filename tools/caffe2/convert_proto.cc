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

#define KP(name) {#name, new caffe2::name()}

const std::map<std::string, google::protobuf::Message*> types({
    KP(NetDef),
    KP(OperatorDef),
    KP(TensorProto),
    KP(TensorProtos)
});

inline void load_pb(google::protobuf::Message *&msg, const char *file, const std::string &type) {
  auto it = types.find(type);
  if (it == types.end()) {
    std::cerr << "Unknown type : " << type << std::endl;
    exit(1);
  }
  if (!caffe2::ReadProtoFromFile(file, it->second)) {
    std::cerr << "Could not read : " << file << std::endl;
    exit(1);
  }
  msg = it->second;
}

inline void dump_pb(google::protobuf::Message &msg, const char *file, bool bin) {
  std::ofstream f(file);
  if (bin) {
    msg.SerializeToOstream(&f);
  } else {
    f << msg.DebugString();
  }
}

int main(int argc, char **argv) {

  const char *type = "NetDef";
  const char *input = "";
  const char *output = "";
  bool bin = false;

  std::string usage = *argv;
  usage += " [-t TYPE] [-i INPUT] [-o OUTPUT] [-b]\n\n"
    "Converts a Protobuf message from binary to human-reable.\n\n"
    "-t TYPE     Protobuf object (set to 'NetDef' by default)\n"
    "-i INPUT    Input file, mandatory\n"
    "-o OUTPUT   Output file, mandatory\n"
    "-b          If set, convert into binary instead of human readable\n\n"
    "Known types: ";
  for (auto kp : types) {
    usage += "\n - " + kp.first;
  }
  auto error = [&usage](){
    std::cerr << usage << std::endl;
    exit(1);
  };

  char c;
  while ((c = getopt(argc, argv, "t:i:o:b")) != -1) {
    switch (c) {
    case 't':
      type = optarg; break;
    case 'i':
      input = optarg; break;
    case 'o':
      output = optarg; break;
    case 'b':
      bin = true; break;
    case '?': exit(1);
    default: error();
    }
  }

  if (!*type || !*input || !*output) error();

  google::protobuf::Message *msg;
  load_pb(msg, input, type);
  dump_pb(*msg, output, bin);
  return 0;
}
