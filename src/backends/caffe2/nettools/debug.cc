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

#include <fstream>
#include "backends/caffe2/nettools/internal.h"

namespace dd {
  namespace Caffe2NetTools {

    /*
     *  Debug
     */

    void net_to_svg(const caffe2::NetDef &net, const std::string &path) {
      int size = std::min(net.op().size(), 600);
      auto graph = path + ".tmpgraph";
      std::ofstream file(graph);
      file << "digraph {" << std::endl;
      file << '\t' << "node [shape=box];";
      for (int i = 0; i < size; ++i) {
	file << ' ' << net.op(i).type() + '_' + std::to_string(i);
      }
      file << "; node [shape=oval];" << std::endl;
      for (int i = 0; i < size; ++i) {
	const caffe2::OperatorDef &op = net.op(i);
	std::string name = op.type() + '_' + std::to_string(i);
	for (const std::string &blob : op.input()) {
	  file << "\t\"" << blob << "\" -> \"" << name << "\";" << std::endl;
	}
	for (const std::string &blob : op.output()) {
	  file << "\t\"" << name << "\" -> \"" << blob << "\";" << std::endl;
	}
      }
      file << "}" << std::endl;
      file.close();
      system(("dot -Tsvg -o" + path + " " + graph).c_str());
      remove(graph.c_str());
    }

    void dump_net(const caffe2::NetDef &net, const std::string &path) {

      // Raw protobuf file
      std::ofstream f(path + ".pb");
      net.SerializeToOstream(&f);

      // Human-readable protobuf file
      std::ofstream(path + ".pbtxt") << net.DebugString();

      // SVG file
      net_to_svg(net, path + ".svg");
    }

    void untrain_model(const std::string &input, const std::string &output) {
      caffe2::NetDef net, init;
      CAFFE_ENFORCE(caffe2::ReadProtoFromFile(input + "/predict_net.pb", &net));
      CAFFE_ENFORCE(caffe2::ReadProtoFromFile(input + "/init_net.pb", &init));
      reset_fillers(net, init);
      std::ofstream(output + "/predict_net.pbtxt") << net.DebugString();
      std::ofstream(output + "/init_net.pbtxt") << init.DebugString();
    }

  }
}