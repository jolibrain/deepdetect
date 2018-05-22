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

#include "backends/caffe2/nettools/internal.h"

namespace dd {
  namespace Caffe2NetTools {

    /*
     *  XXX Unclassified tools
     */

    void truncate_net(const caffe2::NetDef &net, caffe2::NetDef &out, const std::string &blob) {
      auto ops = net.op();
      int last_op = ops.size() - 1; // Index of the last operator that updates 'blob'

      while (true) {
	caffe2::OperatorDef &op = ops[last_op];
	auto outputs = op.output();
	if (std::find(outputs.begin(), outputs.end(), blob) != outputs.end()) {
	  break;
	}
	if (!last_op--) {
	  CAFFE_THROW("Blob '", blob, "' not found");
	}
      }

      // Fill the new net
      for (int i = 0; i <= last_op; ++i) {
	out.add_op()->CopyFrom(ops[i]);
      }
      out.mutable_external_input()->CopyFrom(net.external_input());
      out.add_external_output(blob);
    }

    void truncate_net(caffe2::NetDef &net, const std::string &blob) {
      caffe2::NetDef short_net;
      truncate_net(net, short_net, blob);
      net.Swap(&short_net);
    }

  }
}
