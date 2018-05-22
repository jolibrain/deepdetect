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

//XXX Remove that to print the warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"

#ifndef CPU_ONLY
#include <caffe2/core/context_gpu.h>
#endif

#include <caffe2/core/db.h>
#pragma GCC diagnostic pop

#include "caffe2/core/typeid.h"
#include "backends/caffe2/nettools/internal.h"

namespace dd {
  namespace Caffe2NetTools {

    /*
     *  Workspace management
     */

    bool extract_tensor(const caffe2::Workspace &workspace, const caffe2::DeviceOption &device,
			const std::string &name, caffe2::TensorCPU &tensor) {
      const caffe2::Blob &blob = *workspace.GetBlob(name);
      if (!blob.meta().id()) {
	return false; // nullptr (uninitialized)
      }
#ifndef CPU_ONLY
      if (device.device_type() == caffe2::CUDA) {
	tensor.CopyFrom(blob.Get<caffe2::TensorCUDA>());
      } else
#endif
	tensor.CopyFrom(blob.Get<caffe2::TensorCPU>());
      return true;
    }

    void insert_tensor(caffe2::Workspace &workspace,
		       const caffe2::DeviceOption &device,
		       const std::string &name,
		       const caffe2::TensorCPU &tensor) {
      caffe2::Blob &blob = *workspace.CreateBlob(name);
#ifndef CPU_ONLY
      if (device.device_type() == caffe2::CUDA) {
	blob.GetMutable<caffe2::TensorCUDA>()->CopyFrom(tensor);
      } else
#endif
	blob.GetMutable<caffe2::TensorCPU>()->CopyFrom(tensor);
    }

    static bool deserialize_blob(caffe2::Blob &blob, const std::string &path) {
      std::ifstream f(path);
      CAFFE_ENFORCE(f.good());
      std::stringstream s;
      s << f.rdbuf();
      std::string str(s.str());
      if (str == caffe2::TypeMeta().name()) {
	return false; // nullptr (uninitialized)
      }
      blob.Deserialize(str);
      return true;
    }

    float extract_loss(const caffe2::Workspace &workspace,
		       const std::vector<caffe2::DeviceOption> &devices) {
      float loss = 0;
      caffe2::TensorCPU tensor;
      for (const caffe2::DeviceOption &device : devices) {
	CAFFE_ENFORCE(extract_tensor(workspace, device,
				     get_device_prefix(device) + blob_loss_scale, tensor));
	loss += *tensor.data<float>();
      }
      return loss;
    }

    int extract_iter(const caffe2::Workspace &workspace, const caffe2::DeviceOption &device) {
      return *workspace.GetBlob(get_device_prefix(device) + blob_iter)
	->Get<caffe2::TensorCPU>().data<long>();
    }

    void extract_state(const caffe2::Workspace &workspace,
		       const caffe2::DeviceOption &device,
		       std::map<std::string, std::string> &blobs) {
      for (const std::string &name : workspace.Blobs()) {
    	const caffe2::Blob &blob = *workspace.GetBlob(name);
    	if (blob.IsType<caffe2::db::DBReader>()) {
	  blobs[name] = blob.Serialize(name);
	}
      }
      blobs["iter"] = workspace.GetBlob(get_device_prefix(device) + blob_iter)->Serialize("iter");
      caffe2::Blob lr;
      if (extract_tensor(workspace, device, get_device_prefix(device) + blob_lr,
			 *lr.GetMutable<caffe2::TensorCPU>())) {
	blobs["lr"] = lr.Serialize("lr");
      } else {
	blobs["lr"] = caffe2::TypeMeta().name(); // nullptr (uninitialized)
      }
    }

    void create_init_net(const caffe2::Workspace &workspace,
    			 const caffe2::DeviceOption &device,
    			 const caffe2::NetDef &net,
    			 caffe2::NetDef &init) {
      std::vector<std::string> params;
      caffe2::TensorCPU tensor;
      collect_params(net, params, params, get_device_prefix(device));
      for (const std::string &param : params) {
	CAFFE_ENFORCE(extract_tensor(workspace, device, get_device_prefix(device) + param, tensor));
	const float *data = tensor.data<float>();
	const std::vector<long int> &dims = tensor.dims();
	GivenTensorFill(init, param,
			std::vector<int>(dims.begin(), dims.end()),
			std::vector<float>(data, data + tensor.size()));
      }
    }

    int load_iter(const std::string &path) {
      caffe2::Blob blob;
      CAFFE_ENFORCE(deserialize_blob(blob, path));
      return *blob.Get<caffe2::TensorCPU>().data<long>();
    }

    void load_blob(caffe2::Workspace &workspace,
		   const std::string &path,
		   const std::string &name) {
      CAFFE_ENFORCE(deserialize_blob(*workspace.CreateBlob(name), path));
    }

    static void broadcast_blob(caffe2::Workspace &workspace, 
			       const std::vector<caffe2::DeviceOption> &devices,
			       const caffe2::Blob &blob,
			       const std::string &name) {
      const caffe2::TensorCPU &tensor = blob.Get<caffe2::TensorCPU>();
      for (const caffe2::DeviceOption &device : devices) {
	insert_tensor(workspace, device, get_device_prefix(device) + name, tensor);
      }
    }

    void load_lr(caffe2::Workspace &workspace,
		 const std::vector<caffe2::DeviceOption> &devices,
		 const std::string &path) {
      caffe2::Blob blob;
      // A learning rate serialized during the first iteration is still uninitialized
      // In that case we just ignore it
      if (deserialize_blob(blob, path)) {
	broadcast_blob(workspace, devices, blob, blob_lr);
      }
    }

  }
}
