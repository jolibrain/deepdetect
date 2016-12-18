/**
 * DeepDetect
 * Copyright (c) 2014-2016 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
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

#include "net_caffe.h"

namespace dd
{

  /*- NetInputCaffe -*/
  void NetInputCaffe::configure_inputs(const APIData &ad)
  {
    /*caffe::LayerParameter *lparam = _net_param.add_layer(); // train net input layer
    caffe::LayerParameter *dlparam = nullptr;
    std::string top = "data";
    std::string label = "label";
    if (targets > 1)
      {
	top = "fulldata";
	label = "fake_label";
      }

    // deploy net
    dlparam = _dnet_param.add_layer(); // test net input layer
    dlparam->set_name("data");
    dlparam->add_top(top);
    dlparam->add_top(label);

    // train net
    lparam->set_name("data");
    lparam->add_top(top);
    lparam->add_top(label);
    caffe::NetStateRule *nsr = lparam->add_include();
    nsr->set_phase(caffe::TRAIN);

    // sources
    if (db)
      {
	lparam->set_type("Data");
	caffe::TransformationParameter *tparam = lparam->mutable_transform_param();
	if (ad.has("rotate"))
	  tparam->set_mirror(ad.get("rotate").get<bool>());
	if (ad.has("mirror"))
	  tparam->set_rotate(ad.get("mirror").get<bool>());
	if (ad.has("crop_size"))
	  tparam->set_crop_size(ad.get("crop_size").get<int>());
	if (has_mean_file)
	  tparam->set_mean_file("mean.binaryproto");
	caffe::DataParameter *dparam = lparam->mutable_data_param();
	dparam->set_source("train.lmdb");
	dparam->set_batch_size(32); // dummy value, updated before training
	dparam->set_backend(caffe::DataParameter_DB_LMDB);
      }
    else
      {
	lparam->set_type("MemoryData");
	caffe::MemoryDataParameter *mdparam = lparam->mutable_memory_data_param();
	mdparam->set_batch_size(32); // dummy value, updated before training
	if (!text)
	  mdparam->set_channels(channels);
	mdparam->set_height(height);
	mdparam->set_width(width);
      }
    if (!deploy)
      {
	dlparam->set_type("MemoryData");
	caffe::NetStateRule *dnsr = dlparam->add_include();
	dnsr->set_phase(caffe::TEST);
	caffe::MemoryDataParameter *mdparam = dlparam->mutable_memory_data_param();
	mdparam->set_batch_size(32); // dummy value, updated before training
	mdparam->set_channels(channels);
	mdparam->set_height(height);
	mdparam->set_width(width);
      }
    if (targets > 1)
      {
	lparam = add_layer(net_param,top,"data");
	lparam->add_top("label");
	lparam->set_type("Slice");
	lparam->set_name("slice_labels");
	caffe::SliceParameter *sparam = lparam->mutable_slice_param();
	sparam->set_slice_dim(1);
	sparam->add_slice_point(nclasses); // dummy value, updated before training
	}*/
  }

  /*- NetLayersCaffe -*/

  /*- NetLossCaffe -*/

  /*- NetCaffe -*/
  template<class TNetInputCaffe, class TNetLayersCaffe, class TNetLossCaffe>
  NetCaffe<TNetInputCaffe,TNetLayersCaffe,TNetLossCaffe>::NetCaffe(const int &nclasses)
    :_nclasses(nclasses)
  {
    
  }
  
}
