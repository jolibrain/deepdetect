/**
 * DeepDetect
 * Copyright (c) 2014 Emmanuel Benazera
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

#include "caffelib.h"
#include "imginputfileconn.h"
#include "outputconnectorstrategy.h"
#include <iostream>

using caffe::Caffe;
using caffe::Net;
using caffe::Blob;
using caffe::Datum;

namespace dd
{

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::CaffeLib(const CaffeModel &cmodel)
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,CaffeModel>(cmodel)
  {
    if (_gpu)
      {
	Caffe::SetDevice(_gpuid);
	Caffe::set_mode(Caffe::GPU);
      }
    else Caffe::set_mode(Caffe::CPU);
    if (this->_has_predict)
      Caffe::set_phase(Caffe::TEST); // XXX: static within Caffe, cannot go along with training across services.
    _net = new Net<float>(this->_mlmodel._def);
    _net->CopyTrainedLayersFrom(this->_mlmodel._weights);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::CaffeLib(CaffeLib &&cl) noexcept
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,CaffeModel>(std::move(cl))
  {
    _gpu = cl._gpu;
    _gpuid = cl._gpuid;
    _net = cl._net;
    cl._net = nullptr;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~CaffeLib()
  {
    if (_net)
      delete _net; // XXX: for now, crashes.
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad)
  {
    //TODO: get blobs out of input connector.
    //this->_inputc.transform(); //TODO: instanciate local input connector around ad.
    /*std::vector<cv::Mat> imgs(1,this->_inputc._image);
    std::vector<int> labels(1,0);
    const std::shared_ptr<caffe::ImageDataLayer<float>> image_data_layer
      = boost::static_pointer_cast<caffe::ImageDataLayer<float>>(_net.layer_by_name("data")); // XXX: std::static_pointer_cast does not work here
      image_data_layer->AddImagesAndLabels(imgs,labels);*/
    TInputConnectorStrategy inputc;
    inputc.transform(ad);
    Datum datum;
    //ReadImageToDatum(this->_inputc._imgfname,10,227,227,&datum);
    CVMatToDatum(this->_inputc._image,&datum);
    Blob<float> blob(1,datum.channels(),datum.height(),datum.width());
    std::vector<Blob<float>*> bottom = {&blob};
    float loss = 0.0;
    std::vector<Blob<float>*> results = _net->Forward(bottom,&loss);
    std::cout << "accuracy=" << results[0]->cpu_data()[0] << std::endl;
    
    /*int startl = 0;
    int endl = _net.layers().size()-1;
    double loss = _net.ForwardFromTo(startl,endl);
    std::cout << _net.net_output_blobs_.size() << std::endl;*/
    
    return 0;
  }

  template class CaffeLib<ImgInputFileConn,NoOutputConn,CaffeModel>;
}
