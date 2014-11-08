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
    
    if (!this->_mlmodel._mean.empty())
      {
	std::cout << "setting mean\n";
	caffe::BlobProto blob_proto;
	std::cout << "mean file=" << this->_mlmodel._mean << std::endl;
	std::cout << "protoloading=" << ReadProtoFromBinaryFile(this->_mlmodel._mean,&blob_proto) << std::endl;
	std::cout << "blob proto data=" << blob_proto.data(0) << std::endl;
	_data_mean.FromProto(blob_proto);
	std::cout << "loaded mean cpu data=" << _data_mean.cpu_data() << std::endl;
      }
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::CaffeLib(CaffeLib &&cl) noexcept
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,CaffeModel>(std::move(cl))
  {
    _gpu = cl._gpu;
    _gpuid = cl._gpuid;
    _net = cl._net;
    if (cl._data_mean.num())
      {
	std::cout << "copying mean\n";
	_data_mean.CopyFrom(cl._data_mean,false,true);
      }
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
    //ReadImageToDatum(inputc._imgfname,1,227,227,&datum);
    CVMatToDatum(inputc._image,&datum);
    //std::cout << "datum height=" << datum.height() << std::endl;
    //Blob<float> *blob = new Blob<float>(1,datum.channels(),datum.height(),datum.width()); 
    /*if (!this->_mlmodel._mean.empty())
      {
	std::cout << "sub mean\n";
	std::cout << "input num=" << blob.num() << std::endl;
	std::cout << "data mean num=" << _data_mean.num() << std::endl;
	std::cout << "data mean count=" << _data_mean.count() << std::endl;
	int offset = blob.offset(0);
	std::cout << "offset=" << offset << std::endl;
	std::cout << _data_mean.cpu_data() << std::endl;
	caffe::caffe_sub(_data_mean.count(),blob.mutable_cpu_data()+offset,
			 _data_mean.cpu_data(),blob.mutable_cpu_data()+offset);
			 }*/
    //std::vector<Blob<float>*> bottom = {blob};
    //std::cout << blob << std::endl;
    float loss = 0.0;
    std::vector<Datum> dv = {datum};
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddDatumVector(dv);
    //std::vector<Blob<float>*> results = _net->Forward(bottom,&loss);
    std::vector<Blob<float>*> results = _net->ForwardPrefilled(&loss);
    //std::cout << "accuracy=" << results[1]->cpu_data()[0] << std::endl;
    int bcat = -1;
    double bprob = -1.0;
    for (int i=0;i<1000;i++)
      {
	if (results[1]->cpu_data()[i] > bprob)
	  {
	    bcat = i;
	    bprob = results[1]->cpu_data()[i];
	  }
      }
    std::cout << "bcat=" << bcat << " -- accuracy=" << bprob << std::endl;
    
    /*int startl = 0;
    int endl = _net.layers().size()-1;
    double loss = _net.ForwardFromTo(startl,endl);
    std::cout << _net.net_output_blobs_.size() << std::endl;*/
    
    return 0;
  }

  template class CaffeLib<ImgInputFileConn,NoOutputConn,CaffeModel>;
}
