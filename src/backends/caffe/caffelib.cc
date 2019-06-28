/**
 * DeepDetect
 * Copyright (c) 2014-2018 Emmanuel Benazera
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
#include "generators/net_caffe.h"
#include "generators/net_caffe_convnet.h"
#include "generators/net_caffe_resnet.h"
#include "generators/net_caffe_recurrent.h"
#include "utils/fileops.hpp"
#include "utils/utils.hpp"
#include "utils/apitools.h"
#include "caffe/sgd_solvers.hpp"
#include <chrono>
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
    this->_libname = "caffe";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::CaffeLib(CaffeLib &&cl) noexcept
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,CaffeModel>(std::move(cl))
  {
    this->_libname = "caffe";
    _gpu = cl._gpu;
    _gpuid = cl._gpuid;
    _net = cl._net;
    _nclasses = cl._nclasses;
    _regression = cl._regression;
    _ntargets = cl._ntargets;
    //    _targets = cl._targets;
    _autoencoder = cl._autoencoder;
    cl._net = nullptr;
    _crop_size = cl._crop_size;
    _scale = cl._scale;
    _loss = cl._loss;
    _best_metrics = cl._best_metrics;
    _best_metric_value = cl._best_metric_value;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~CaffeLib()
  {
    delete _net;
    _net = nullptr;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::set_gpuid(const APIData &ad)
  {
#if !defined(CPU_ONLY) && !defined(USE_CAFFE_CPU_ONLY)
    if (ad.has("gpuid"))
      {
	try
	  {
	    int gpuid = ad.get("gpuid").get<int>();
	    if (gpuid == -1)
	      {
		int count_gpus = 0;
		cudaGetDeviceCount(&count_gpus);
		for (int i =0;i<count_gpus;i++)
		  _gpuid.push_back(i);
	      }
	    else _gpuid = {gpuid};
	    _gpu = true;
	  }
	catch(std::exception &e)
	  {
	    std::vector<int> gpuid = ad.get("gpuid").get<std::vector<int>>();
	    if (gpuid.size()== 1 && gpuid.at(0) == -1)
	      {
		int count_gpus = 0;
		cudaGetDeviceCount(&count_gpus);
		for (int i =0;i<count_gpus;i++)
		  _gpuid.push_back(i);
	      }
	    else _gpuid = gpuid;
	    _gpu = true;
	  }
	for (auto i: _gpuid)
	  this->_logger->info("Using GPU {}",i);
      }
#else
    (void)ad;
#endif
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::instantiate_template(const APIData &ad)
  {
    // - check whether there's a risk of erasing model files
    if (this->_mlmodel.read_from_repository(this->_mlmodel._repo,this->_logger))
      throw MLLibBadParamException("error reading or listing Caffe models in repository " + this->_mlmodel._repo);
    if (!this->_mlmodel._sstate.empty())
      {
	// we assuming training with resume from solverstate, error is deferred to the training
	// call assessing the resume is set to true
	this->_logger->warn("ignoring template argument since .solverstate file exists, assuming a training job will resume, and deferring errors to potential future training call");
	update_deploy_protofile_softmax(ad); // temperature scaling
	create_model(true); // phase changed to train as needed later on
	return;
      }
    if (!this->_mlmodel._weights.empty())
      {
	if (ad.has("finetuning") && ad.get("finetuning").get<bool>()
	    && !this->_mlmodel._trainf.empty()) // may want to finetune from a template only if no neural net definition present
	  throw MLLibBadParamException("using template for finetuning but model prototxt already exists, remove 'template' from 'mllib', or remove existing 'prototxt' files ?");
	/*else if (ad.has("resume") && ad.get("resume").get<bool>()) // resuming from state, may not want to override the existing network definition (e.g. after finetuning)
	  throw MLLibBadParamException("using template while resuming from existing model, remove 'template' from 'mllib' ?");*/
	else if (!this->_mlmodel._trainf.empty())
	  throw MLLibBadParamException("using template while model prototxt and network weights exist, remove 'template' from 'mllib' or remove prototxt files instead instead ?");
      }
      
    // - locate template repository
    std::string model_tmpl = ad.get("template").get<std::string>();
    this->_mlmodel._model_template = model_tmpl;
    this->_logger->info("instantiating model template {}",model_tmpl);
    
    // - copy files to model repository
    std::string source = this->_mlmodel._mlmodel_template_repo + '/' + model_tmpl + "/";
    this->_logger->info("source={}",source);
    this->_logger->info("dest={}",this->_mlmodel._repo + '/' + model_tmpl + ".prototxt");
    std::string dest_net = this->_mlmodel._repo + '/' + model_tmpl + ".prototxt";
    std::string dest_deploy_net = this->_mlmodel._repo + "/deploy.prototxt";
    if (model_tmpl != "mlp" && model_tmpl != "convnet" && model_tmpl != "resnet" && model_tmpl != "recurrent")
      {
	int err = fileops::copy_file(source + model_tmpl + ".prototxt", dest_net);
	if (err == 1)
	  throw MLLibBadParamException("failed to locate model template " + source + ".prototxt");
	else if (err == 2)
	  throw MLLibBadParamException("failed to create model template destination " + dest_net);
	err = fileops::copy_file(source + "deploy.prototxt", dest_deploy_net);
	if (err == 1)
	  throw MLLibBadParamException("failed to locate deploy template " + source + "deploy.prototxt");
	else if (err == 2)
	  throw MLLibBadParamException("failed to create destination deploy solver file " + dest_deploy_net);
      }
    int err = fileops::copy_file(source + model_tmpl + "_solver.prototxt",
				 this->_mlmodel._repo + '/' + model_tmpl + "_solver.prototxt");
    if (err == 1)
      throw MLLibBadParamException("failed to locate solver template " + source + model_tmpl + "_solver.prototxt");
    else if (err == 2)
      throw MLLibBadParamException("failed to create destination template solver file " + this->_mlmodel._repo + '/' + model_tmpl + "_solver.prototxt");
    
    // if mlp template, set the net structure as number of layers.
    if (model_tmpl == "mlp" || model_tmpl == "mlp_db" || model_tmpl == "lregression")
      {
	caffe::NetParameter net_param,deploy_net_param;
	configure_mlp_template(ad,this->_inputc,net_param,deploy_net_param);
	caffe::WriteProtoToTextFile(net_param,dest_net);
	caffe::WriteProtoToTextFile(deploy_net_param,dest_deploy_net);
      }
    else if (model_tmpl == "convnet")
      {
	caffe::NetParameter net_param,deploy_net_param;
	configure_convnet_template(ad,this->_inputc,net_param,deploy_net_param);
	if (typeid(this->_inputc) == typeid(ImgCaffeInputFileConn))
	  configure_image_augmentation(ad,net_param);
	caffe::WriteProtoToTextFile(net_param,dest_net);
	caffe::WriteProtoToTextFile(deploy_net_param,dest_deploy_net);
      }
    else if (model_tmpl == "resnet")
      {
	caffe::NetParameter net_param,deploy_net_param;
	configure_resnet_template(ad,this->_inputc,net_param,deploy_net_param);
	if (typeid(this->_inputc) == typeid(ImgCaffeInputFileConn))
	  configure_image_augmentation(ad,net_param);
	caffe::WriteProtoToTextFile(net_param,dest_net);
	caffe::WriteProtoToTextFile(deploy_net_param,dest_deploy_net);
      }
    else if (model_tmpl.find("ssd")!=std::string::npos
	     || model_tmpl.find("refinedet")!=std::string::npos)
      {
	bool refinedet = (model_tmpl.find("refinedet")!=std::string::npos);
	configure_ssd_template(dest_net,dest_deploy_net,ad,refinedet);
      }
    else if (model_tmpl.find("recurrent") != std::string::npos)
      {
        caffe::NetParameter net_param,deploy_net_param;
        configure_recurrent_template(ad,this->_inputc,net_param,deploy_net_param);
        caffe::WriteProtoToTextFile(net_param,dest_net);
        caffe::WriteProtoToTextFile(deploy_net_param,dest_deploy_net);
      }
    else
      {
	caffe::NetParameter net_param,deploy_net_param;
	caffe::ReadProtoFromTextFile(dest_net,&net_param); //TODO: catch parsing error (returns bool true on success)
	caffe::ReadProtoFromTextFile(dest_deploy_net,&deploy_net_param);

       if (this->_loss == "dice")
      // dice loss!!
      {
        if (net_param.name().compare("deeplab_vgg16")==0
	    || net_param.name().compare("pspnet_vgg16")==0
	    || net_param.name().compare("pspnet_50")==0
	    || net_param.name().compare("pspnet_101")==0)
          {
            update_protofiles_dice_deeplab_vgg16(net_param, deploy_net_param, ad);
	  }
        else if (net_param.name().compare("unet") == 0)
          {
            update_protofiles_dice_unet(net_param,deploy_net_param, ad);
	  }

      }

    // switch to imageDataLayer
    //TODO: should apply to all templates with images
    if (!this->_inputc._db && !this->_inputc._bbox && !this->_inputc._segmentation && !this->_inputc._ctc && typeid(this->_inputc) == typeid(ImgCaffeInputFileConn))
      {
        update_protofile_imageDataLayer(net_param);
      }
    
    // input should be ok, now do the output
    if (this->_inputc._multi_label)
      {
        int k = net_param.layer_size();
        for (int l=k-1;l>0;l--)
          {
            caffe::LayerParameter *lparam = net_param.mutable_layer(l);
            if (lparam->type() == "SoftmaxWithLoss")
              {
                lparam->set_type("MultiLabelSigmoidLoss");
                lparam->clear_include();
                caffe::NetStateRule *nsr = lparam->add_include();
                nsr->set_phase(caffe::TRAIN);
                break;
              }
	  }
	// XXX: code below removes the softmax layer
	// protobuf only allows to remove last element from repeated field.
	int softm_pos = -1;
	for (int l=k-1;l>0;l--)
	  {
            caffe::LayerParameter *lparam = net_param.mutable_layer(l);
            if (lparam->type() == "Softmax")
              {
                softm_pos = l;
                break;
              }
	  }
	if (softm_pos > 0)
	  {
            if (!_regression)
              {
                for (int l=softm_pos;l<net_param.layer_size()-1;l++)
                  {
                    caffe::LayerParameter *lparam = net_param.mutable_layer(l);
                    *lparam = net_param.layer(l+1);
                  }
                net_param.mutable_layer()->RemoveLast();
              }
            else
              {
                caffe::LayerParameter *lparam = net_param.mutable_layer(softm_pos);
                lparam->set_type("Sigmoid");
                lparam->set_name("pred");
                *lparam->mutable_top(0) = "pred";
		
                //lparam->add_sigmoid_param();
                //lparam->sigmoid_param().set_engine(lparam->softmax_param().engine());
                // for doing so in a clean way, need to match softmaxParameter::engine
                // with sigmoidparameter::engine
                // for now, rewrite engine filed as is
                lparam->clear_softmax_param();
              }
	    
	  }
	else throw MLLibInternalException("Couldn't find Softmax layer to replace for multi-label training");
      
	k = deploy_net_param.layer_size();
	caffe::LayerParameter *lparam = deploy_net_param.mutable_layer(k-1);
	if (lparam->type() == "Softmax")
	  {
            if (!_regression)
              deploy_net_param.mutable_layer()->RemoveLast();
            else
              {
                lparam->set_type("Sigmoid");
                lparam->set_name("pred");
                *lparam->mutable_top(0) = "pred";
		//lparam->add_sigmoid_param();
                //lparam->sigmoid_param().set_engine(lparam->softmax_param().engine());
                // see 20 lines above for comment
                lparam->clear_softmax_param();
              }
	  }
      } // end multi_label
    else if (_regression)
      {
	int k = net_param.layer_size();
        for (int l=k-1;l>0;l--)
          {
            caffe::LayerParameter *lparam = net_param.mutable_layer(l);
            if (lparam->type() == "SoftmaxWithLoss")
              {
                lparam->set_type("SigmoidCrossEntropyLoss");
		lparam->clear_include();
                caffe::NetStateRule *nsr = lparam->add_include();
                nsr->set_phase(caffe::TRAIN);
                break;
              }
	    else if (lparam->type() == "Softmax")
	      {
		lparam->set_type("Sigmoid");
		lparam->clear_include();
		caffe::NetStateRule *nsr = lparam->add_include();
                nsr->set_phase(caffe::TEST);
	      }
	  }
	
      }
          
	// input size
	caffe::LayerParameter *lparam = net_param.mutable_layer(1); // test
	caffe::LayerParameter *dlparam = deploy_net_param.mutable_layer(0);
	if (!this->_inputc._bbox && !this->_inputc._ctc &&(_crop_size > 0 || (this->_inputc.width() != -1 && this->_inputc.height() != -1))) // forced width & height
	  {
	    int width = this->_inputc.width();
	    int height = this->_inputc.height();
	    if (_crop_size > 0)
	      width = height = _crop_size;
	    lparam->mutable_memory_data_param()->set_channels(this->_inputc.channels());
	    lparam->mutable_memory_data_param()->set_height(height);
	    lparam->mutable_memory_data_param()->set_width(width);
	    dlparam->mutable_memory_data_param()->set_channels(this->_inputc.channels());
	    dlparam->mutable_memory_data_param()->set_height(height);
	    dlparam->mutable_memory_data_param()->set_width(width);
	  }
		
	// image data augmentation
	if (typeid(this->_inputc) == typeid(ImgCaffeInputFileConn))
	  configure_image_augmentation(ad,net_param);
	    
	// adapt number of neuron output
	update_protofile_classes(net_param);
	update_protofile_classes(deploy_net_param);
	caffe::WriteProtoToTextFile(net_param,dest_net);
	caffe::WriteProtoToTextFile(deploy_net_param,dest_deploy_net);
  }
    if (ad.has("finetuning") && ad.get("finetuning").get<bool>())
      {
	if (this->_mlmodel._weights.empty()) // weights should have been specified or detected on the first pass into the model repository
	  throw MLLibBadParamException("finetuning requires specifying an existing weights file");	
	caffe::NetParameter net_param,deploy_net_param;
	caffe::ReadProtoFromTextFile(dest_net,&net_param); //TODO: catch parsing error (returns bool true on success)
	caffe::ReadProtoFromTextFile(dest_deploy_net,&deploy_net_param);
	update_protofile_finetune(net_param);
	update_protofile_finetune(deploy_net_param);
	caffe::WriteProtoToTextFile(net_param,dest_net);
	caffe::WriteProtoToTextFile(deploy_net_param,dest_deploy_net);
      }
    
    if (this->_mlmodel.read_from_repository(this->_mlmodel._repo,this->_logger))
      throw MLLibBadParamException("error reading or listing Caffe models in repository " + this->_mlmodel._repo);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  caffe::LayerParameter*
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::insert_layer_before(caffe::NetParameter &net_param, int layer_number)
  {
    
    net_param.add_layer();

    for (int i=net_param.layer_size()-1; i>layer_number; --i)
      {
        caffe::LayerParameter *lparam = net_param.mutable_layer(i);
        *lparam = net_param.layer(i-1);
      }
    net_param.mutable_layer(layer_number)->Clear();
    return  net_param.mutable_layer(layer_number);
  }


  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  caffe::LayerParameter*
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::find_layer_by_name(caffe::NetParameter &net_param, std::string name)
  {
    int k = net_param.layer_size();
    for (int l=k-1;l>0;l--)
      {
        caffe::LayerParameter *lparam = net_param.mutable_layer(l);
        if (lparam->name().compare(name) == 0)
          return net_param.mutable_layer(l);
      }
    return NULL;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  caffe::LayerParameter*
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::find_layer_by_type(caffe::NetParameter &net_param, std::string type)
  {
    int k = net_param.layer_size();
    for (int l=k-1; l>=0; l--)
      {
        caffe::LayerParameter *lparam = net_param.mutable_layer(l);
        if (lparam->type() == type)
          return net_param.mutable_layer(l);
      }
    return NULL;
  }


  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::find_index_layer_by_type(caffe::NetParameter &net_param, std::string type)
  {
    int k = net_param.layer_size();
    for (int l=k-1;l>0;l--)
      {
        caffe::LayerParameter *lparam = net_param.mutable_layer(l);
        if (lparam->type() == type)
          return l;
      }
    return -1;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::find_index_layer_by_name(caffe::NetParameter &net_param, std::string name)
  {
    int k = net_param.layer_size();
    for (int l=k-1;l>0;l--)
      {
        caffe::LayerParameter *lparam = net_param.mutable_layer(l);
        if (lparam->name() == name)
          return l;
      }
    return -1;
  }


  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::configure_image_augmentation(const APIData &ad,
													 caffe::NetParameter &net_param)
  {
    if ((ad.has("rotate") && ad.get("rotate").get<bool>()) 
	|| (ad.has("mirror") && ad.get("mirror").get<bool>())
	|| ad.has("crop_size") || _scale != 1.0)
      {
	caffe::LayerParameter *lparam = net_param.mutable_layer(0); // data input layer
	if (lparam->type() != "DenseImageData")
	  {
	    if (ad.has("mirror"))
	      lparam->mutable_transform_param()->set_mirror(ad.get("mirror").get<bool>());
	    if (ad.has("rotate"))
	      lparam->mutable_transform_param()->set_rotate(ad.get("rotate").get<bool>());
	    if (ad.has("crop_size"))
	      {
		_crop_size = ad.get("crop_size").get<int>();
		lparam->mutable_transform_param()->set_crop_size(_crop_size);
		caffe::LayerParameter *dlparam = net_param.mutable_layer(1); // test input layer
		dlparam->mutable_transform_param()->set_crop_size(_crop_size);
	      }
	    else lparam->mutable_transform_param()->clear_crop_size();
	    if (_scale != 1.0)
	      {
		lparam->mutable_transform_param()->set_scale(_scale);
		caffe::LayerParameter *dlparam = net_param.mutable_layer(1); // test input layer
		dlparam->mutable_transform_param()->set_scale(_scale);
	      }
	  }
	else
	  {
	    if (ad.has("mirror"))
	      lparam->mutable_dense_image_data_param()->set_mirror(ad.get("mirror").get<bool>());
	    if (ad.has("rotate"))
	      lparam->mutable_dense_image_data_param()->set_rotate(ad.get("rotate").get<bool>());
	    lparam->mutable_dense_image_data_param()->set_new_height(this->_inputc.height());
	    lparam->mutable_dense_image_data_param()->set_new_width(this->_inputc.width());
	    // XXX: DenseImageData supports crop_height and crop_width
	  }
      }
    if (ad.has("noise"))
      {
	std::vector<std::string> noise_options = {
	  "decolorize","hist_eq","inverse","gauss_blur","posterize","erode",
	  "saltpepper","clahe","convert_to_hsv","convert_to_lab"
	};
	APIData ad_noise = ad.getobj("noise");
	caffe::LayerParameter *lparam = net_param.mutable_layer(0); // data input layer
	caffe::TransformationParameter *trparam = lparam->mutable_transform_param();
	caffe::NoiseParameter *nparam = trparam->mutable_noise_param();
	if (ad_noise.has("all_effects") && ad_noise.get("all_effects").get<bool>())
	  nparam->set_all_effects(true);
	else
	  {
	    for (auto s: noise_options)
	      {
		if (ad_noise.has(s))
		  {
		    if (s == "decolorize")
		      nparam->set_decolorize(ad_noise.get(s).get<bool>());
		    else if (s == "hist_eq")
		      nparam->set_hist_eq(ad_noise.get(s).get<bool>());
		    else if (s == "inverse")
		      nparam->set_inverse(ad_noise.get(s).get<bool>());
		    else if (s == "gauss_blur")
		      nparam->set_gauss_blur(ad_noise.get(s).get<bool>());
		    else if (s == "posterize")
		      nparam->set_hist_eq(ad_noise.get(s).get<bool>());
		    else if (s == "erode")
		      nparam->set_erode(ad_noise.get(s).get<bool>());
		    else if (s == "saltpepper")
			  nparam->set_saltpepper(ad_noise.get(s).get<bool>());
		    else if (s == "clahe")
		      nparam->set_clahe(ad_noise.get(s).get<bool>());
		    else if (s == "convert_to_hsv")
		      nparam->set_convert_to_hsv(ad_noise.get(s).get<bool>());
		    else if (s == "convert_to_lab")
		      nparam->set_convert_to_lab(ad_noise.get(s).get<bool>());
		  }
	      }
	  }
	if (ad_noise.has("prob"))
	  nparam->set_prob(ad_noise.get("prob").get<double>());
      }
    if (ad.has("distort"))
      {
	std::vector<std::string> distort_options = {
	  "brightness","contrast","saturation","hue","random_order"
	};
	APIData ad_distort = ad.getobj("distort");
	caffe::LayerParameter *lparam = net_param.mutable_layer(0); // data input layer
	caffe::TransformationParameter *trparam = lparam->mutable_transform_param();
	caffe::DistortionParameter *nparam = trparam->mutable_distort_param();
	if (ad_distort.has("all_effects") && ad_distort.get("all_effects").get<bool>())
	  nparam->set_all_effects(true);
	else
	  {
	    for (auto s: distort_options)
	      {
		if (ad_distort.has(s))
		  {
		    if (s == "brightness")
		      nparam->set_brightness(ad_distort.get(s).get<bool>());
		    else if (s == "contrast")
		      nparam->set_contrast(ad_distort.get(s).get<bool>());
		    else if (s == "saturation")
		      nparam->set_saturation(ad_distort.get(s).get<bool>());
		    else if (s == "hue")
		      nparam->set_hue(ad_distort.get(s).get<bool>());
		    else if (s == "random_order")
		      nparam->set_random_order(ad_distort.get(s).get<bool>());
		  }
	      }
	  }
	if (ad_distort.has("prob"))
	  nparam->set_prob(ad_distort.get("prob").get<double>());
      }
    if(ad.has("geometry"))
      {
	std::vector<std::string> geometry_options = {
	  "persp_horizontal", "persp_vertical", "zoom_out", "zoom_in"
	};
	APIData ad_geometry = ad.getobj("geometry");
	caffe::LayerParameter *lparam = net_param.mutable_layer(0); // data input layer
	caffe::TransformationParameter *trparam = lparam->mutable_transform_param();
	caffe::GeometryParameter *gparam = trparam->mutable_geometry_param();
	if (ad_geometry.has("all_effects") && ad_geometry.get("all_effects").get<bool>())
	  gparam->set_all_effects(true);
	else
	  {
	    for (auto s: geometry_options)
	      {
		if (ad_geometry.has(s))
		  {
		    if (s == "persp_horizontal")
		      gparam->set_persp_horizontal(ad_geometry.get(s).get<bool>());
		    else if (s == "persp_vertical")
		      gparam->set_persp_vertical(ad_geometry.get(s).get<bool>());
		    else if (s == "zoom_out")
		      gparam->set_zoom_out(ad_geometry.get(s).get<bool>());
		    else if (s == "zoom_in")
		      gparam->set_zoom_in(ad_geometry.get(s).get<bool>());
		  }
	      }
	    if (ad_geometry.has("persp_factor"))
	      gparam->set_persp_factor(ad_geometry.get("persp_factor").get<double>());
	    if (ad_geometry.has("zoom_factor"))
	      gparam->set_zoom_factor(ad_geometry.get("zoom_factor").get<double>());
	    if (ad_geometry.has("pad_mode"))
	      {
		std::string pmode = ad_geometry.get("pad_mode").get<std::string>();
		if (pmode == "constant")
		  gparam->set_pad_mode(::caffe::GeometryParameter_Pad_mode_CONSTANT);
		else if (pmode == "mirrored")
		  gparam->set_pad_mode(::caffe::GeometryParameter_Pad_mode_MIRRORED);
		else if (pmode == "repeat_nearest")
		  gparam->set_pad_mode(::caffe::GeometryParameter_Pad_mode_REPEAT_NEAREST);
	      }
	  }
	if (ad_geometry.has("prob"))
	  gparam->set_prob(ad_geometry.get("prob").get<double>());
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::configure_mlp_template(const APIData &ad,
												   const TInputConnectorStrategy &inputc,
												   caffe::NetParameter &net_param,
												   caffe::NetParameter &dnet_param)
    {
      NetCaffe<NetInputCaffe<TInputConnectorStrategy>,NetLayersCaffeMLP,NetLossCaffe> netcaffe(&net_param,&dnet_param,this->_logger);
	netcaffe._nic.configure_inputs(ad,inputc);
	if (inputc._sparse)
	  const_cast<APIData&>(ad).add("sparse",true);
	if (_regression)
	  const_cast<APIData&>(ad).add("regression",true);
	if (_autoencoder)
	  {
	    const_cast<APIData&>(ad).add("autoencoder",true);
	    const_cast<APIData&>(ad).add("ntargets",inputc.width()*inputc.height());
	  }
	netcaffe._nlac.configure_net(ad);
    }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::configure_convnet_template(const APIData &ad,
												       const TInputConnectorStrategy &inputc,
												       caffe::NetParameter &net_param,
												       caffe::NetParameter &dnet_param)
  {
    NetCaffe<NetInputCaffe<TInputConnectorStrategy>,NetLayersCaffeConvnet,NetLossCaffe> netcaffe(&net_param,&dnet_param,this->_logger);
    netcaffe._nic.configure_inputs(ad,inputc);
    if (inputc._flat1dconv)
      const_cast<APIData&>(ad).add("flat1dconv",static_cast<bool>(inputc._flat1dconv));
    if (_regression)
      const_cast<APIData&>(ad).add("regression",true);
    if (_autoencoder)
      {
	const_cast<APIData&>(ad).add("autoencoder",true);
	const_cast<APIData&>(ad).add("ntargets",inputc.channels());
	const_cast<APIData&>(ad).add("width",inputc.width());
	const_cast<APIData&>(ad).add("height",inputc.height());
      }
    netcaffe._nlac.configure_net(ad);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::configure_resnet_template(const APIData &ad,
												      const TInputConnectorStrategy &inputc,
												      caffe::NetParameter &net_param,
												      caffe::NetParameter &dnet_param)
  {
    NetCaffe<NetInputCaffe<TInputConnectorStrategy>,NetLayersCaffeResnet,NetLossCaffe> netcaffe(&net_param,&dnet_param,this->_logger);
    netcaffe._nic.configure_inputs(ad,inputc);
    if (inputc._sparse)
      const_cast<APIData&>(ad).add("sparse",true);
    if (inputc._flat1dconv)
      const_cast<APIData&>(ad).add("flat1dconv",static_cast<bool>(inputc._flat1dconv));
    if (_regression)
      const_cast<APIData&>(ad).add("regression",true);
    netcaffe._nlac.configure_net(ad);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::configure_ssd_template(const std::string &dest_net,
												   const std::string &dest_deploy_net,
												   const APIData &ad,
												   const bool &refinedet)
  {
    //- load prototxt
    caffe::NetParameter net_param,deploy_net_param;
    caffe::ReadProtoFromTextFile(dest_net,&net_param);
    caffe::ReadProtoFromTextFile(dest_deploy_net,&deploy_net_param);

    //- if finetuning, change the proper layer names
    std::string postfix = "_ftune";
    const bool finetune = (ad.has("finetuning") && ad.get("finetuning").get<bool>());
    int k = net_param.layer_size();
    for (int l=0;l<k;l++)
      {
	caffe::LayerParameter *lparam = net_param.mutable_layer(l);
	if (finetune)
	  {
	    if (lparam->name().find("mbox_conf") != std::string::npos
		|| lparam->name().find("mbox_loc") != std::string::npos)
	      {
		lparam->set_name(lparam->name() + postfix);
	      }
	    for (int t=0;t<lparam->top_size();t++)
	      {
		if (lparam->top(t).find("_conf") != std::string::npos
		    || lparam->top(t).find("_loc") != std::string::npos)
		  {
		    lparam->set_top(t,lparam->top(t) + postfix);
		  }
	      }
	    for (int t=0;t<lparam->bottom_size();t++)
	      {
		if (lparam->bottom(t).find("_conf") != std::string::npos
		    || lparam->bottom(t).find("_loc") != std::string::npos)
		  {
		    lparam->set_bottom(t,lparam->bottom(t) + postfix);
		  }
	      }
	  }
	//- set correct layer parameters based on nclasses
	if (lparam->name() == "mbox_loss" || lparam->name() == "odm_loss")
	  {
	      lparam->mutable_multibox_loss_param()->set_num_classes(_nclasses);
	  }
	else if (lparam->name() == "detection_out")
	  {
	    lparam->mutable_detection_output_param()->set_num_classes(_nclasses);
	  }
	else if (lparam->name() == "detection_eval")
	  {
	    lparam->mutable_detection_evaluate_param()->set_num_classes(_nclasses);
	  }
	else if (lparam->name().find("mbox_conf_reshape") != std::string::npos
		 || lparam->name().find("odm_conf_reshape") != std::string::npos)
	  {
	    lparam->mutable_reshape_param()->mutable_shape()->set_dim(2,_nclasses);
	  }
      }
	
    // fix other layer parameters
    for (int l=0;l<k;l++)
      {
	caffe::LayerParameter *lparam = net_param.mutable_layer(l);
	if (!refinedet && lparam->name().find("mbox_conf") != std::string::npos
	    && lparam->type() == "Convolution")
	  {
	    int num_priors_per_location = lparam->mutable_convolution_param()->num_output() / 2;
	    lparam->mutable_convolution_param()->set_num_output(num_priors_per_location * _nclasses);
	  }
	else if (refinedet && lparam->name().find("mbox_conf") != std::string::npos
		 && lparam->name()[0] == 'P' && lparam->type() == "Convolution")
	  {
	    int num_priors_per_location = lparam->mutable_convolution_param()->num_output() / 2;
	    lparam->mutable_convolution_param()->set_num_output(num_priors_per_location * _nclasses);
	  }
      }

    k = deploy_net_param.layer_size();
    for (int l=0;l<k;l++)
      {
	caffe::LayerParameter *lparam = deploy_net_param.mutable_layer(l);
	if (finetune)
	  {
	    if (lparam->name().find("mbox_conf") != std::string::npos
		|| lparam->name().find("mbox_loc") != std::string::npos)
	      {
		lparam->set_name(lparam->name() + postfix);
	      }
	    for (int t=0;t<lparam->top_size();t++)
	      {
		if (lparam->top(t).find("_conf") != std::string::npos
		    || lparam->top(t).find("_loc") != std::string::npos)
		  {
		    lparam->set_top(t,lparam->top(t) + postfix);
		  }
	      }
	    for (int t=0;t<lparam->bottom_size();t++)
	      {
		if (lparam->bottom(t).find("_conf") != std::string::npos
		    || lparam->bottom(t).find("_loc") != std::string::npos)
		  {
		    lparam->set_bottom(t,lparam->bottom(t) + postfix);
		  }
	      }
	  }
	//- set correct layer parameters based on nclasses
	if (!refinedet && lparam->name().find("mbox_conf") != std::string::npos
	    && lparam->type() == "Convolution")
	  {
	    int num_priors_per_location = lparam->mutable_convolution_param()->num_output() / 2;
	    lparam->mutable_convolution_param()->set_num_output(num_priors_per_location * _nclasses);
	  }
	else if (refinedet && lparam->name().find("mbox_conf") != std::string::npos
		 && lparam->name()[0] == 'P' && lparam->type() == "Convolution")
	  {
	    int num_priors_per_location = lparam->mutable_convolution_param()->num_output() / 2;
	    lparam->mutable_convolution_param()->set_num_output(num_priors_per_location * _nclasses);
	  }
	else if (lparam->name() == "detection_out")
	  {
	    lparam->mutable_detection_output_param()->set_num_classes(_nclasses);
	  }
	else if (lparam->name().find("mbox_conf_reshape") != std::string::npos
		 || lparam->name().find("odm_conf_reshape") != std::string::npos)
	  {
	    lparam->mutable_reshape_param()->mutable_shape()->set_dim(2,_nclasses);
	  }
      }
    
    //- write it down
    caffe::WriteProtoToTextFile(net_param,dest_net);
    caffe::WriteProtoToTextFile(deploy_net_param,dest_deploy_net);
  }


  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::configure_recurrent_template(const APIData &ad,
                                                                                                   const TInputConnectorStrategy &inputc,
                                                                                                   caffe::NetParameter &net_param,
                                                                                                         caffe::NetParameter &dnet_param)
  {

    NetCaffe<NetInputCaffe<TInputConnectorStrategy>,NetLayersCaffeRecurrent,NetLossCaffe> netcaffe(&net_param,&dnet_param,this->_logger);
    netcaffe._nic.configure_inputs(ad,inputc);
    // add ntargets to ad.
    const_cast<APIData&>(ad).add("ntargets",this->_ntargets);
    netcaffe._nlac.configure_net(ad);
    // will be changed at predict time
    // small default for debug
    dnet_param.mutable_layer(0)->mutable_memory_data_param()->set_channels(10);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::create_model(const bool &test)
  {
    // create net and fill it up
    if (!this->_mlmodel._def.empty() && !this->_mlmodel._weights.empty())
      {
	delete _net;
	_net = nullptr;	
	try
	  {
	    if (!test)
	      _net = new Net<float>(this->_mlmodel._def,caffe::TRAIN);
	    else
	      _net = new Net<float>(this->_mlmodel._def,caffe::TEST);
	  }
	catch (std::exception &e)
	  {
	    this->_logger->error("Error creating network");
	    throw;
	  }
	this->_logger->info("Using pre-trained weights from {}",this->_mlmodel._weights);
	try
	  {
	    _net->CopyTrainedLayersFrom(this->_mlmodel._weights);
	  }
	catch (std::exception &e)
	  {
	    this->_logger->error("Error copying pre-trained weights");
	    delete _net;
	    _net = nullptr;
	    throw;
	  }
	try
	  {
             model_complexity(this->_model_flops,this->_model_params);
	  }
	catch(std::exception &e)
	  {
	    // nets can be exotic, let's make sure we don't get killed here
	    this->_logger->error("failed computing net's complexity");
	  }
	try
	  {
	    model_type(_net,this->_mltype);
	  }
	catch (std::exception &e)
	  {
	    this->_logger->error("failed determining mltype");
	  }
	return 0;
      }
    // net definition is missing
    else if (this->_mlmodel._def.empty())
      return 2; // missing 'deploy' file.
    return 1;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::init_mllib(const APIData &ad)
  {
    if (ad.has("gpu"))
      _gpu = ad.get("gpu").get<bool>();
    set_gpuid(ad);
#if !defined(CPU_ONLY) && !defined(USE_CAFFE_CPU_ONLY)
    if (_gpu)
      {
	for (auto i: _gpuid)
	  {
	    Caffe::SetDevice(i);
	    Caffe::DeviceQuery();
	  }
	Caffe::set_mode(Caffe::GPU);
      }
    else Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::CPU);
#endif
    const std::string scale_key = "scale";
    if (ad.has(scale_key))
      apitools::get_float(ad, scale_key, _scale);
    if (ad.has("db"))
      this->_inputc._db = ad.get("db").get<bool>(); // XXX: API backward compatibility, if db is in mllib, assume it applies to input as well
    if (ad.has("nclasses"))
      _nclasses = ad.get("nclasses").get<int>();
    if (ad.has("regression") && ad.get("regression").get<bool>())
      {
	_regression = true;
    	_nclasses = 1;
	this->_mltype = "regression";
      }
    if (ad.has("loss"))
      {
        _loss = ad.get("loss").get<std::string>();
        if (this->_loss == "dice")
          if (! this->_inputc._segmentation)
            throw MLLibBadParamException("asked for  dice loss without segmentation");
      }

    if (ad.has("ntargets"))
      _ntargets = ad.get("ntargets").get<int>();
    if (this->_inputc._ntargets != -1)
      _ntargets = this->_inputc._ntargets;
    if (ad.has("autoencoder") && ad.get("autoencoder").get<bool>())
      _autoencoder = true;
    if (!_autoencoder && _nclasses == 0)
      throw MLLibBadParamException("number of classes is unknown (nclasses == 0)");
    bool multi =
      this->_inputc._multi_label &&  !(this->_inputc._db)
      && typeid(this->_inputc) == typeid(ImgCaffeInputFileConn);
    if (_regression && _ntargets == 0 && !multi)
      throw MLLibBadParamException("number of regression targets is unknown (ntargets == 0)");
    if (_regression && multi) // multisoft case
      if (ad.has("nclasses"))
        _nclasses = ad.get("nclasses").get<int>();
    // instantiate model template here, if any
    if (ad.has("template") && ad.get("template").get<std::string>() != "")
      instantiate_template(ad);
    else // model template instantiation is defered until training call
      {

	update_deploy_protofile_softmax(ad); // temperature scaling
	create_model(true);
      }

    // the first present measure will be used to snapshot best model
    _best_metrics = {"map", "meaniou", "mlacc", "delta_score_0.1", "bacc", "f1", "net_meas", "acc", "L1_mean_error", "eucll"};
    _best_metric_value = std::numeric_limits<double>::infinity();

    // import model from existing directory upon request
    if (ad.has("from_repository"))
      this->_mlmodel.copy_to_target(ad.get("from_repository").get<std::string>(),
				    this->_mlmodel._repo,
				    this->_logger);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::clear_mllib(const APIData &ad)
  {
    (void)ad;
    std::vector<std::string> extensions = {".solverstate",".caffemodel",".json"};
    if (!this->_inputc._db)
      extensions.push_back(".dat"); // e.g., for txt input connector and db, do not delete the vocab.dat since the db is not deleted
    fileops::remove_directory_files(this->_mlmodel._repo,extensions);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::train(const APIData &ad,
										 APIData &out)
  {
    if (this->_mlmodel._solver.empty())
      {
	throw MLLibBadParamException("missing solver file in " + this->_mlmodel._repo);
      }


    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
    int timesteps = 0;
    if (ad_mllib.has("timesteps"))
      timesteps = ad_mllib.get("timesteps").get<int>();
    if (timesteps <= 0 && this->_inputc._ctc)
      throw MLLibBadParamException("Need to specify timesteps > 0 with sequence (e.g. OCR / CTC) models");
    if  (timesteps <= 0 && typeid(this->_inputc) == typeid(CSVTSCaffeInputFileConn))
      throw MLLibBadParamException("Need to specify timesteps > 0 with recurrent  models");

    std::lock_guard<std::mutex> lock(_net_mutex); // XXX: not mandatory as train calls are locking resources from above
    TInputConnectorStrategy inputc(this->_inputc);
    this->_inputc._dv.clear();
    this->_inputc._dv_test.clear();
    this->_inputc._dv_sparse.clear();
    this->_inputc._dv_test_sparse.clear();
    this->_inputc._ids.clear();
    inputc._train = true;
    APIData cad = ad;
    cad.add("has_mean_file",this->_mlmodel._has_mean_file);
    if (_crop_size > 0)
      cad.add("crop_size",_crop_size);
    if (_autoencoder)
      cad.add("autoencoder",_autoencoder);
    if (typeid(this->_inputc) == typeid(CSVTSCaffeInputFileConn))
        inputc._timesteps = timesteps;
    try
      {
	inputc.transform(cad);
      }
    catch (...)
      {
	throw;
      }
    
    // instantiate model template here, as a defered from service initialization
    // since inputs are necessary in order to fit the inner net input dimension.
    if (!this->_mlmodel._model_template.empty())
      {
	// modifies model structure, template must have been copied at service creation with instantiate_template
	bool has_class_weights = ad_mllib.has("class_weights");
	int ignore_label = -1;
	if (ad_mllib.has("ignore_label"))
	  ignore_label = ad_mllib.get("ignore_label").get<int>();
	update_protofile_net(this->_mlmodel._repo + '/' + this->_mlmodel._model_template + ".prototxt",
                            this->_mlmodel._repo + "/deploy.prototxt",
                            inputc, has_class_weights, ignore_label, timesteps);
      }


    create_model(); // creates initial net.

    caffe::SolverParameter solver_param;
    if (!caffe::ReadProtoFromTextFile(this->_mlmodel._solver,&solver_param))
      throw MLLibInternalException("failed opening solver file " + this->_mlmodel._solver);
    bool has_mean_file = false;
    int user_batch_size, batch_size, test_batch_size, test_iter;
    update_in_memory_net_and_solver(solver_param,cad,inputc,has_mean_file,user_batch_size,batch_size,test_batch_size,test_iter);
    //caffe::ReadProtoFromTextFile(this->_mlmodel._solver,&solver_param);

    // parameters
#if !defined(CPU_ONLY) && !defined(USE_CAFFE_CPU_ONLY)
    bool gpu = _gpu;
    if (ad_mllib.has("gpu"))
      {
	gpu = ad_mllib.get("gpu").get<bool>();
	if (gpu)
	  set_gpuid(ad_mllib);
      }
    if (gpu)
      {
	solver_param.set_device_id(_gpuid.at(0));
	Caffe::SetDevice(_gpuid.at(0));
	Caffe::set_mode(Caffe::GPU);
	Caffe::set_solver_count(_gpuid.size());
      }
    else Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::CPU);
#endif
    
    // solver's parameters
    APIData ad_solver = ad_mllib.getobj("solver");
    if (ad_solver.size())
      {
        int max_iter = -1;
        if (ad_solver.has("iterations"))
	  {
	    max_iter = ad_solver.get("iterations").get<int>();
	    solver_param.set_max_iter(max_iter);
	  }
	if (ad_solver.has("snapshot")) // iterations between snapshots
	  {
	    int snapshot = ad_solver.get("snapshot").get<int>();
	    solver_param.set_snapshot(snapshot);
	  }
	if (ad_solver.has("snapshot_prefix")) // overrides default internal dd prefix which is model repo
	  {
	    std::string snapshot_prefix = ad_solver.get("snapshot_prefix").get<std::string>();
	    solver_param.set_snapshot_prefix(snapshot_prefix);
	  }
	if (ad_solver.has("solver_type"))
	  {
	    std::string solver_type = ad_solver.get("solver_type").get<std::string>();
	    if (strcasecmp(solver_type.c_str(),"SGD") == 0)
	      solver_param.set_solver_type(caffe::SolverParameter_SolverType_SGD);
	    else if (strcasecmp(solver_type.c_str(),"ADAGRAD") == 0)
	      {
		solver_param.set_solver_type(caffe::SolverParameter_SolverType_ADAGRAD);
		solver_param.set_momentum(0.0); // cannot be used with adagrad
	      }
	    else if (strcasecmp(solver_type.c_str(),"NESTEROV") == 0)
	      solver_param.set_solver_type(caffe::SolverParameter_SolverType_NESTEROV);
	    else if (strcasecmp(solver_type.c_str(),"RMSPROP") == 0)
	      {
		solver_param.set_solver_type(caffe::SolverParameter_SolverType_RMSPROP);
		solver_param.set_momentum(0.0); // not supported by Caffe PR
	      }
	    else if (strcasecmp(solver_type.c_str(),"ADADELTA") == 0)
	      solver_param.set_solver_type(caffe::SolverParameter_SolverType_ADADELTA);
	    else if (strcasecmp(solver_type.c_str(),"ADAM") == 0)
	      solver_param.set_solver_type(caffe::SolverParameter_SolverType_ADAM);
	    else if (strcasecmp(solver_type.c_str(),"AMSGRAD") == 0)
	      {
		solver_param.set_solver_type(caffe::SolverParameter_SolverType_ADAM);
        solver_param.set_amsgrad(true);
	      }
	  }
	if (ad_solver.has("test_interval"))
	  solver_param.set_test_interval(ad_solver.get("test_interval").get<int>());
	if (ad_solver.has("test_initialization"))
	  solver_param.set_test_initialization(ad_solver.get("test_initialization").get<bool>());
	if (ad_solver.has("lr_policy"))
	  solver_param.set_lr_policy(ad_solver.get("lr_policy").get<std::string>());
	if (ad_solver.has("base_lr"))
	  solver_param.set_base_lr(ad_solver.get("base_lr").get<double>());
	if (ad_solver.has("gamma"))
	  solver_param.set_gamma(ad_solver.get("gamma").get<double>());
	if (ad_solver.has("stepsize"))
	  solver_param.set_stepsize(ad_solver.get("stepsize").get<int>());
	if (ad_solver.has("momentum") && solver_param.solver_type() != caffe::SolverParameter_SolverType_ADAGRAD)
	  solver_param.set_momentum(ad_solver.get("momentum").get<double>());
	if (ad_solver.has("weight_decay"))
	  solver_param.set_weight_decay(ad_solver.get("weight_decay").get<double>());
	if (ad_solver.has("power"))
	  solver_param.set_power(ad_solver.get("power").get<double>());
	if (ad_solver.has("rms_decay"))
	  solver_param.set_rms_decay(ad_solver.get("rms_decay").get<double>());
	if (ad_solver.has("iter_size"))
	  solver_param.set_iter_size(ad_solver.get("iter_size").get<int>());

	if (ad_solver.has("min_lr"))
	  solver_param.set_min_lr(ad_solver.get("min_lr").get<double>());
	if (ad_solver.has("lr_mult"))
	  solver_param.set_lr_mult(ad_solver.get("lr_mult").get<double>());
	double p_mult = -1;
	if (ad_solver.has("p_mult")) {
	  p_mult = ad_solver.get("p_mult").get<double>();
	  solver_param.set_p_mult(p_mult);
	}
	if (ad_solver.has("period"))
	  solver_param.set_period(ad_solver.get("period").get<int>());
	else if (ad_solver.has("ncycles")) {
	  // compute initial period length in order to have num_period cycle until max_iter
	  int ncycles = ad_solver.get("ncycles").get<int>();
	  if (p_mult < 0) {
	    p_mult = 2.0;
	    solver_param.set_p_mult(p_mult);
	  }
	  if (max_iter > 0) {
	    int period = ((float) max_iter)  / ( pow(p_mult, ncycles) - 1.0);
	    solver_param.set_period(period);
	  } else
	    throw MLLibBadParamException("sgdr + ncycles  requires iterations to be set");
	}
      }
    
    // optimize
    this->_tjob_running = true;
    boost::shared_ptr<caffe::Solver<float>> solver;
    try
      {
	solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));
      }
    catch(...)
      {
	throw MLLibInternalException("solver creation exception");
      }
    try
      {
	model_type(solver->net().get(),this->_mltype);
      }
    catch (std::exception &e)
      {
	this->_logger->error("failed determining mltype");
      }
    if (!inputc._dv.empty() || !inputc._dv_sparse.empty())
      {
	this->_logger->info("filling up net prior to training");
	try {
	  if (!inputc._sparse)
	    {
	      if (boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(solver->net()->layers()[0]) == 0)
		{
		  throw MLLibBadParamException("solver's net's first layer is required to be of MemoryData type");
		}
	      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(solver->net()->layers()[0])->AddDatumVector(inputc._dv);
	    }
	  else
	    {
	      if (boost::dynamic_pointer_cast<caffe::MemorySparseDataLayer<float>>(solver->net()->layers()[0]) == 0)
		{
		  throw MLLibBadParamException("solver's net's first layer is required to be of MemorySparseData type");
		}
	      boost::dynamic_pointer_cast<caffe::MemorySparseDataLayer<float>>(solver->net()->layers()[0])->AddDatumVector(inputc._dv_sparse);
	    }
	}
	catch(std::exception &e)
	  {
	    throw;
	  }
	inputc._dv.clear();
	inputc._dv_sparse.clear();
	inputc._ids.clear();
      }
    if (this->_mlmodel.read_from_repository(this->_mlmodel._repo,this->_logger))
      throw MLLibBadParamException("error reading or listing Caffe models in repository " + this->_mlmodel._repo);
    this->_mlmodel.read_corresp_file();
    if (ad_mllib.has("resume") && ad_mllib.get("resume").get<bool>())
      {
        std::string bestfilename = this->_mlmodel._repo + this->_mlmodel._best_model_filename;
        std::ifstream bestfile;
        try
          {
            std::string tmp;
            bestfile.open(bestfilename,std::ios::in);
            // first three fields are thrown away
            bestfile >> tmp >> tmp >> tmp >> tmp;
            bestfile.close();
            _best_metric_value = std::atof(tmp.c_str());
          }
        catch (std::exception &e)
          {
            this->_logger->info("no previous best model file");
	  }

	if (this->_mlmodel._sstate.empty())
	  {
	    this->_logger->error("resuming a model requires a .solverstate file in model repository");
	    throw MLLibBadParamException("resuming a model requires a .solverstate file in model repository");
	  }
	else 
	  {
	    try
	      {
		solver->Restore(this->_mlmodel._sstate.c_str());
	      }
	    catch(std::exception &e)
	      {
		this->_logger->error("Failed restoring network state");
		throw;
	      }
	  }
      }
    else if (!this->_mlmodel._sstate.empty())
      {
	this->_logger->error("not resuming while a .solverstate file remains in model repository");
	throw MLLibBadParamException("a .solverstate file requires a resume argument for training, otherwise delete existing training state files (with clear=lib) to cleanup the model repository");
      }
    else if (!this->_mlmodel._weights.empty())
      {
	try
	  {
	    solver->net()->CopyTrainedLayersFrom(this->_mlmodel._weights);
	  }
	catch(std::exception &e)
	  {
	    throw;
	  }
      }
    else
      {
	solver->iter_ = 0;
	solver->current_step_ = 0;
      }
    
    if (_gpuid.size() > 1)
      {
	_sync = new caffe::P2PSync<float>(solver,nullptr,solver->param());
	_syncs = std::vector<boost::shared_ptr<caffe::P2PSync<float>>>(_gpuid.size());
	_sync->Prepare(_gpuid, &_syncs); 
	for (size_t i=1;i<_syncs.size();++i)
	  _syncs[i]->StartInternalThread();
      }

    this->_mem_used_train = solver->net_->memory_used();
    for (caffe::shared_ptr<caffe::Net<float > > n: solver->test_nets())
      this->_mem_used_test+= n->memory_used();

    const int start_iter = solver->iter_;
    int average_loss = solver->param_.average_loss();
    std::vector<float> losses;
    if (!ad_mllib.has("resume") || !ad_mllib.get("resume").get<bool>())
      this->clear_all_meas_per_iter();
    float smoothed_loss = 0.0;
    while(solver->iter_ < solver->param_.max_iter()
	  && this->_tjob_running.load())
      {
	this->add_meas("iteration",solver->iter_);

	solver->net_->ClearParamDiffs();
	
	// Save a snapshot if needed.
       bool already_snapshoted = false;
	if (solver->param_.snapshot() && solver->iter_ > start_iter &&
	    solver->iter_ % solver->param_.snapshot() == 0) {
	  solver->Snapshot();
         already_snapshoted = true;
	}
	if (solver->param_.test_interval() && solver->iter_ % solver->param_.test_interval() == 0
	    && (solver->iter_ > 0 || solver->param_.test_initialization())) 
	  {
	    APIData meas_out;
	    solver->test_nets().at(0).get()->ShareTrainedLayersWith(solver->net().get());
	    test(solver->test_nets().at(0).get(),ad,inputc,test_batch_size,has_mean_file,test_iter,meas_out);
           APIData meas_obj = meas_out.getobj("measure");

	    // save best iteration snapshot
	    save_if_best(meas_obj, solver, already_snapshoted);

	    std::vector<std::string> meas_str = meas_obj.list_keys();
	    this->_logger->info("batch size={}",batch_size);
	    
	    for (auto m: meas_str)
	      {
		if (m != "cmdiag" && m != "cmfull" && m != "clacc" && m != "labels" && m!= "cliou") // do not report confusion matrix in server logs
		  {
		    double mval = meas_obj.get(m).get<double>();
		    this->_logger->info("{}={}",m,mval);
		    this->add_meas(m,mval);
		    this->add_meas_per_iter(m,mval);
		  }
		else if (m == "cmdiag" || m == "clacc" || m == "cliou")
		  {
		    std::vector<double> mdiag = meas_obj.get(m).get<std::vector<double>>();
		    std::vector<std::string> cnames;
		    std::string mdiag_str;
		    for (size_t i=0;i<mdiag.size();i++)
		      {
			mdiag_str += this->_mlmodel.get_hcorresp(i) + ":" + std::to_string(mdiag.at(i)) + " ";
			this->add_meas_per_iter(m+'_'+this->_mlmodel.get_hcorresp(i),mdiag.at(i));
			cnames.push_back(this->_mlmodel.get_hcorresp(i));
		      }
		    this->_logger->info("{}=[{}]",m,mdiag_str);
		    this->add_meas(m,mdiag,cnames);
		  }
	      }
	  }
	
	float loss = 0.0;
	float avg_fb_time = 0.0;
	float est_remain_time = 0.0;
	std::string est_remain_time_str;
	try
	  {
	    for (size_t i = 0; i < solver->callbacks().size(); ++i) {
	      solver->callbacks()[i]->on_start();
	    }
	    for (int i = 0; i < solver->param_.iter_size(); ++i)
	      {
		std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
		loss += solver->net_->ForwardBackward();
		std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
		avg_fb_time += std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
	      }
	    loss /= solver->param_.iter_size();
	    avg_fb_time /= solver->param_.iter_size();
	    est_remain_time = avg_fb_time * solver->param_.iter_size() * (solver->param_.max_iter() - solver->iter_) / 1000.0;
	  }
	catch(std::exception &e)
	  {
	    this->_logger->error("exception while forward/backward pass through the network");
	    if (_sync)
	      {
		for (size_t i=1;i<_syncs.size();++i)
		  _syncs[i]->StopInternalThread();
	      }
	    delete _sync;
	    throw;
	  }
	if (static_cast<int>(losses.size()) < average_loss) 
	  {
	    losses.push_back(loss);
	    int size = losses.size();
	    smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
	  } 
	else 
	  {
	    int idx = (solver->iter_ - start_iter) % average_loss;
	    smoothed_loss += (loss - losses[idx]) / average_loss;
	    losses[idx] = loss;
	  }
	this->add_meas("train_loss",smoothed_loss);
	this->add_meas_per_iter("train_loss",smoothed_loss);
	this->add_meas("iter_time",avg_fb_time);
	this->add_meas("remain_time",est_remain_time);
	
	if ((solver->param_.display() && solver->iter_ % solver->param_.display() == 0)
	    || (solver->param_.test_interval() && solver->iter_ % solver->param_.test_interval() == 0))
	  {
	    caffe::SGDSolver<float> *sgd_solver = static_cast<caffe::SGDSolver<float>*>(solver.get());
	    this->_logger->info("Iteration {}, lr = {}, smoothed_loss={}",solver->iter_,sgd_solver->GetLearningRate(),this->get_meas("train_loss"));
	  }
	try
	  {
	    for (size_t i = 0; i < solver->callbacks().size(); ++i) {
	      solver->callbacks()[i]->on_gradients_ready();
	    }
	    solver->iter_++;
	    solver->ApplyUpdate(false); // no iteration logging from within caffe
	  }
	catch (std::exception &e)
	  {
	    this->_logger->error("exception while updating network");
	    if (_sync)
	      {
		for (size_t i=1;i<_syncs.size();++i)
		  _syncs[i]->StopInternalThread();
	      }
	    delete _sync;
	    throw;
	  }
      }

    // always save final snapshot.
    if (solver->param_.snapshot_after_train())
      solver->Snapshot();

    // destroy the net
    delete _net;
    _net = nullptr;

    if (_sync)
      {
	for (size_t i=1;i<_syncs.size();++i)
	  _syncs[i]->StopInternalThread();
	delete _sync;
      }

    // bail on forced stop, i.e. not testing the net further.
    if (!this->_tjob_running.load())
      {
	inputc._dv_test.clear();
	inputc._dv_test_sparse.clear();
	return 0;
      }
    
    solver_param = caffe::SolverParameter();
    if (this->_mlmodel.read_from_repository(this->_mlmodel._repo,this->_logger,true))
      throw MLLibBadParamException("error reading or listing Caffe models in repository " + this->_mlmodel._repo);
    int cm = create_model();
    if (cm == 1)
      throw MLLibInternalException("no model in " + this->_mlmodel._repo + " for initializing the net");
    else if (cm == 2)
      throw MLLibBadParamException("no deploy file in " + this->_mlmodel._repo + " for initializing the net");
    
    // test
    if (!inputc._ctc && !inputc._bbox) // ctc uses the accuracy computed from within the net, and bbox models use special layer as well, can't run test with deploy
      test(_net,ad,inputc,test_batch_size,has_mean_file,test_iter,out);
    inputc._dv_test.clear();
    inputc._dv_test_sparse.clear();

    // add whatever the input connector needs to transmit out
    inputc.response_params(out);

    // if batch_size has been recomputed, let the user know
    if (user_batch_size != batch_size)
      {
	APIData advb;
	advb.add("batch_size",batch_size);
	if (!out.has("parameters"))
	  {
	    APIData adparams;
	    adparams.add("mllib",advb);
	    out.add("parameters",adparams);
	  }
	else
	  {
	    APIData adparams = out.getobj("parameters");
	    adparams.add("mllib",advb);
	    out.add("parameters",adparams);
	  }
      }

    // reset db input connector, ready for in-memory prediction
    if (this->_inputc._db)
      this->_inputc._db = false;

    // if there's target repository, copy relevant data there
    APIData ad_out = ad.getobj("parameters").getobj("output");
    if (ad_out.has("target_repository"))
      this->_mlmodel.copy_to_target(this->_mlmodel._repo,
				    ad_out.get("target_repository").get<std::string>(),
				    this->_logger);
    
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::test(caffe::Net<float> *net,
										 const APIData &ad,
										 TInputConnectorStrategy &inputc,
										 const int &test_batch_size,
										 const bool &has_mean_file,
										 const int &test_iter,
										 APIData &out)
  {
    APIData ad_res;
    ad_res.add("iteration",this->get_meas("iteration"));
    ad_res.add("train_loss",this->get_meas("train_loss"));
    APIData ad_out = ad.getobj("parameters").getobj("output");

    if (ad.getobj("parameters").getobj("mllib").has("ignore_label"))
      {
        int ignore_label = ad.getobj("parameters").getobj("mllib").get("ignore_label").get<int>();
        ad_res.add("ignore_label", ignore_label);
      }

    if (ad_out.has("measure"))
      {
	float mean_loss = 0.0;
	int tresults = 0;
	int nout = _nclasses;
	int inner_meas_iter = 0;
	float net_meas = 0.0;
	if (_regression && _ntargets > 1)
	  nout = _ntargets;
	if (_autoencoder)
	  {
	    if (typeid(inputc) == typeid(ImgCaffeInputFileConn))
	      nout = inputc.channels() * inputc.width() * inputc.height();
	    else nout = inputc.channels();
	  }
	ad_res.add("nclasses",_nclasses);
	inputc.reset_dv_test();
	std::map<int,std::map<int,std::vector<std::pair<float,int>>>> all_true_pos;
	std::map<int,std::map<int,std::vector<std::pair<float,int>>>> all_false_pos;
	std::map<int,std::map<int,int>> all_num_pos;
	while(true)
	  {
	    size_t dv_size = 0;
	    std::vector<float> dv_labels;
	    std::vector<std::vector<double>> dv_float_data;
	    try
	      {
               if (inputc._ctc || inputc._bbox)
		  {
		    // do nothing, db input
		    dv_size = test_batch_size;
		  }
		else if (!inputc._sparse)
		  {
		    std::vector<caffe::Datum> seg_dv;
		    std::vector<caffe::Datum> dv = inputc.get_dv_test(test_batch_size,!_autoencoder ? has_mean_file : false);
		    if (dv.empty())
		      break;
		    dv_size = dv.size();
		    for (size_t s=0;s<dv_size;s++)
		      {
			if (inputc._segmentation)
			  {
			    // dv_labels will need to be of size width x height x batch size -> use dv_float_data
			    // -> read datum 2 by 2 -> skip s (read s 2 by 2)
			    if (!(s % 2))
			      {
				std::vector<double> vals;
				for (int k=0;k<dv.at(s+1).float_data_size();k++)
				  vals.push_back(dv.at(s+1).float_data(k));
				dv_float_data.push_back(vals);
				seg_dv.push_back(dv.at(s));
			      }
			  }
			// imagedata layer
			//- store labels in float_data
			//- similar to segmentation for computing multi-label accuracy
			else if (inputc._multi_label && !(inputc._db) && typeid(inputc) == typeid(ImgCaffeInputFileConn)
				 && _nclasses > 1)
			  {
			    std::vector<double> vals;
			    for (int k=0;k<dv.at(s).float_data_size();k++)
			      vals.push_back(dv.at(s).float_data(k));
			    dv_float_data.push_back(vals);
			  }
                     else if (typeid(inputc) == typeid(CSVTSCaffeInputFileConn))  // timeseries case
                       {
			    std::vector<double> vals;
                         for (int t=0; t<inputc._timesteps; ++t)
                           {
                             for (int k=0; k<_ntargets; ++k)
                               {
                                 vals.push_back(dv.at(s).float_data(t*inputc._datadim+k+1));
                               }
                           }
			    dv_float_data.push_back(vals);
                       }
			else if (!_autoencoder)
			  {
			    dv_labels.push_back(dv.at(s).label());
			    if (_ntargets > 1)
			      {
				std::vector<double> vals;
				for (int k=inputc.channels();k<dv.at(s).float_data_size();k++)
				  vals.push_back(dv.at(s).float_data(k));
				dv_float_data.push_back(vals);
			      }
			  }
			else if (_autoencoder && (typeid(inputc) == typeid(ImgCaffeInputFileConn)))
			  {
			    Datum datum = dv.at(s);
			    int height = datum.height();
			    int width = datum.width();
			    std::vector<double> vals;
			    for (int c=0;c<datum.channels();++c) //TODO: store for each channel or flatten before sigmoid
			      for (int h=0;h<height;++h)
				for (int w=0;w<width;++w)
				  {
				    int data_index = (c*height+h)*width+w;
				    double datum_element;
				    datum_element = static_cast<double>(static_cast<uint8_t>(datum.data()[data_index]));
				    datum_element *= _scale;
				    vals.push_back(datum_element);
				  }
			    dv_float_data.push_back(vals);
			  }
			else
			  {
			    std::vector<double> vals;
			    for (int k=0;k<inputc.channels();k++)
			      {
				vals.push_back(dv.at(s).float_data(k));
			      }
			    dv_float_data.push_back(vals);
			  }
		      }
		    if (boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layers()[0]) == 0)
		      throw MLLibBadParamException("test net's first layer is required to be of MemoryData type");
		    if (inputc._segmentation)
		      {
			dv = seg_dv;
			dv_size = dv.size();
			seg_dv.clear();
		      }
		    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layers()[0])->set_batch_size(dv.size());
		    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layers()[0])->AddDatumVector(dv);
		  }
		else // sparse
		  {
		    std::vector<caffe::SparseDatum> dv = inputc.get_dv_test_sparse(test_batch_size);
		    if (dv.empty())
		      break;
		    dv_size = dv.size();
		    for (size_t s=0;s<dv_size;s++)
		      {
			dv_labels.push_back(dv.at(s).label());
			if (_ntargets > 1)
			  {
			    // SparseDatum has no float_data and source cannot be sliced
			    throw MLLibBadParamException("sparse inputs cannot accomodate multi-target objectives, use single target instead");
			  }
		      }
		    if (boost::dynamic_pointer_cast<caffe::MemorySparseDataLayer<float>>(net->layers()[0]) == 0)
		      throw MLLibBadParamException("test net's first layer is required to be of MemorySparseData type");
		    boost::dynamic_pointer_cast<caffe::MemorySparseDataLayer<float>>(net->layers()[0])->set_batch_size(dv.size());
		    boost::dynamic_pointer_cast<caffe::MemorySparseDataLayer<float>>(net->layers()[0])->AddDatumVector(dv);
		  }
	      }
	    catch(std::exception &e)
	      {
		this->_logger->error("Error while filling up network for testing");
		// XXX: might want to clean up here...
		throw;
	      }
	    float loss = 0.0;
	    std::vector<Blob<float>*> lresults;
	    try
	      {
		lresults = net->Forward(&loss);
	      }
	    catch(std::exception &e)
	      {
		this->_logger->error("Error while proceeding with test forward pass");
		// XXX: might want to clean up here...
		throw;
	      }
	    int slot = lresults.size() - 1;
	    
	    if (_regression && _ntargets > 1) // slicing is involved
	      slot--; // labels appear to be last
	    else if (inputc._multi_label && ( inputc._db || ! (typeid(inputc) == typeid(ImgCaffeInputFileConn))
                                           ||  _nclasses <= 1))
	      slot--;
	    else if (_autoencoder && typeid(inputc) == typeid(ImgCaffeInputFileConn))
	      slot = 0; // flatten output
           else if (_regression && _ntargets == 1 && typeid(inputc) == typeid(CSVCaffeInputFileConn))
             {
               slot = findOutputSlotNumberByBlobName(net, "ip_loss");
             }

           if (typeid(inputc) == typeid(CSVTSCaffeInputFileConn))
             {
               slot = findOutputSlotNumberByBlobName(net,"rnn_pred");
             }

	    int scount = lresults[slot]->count();
	    int scperel = scount / dv_size;

	    if (inputc._ctc) // special case, we're using the accuracy computed from within the network
	      {
		slot = 0;
		net_meas += lresults[slot]->cpu_data()[0];
		if (inner_meas_iter >= test_iter)
		  {
		    net_meas /= static_cast<float>(test_iter);
		    std::vector<double> predictions;
		    predictions.push_back(net_meas);
		    APIData bad;
		    bad.add("pred",predictions);
		    ad_res.add("0",bad);
		    break;
		  }
		++inner_meas_iter;
	      }
	    else if (inputc._bbox)
	      {
		for (size_t j=0;j<lresults.size();j++)
		  {
		    if (lresults[j]->width() != 5)
		      throw MLLibBadParamException("wrong width in bbox result");
		    int pos = tresults + j;
		    const float *result_vec = lresults[j]->cpu_data();
		    int num_det = lresults[j]->height();
		    for (int k=0;k<num_det;k++)
		      {
			int item_id = static_cast<int>(result_vec[k * 5]);
			int label = static_cast<int>(result_vec[k * 5 + 1]);
			if (item_id == -1) {
			  // Special row of storing number of positives for a label.
			  if (all_num_pos[pos].find(label) == all_num_pos[pos].end()) {
			    all_num_pos[pos][label] = static_cast<int>(result_vec[k * 5 + 2]);
			  } else {
			    all_num_pos[pos][label] += static_cast<int>(result_vec[k * 5 + 2]);
			  }
			} else {
			  // Normal row storing detection status.
			  float score = result_vec[k * 5 + 2];
			  int tp = static_cast<int>(result_vec[k * 5 + 3]);
			  int fp = static_cast<int>(result_vec[k * 5 + 4]);
			  if (tp == 0 && fp == 0) {
			    // Ignore such case. It happens when a detection bbox is matched to
			    // a difficult gt bbox and we don't evaluate on difficult gt bbox.
			    this->_logger->warn("skipping bbox");
			    continue;
			  }
			  all_true_pos[pos][label].push_back(std::make_pair(score, tp));
			  all_false_pos[pos][label].push_back(std::make_pair(score, fp));
			}
		      }
		  }

		// wrapping up
		if (inner_meas_iter >= test_iter)
		  {
		    APIData bad;
		    int pos_count = 0;
		    for (size_t i=0;i<all_true_pos.size();i++)
		      {
			if (all_true_pos.find(i) == all_true_pos.end())
			  throw MLLibInternalException("Missing output_blob true_pos: " + std::to_string(i));

			const std::map<int, std::vector<std::pair<float, int> > >& true_pos =
			  all_true_pos.find(i)->second;
			if (all_false_pos.find(i) == all_false_pos.end())
			  throw MLLibInternalException("Missing output_blob false_pos: " + std::to_string(i));
			const std::map<int, std::vector<std::pair<float, int> > >& false_pos =
			  all_false_pos.find(i)->second;
			if (all_num_pos.find(i) == all_num_pos.end())
			  throw MLLibInternalException("Missing output_blob num_pos: " + std::to_string(i));
			const std::map<int, int>& num_pos = all_num_pos.find(i)->second;

			// Sort true_pos and false_pos with descend scores.
			std::vector<APIData> vbad;
			for (std::map<int, int>::const_iterator it = num_pos.begin();
			     it != num_pos.end(); ++it)
			  {
			    APIData lbad;
			    int label = it->first;
			    int label_num_pos = it->second;
			    if (true_pos.find(label) == true_pos.end()) {
			      this->_logger->error("Missing true_pos for label: {}",label);
			      continue;
			    }
			    const std::vector<std::pair<float, int> >& label_true_pos =
			      true_pos.find(label)->second;
			    if (false_pos.find(label) == false_pos.end()) {
			      this->_logger->error("Missing false_pos for label: {}",label);
			      continue;
			    }
			    const std::vector<std::pair<float, int> >& label_false_pos =
			      false_pos.find(label)->second;
			    
			    //XXX: AP computed here, store in apidata instead
			    std::vector<double> tp_d;
			    std::vector<int> tp_i;
			    for (size_t v=0;v<label_true_pos.size();v++)
			      {
				tp_d.push_back(label_true_pos.at(v).first);
				tp_i.push_back(label_true_pos.at(v).second);
			      }

			    std::vector<double> fp_d;
			    std::vector<int> fp_i;
			    for (size_t v=0;v<label_false_pos.size();v++)
			      {
				fp_d.push_back(label_false_pos.at(v).first);
				fp_i.push_back(label_false_pos.at(v).second);
			      }
			    
			    lbad.add("tp_d",tp_d);
			    lbad.add("tp_i",tp_i);
			    lbad.add("fp_d",fp_d);
			    lbad.add("fp_i",fp_i);
			    lbad.add("num_pos",label_num_pos);
			    lbad.add("label",label);
			    vbad.push_back(lbad);
			  }
			bad.add(std::to_string(pos_count),vbad);
			++pos_count;
		      }
		    ad_res.add("0",bad);
		    ad_res.add("pos_count",pos_count);
		    break;
		  }
		++inner_meas_iter;
	      }
	    else
	      {
		for (int j=0;j<(int)dv_size;j++)
		  {
		    APIData bad;
		    std::vector<double> predictions;
		    if (inputc._segmentation)
		      {
			APIData bad2;
			std::vector<double> preds, targets;
			for (size_t l=0;l<dv_float_data.at(j).size();l++)
			  {
			    double target = dv_float_data.at(j).at(l);
			    double best_prob = -1.0;
			    double best_cat = -1.0;
			    if (lresults[slot]->shape(1) != 1)
			      {
				for (int k=0;k<nout;k++)
				  {
				    double prob = lresults[slot]->cpu_data()[l+(nout*j+k)*dv_float_data.at(j).size()];
				    if (prob >= best_prob)
				      {
					best_prob = prob;
					best_cat = k;
				      }
				  }
			      }
			    else
			      {
				// in case of dice loss + deeplab_vgg16, predictions are at pos 0
				double res = lresults[slot]->cpu_data()[l+j*dv_float_data.at(j).size()];
				if (res > 0.5)
				  {
				    best_cat = 1;
				  }
				else
				  {
				    best_cat = 0;
				  }
			      }
			    preds.push_back(best_cat);
			    targets.push_back(target);
			  }
			bad2.add("target",targets);
			bad2.add("pred",preds);
			ad_res.add(std::to_string(tresults+j),bad2);
		      }
		    // multilabel image  => also covers soft labels
		    else if (inputc._multi_label && !inputc._db && typeid(inputc) == typeid(ImgCaffeInputFileConn)
			     && _nclasses > 1)
		      {
			// grab multi-label prediction from last layer
			std::vector<double> targets;
			for (int k=0;k<nout;k++)
			  {
			    double target = dv_float_data.at(j).at(k);
			    targets.push_back(target);
			    predictions.push_back(lresults[slot]->cpu_data()[j*scperel+k]);
			  }
			bad.add("target",targets);
			bad.add("pred",predictions);
		      }
                  else if (typeid(inputc) == typeid(CSVTSCaffeInputFileConn))
                    {
                      std::vector<double> target;
                      for (size_t k=0;k<dv_float_data.at(j).size();k++)
                        {
                          target.push_back(dv_float_data.at(j).at(k));
                        }
                      bad.add("target", target);

                      for (int t=0; t<inputc._timesteps; ++t)
                        {
                          for (int k=0;k<nout;k++)
                            {
                              std::vector<int> loc = {t,j,k};
                              predictions.push_back(lresults[slot]->data_at(loc));
                            }
                        }
                    }
		    else if ((!_regression && !_autoencoder)|| _ntargets == 1)
		      {
			double target = dv_labels.at(j);
			for (int k=0;k<nout;k++)
			  {
			    predictions.push_back(lresults[slot]->cpu_data()[j*scperel+k]);
			  }
			bad.add("target",target);
		      }
		    else // regression with ntargets > 1 or autoencoder
		      {
			std::vector<double> target;
			for (size_t k=0;k<dv_float_data.at(j).size();k++)
			  target.push_back(dv_float_data.at(j).at(k));
			for (int k=0;k<nout;k++)
			  {
			    predictions.push_back(lresults[slot]->cpu_data()[j*scperel+k]);
			  }
			bad.add("target",target);
		      }
		    if (!inputc._segmentation)
		      {
			bad.add("pred",predictions);
			ad_res.add(std::to_string(tresults+j),bad);
		      }
		  }
	      }
	    if (inputc._bbox)
	      tresults += 1;
	    else tresults += dv_size;
	    mean_loss += loss;
	  }
	std::vector<std::string> clnames;
	for (int i=0;i<nout;i++)
	  clnames.push_back(this->_mlmodel.get_hcorresp(i));
	ad_res.add("clnames",clnames);
	if (inputc._segmentation)
	  ad_res.add("segmentation",true);
	if (inputc._multi_label)
	  ad_res.add("multilabel",true);
	if (inputc._bbox)
	  ad_res.add("bbox",true);
	ad_res.add("batch_size",tresults);
	if (_regression)
	  ad_res.add("regression",_regression);
	if (inputc._ctc)
	  ad_res.add("net_meas",true);
       if (typeid(inputc) == typeid(CSVTSCaffeInputFileConn))
         {
           ad_res.add("timeserie",true);
           ad_res.add("timeseries",nout);
         }
      }
    SupervisedOutput::measure(ad_res,ad_out,out);
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad,
										   APIData &out)
  {
    std::lock_guard<std::mutex> lock(_net_mutex); // no concurrent calls since the net is not re-instantiated


    // check for net
    if (!_net || _net->phase() == caffe::TRAIN)
      {
	int cm = create_model(true);
	if (cm != 0)
	  this->_logger->error("Error creating model for prediction");
	if (cm == 1)
	  throw MLLibInternalException("no model in " + this->_mlmodel._repo + " for initializing the net");
	else if (cm == 2)
	  throw MLLibBadParamException("no deploy file in " + this->_mlmodel._repo + " for initializing the net");
      }

    TInputConnectorStrategy inputc(this->_inputc);
    TOutputConnectorStrategy tout;
    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
    APIData ad_output = ad.getobj("parameters").getobj("output");
    bool bbox = false;
    bool rois = false;
    bool multibox_rois = false;
    bool ctc = false;
    int blank_label = -1;
    std::string roi_layer;
    double confidence_threshold = 0.0;
    if (ad_output.has("confidence_threshold"))
      {
	try
	  {
	    confidence_threshold = ad_output.get("confidence_threshold").get<double>();
	  }
	catch(std::exception &e)
	  {
	    // try from int
	    confidence_threshold = static_cast<double>(ad_output.get("confidence_threshold").get<int>());
	  }
      }


    if (typeid(inputc) == typeid(CSVTSCaffeInputFileConn)
        && ad.getobj("parameters").getobj("input").has("timesteps"))
      {
        int timesteps = ad.getobj("parameters").getobj("input").get("timesteps").get<int>();

        bool changed = update_timesteps(timesteps);
        if (changed)
          {
            int cm = create_model(true);
            if (cm != 0)
              this->_logger->error("Error creating model for prediction");
            if (cm == 1)
              throw MLLibInternalException("no model in " + this->_mlmodel._repo + " for initializing the net");
            else if (cm == 2)
              throw MLLibBadParamException("no deploy file in " + this->_mlmodel._repo + " for initializing the net");
          }
      }


    if (ad_output.has("bbox") && ad_output.get("bbox").get<bool>())
      bbox = true;
    if (ad_output.has("rois")) {
      roi_layer =  ad_output.get("rois").get<std::string>();
      rois = true;
      if (ad_output.has("multibox_rois"))
	multibox_rois = ad_output.get("multibox_rois").get<bool>();
    }
    if (ad_output.has("ctc"))
      {
	ctc = ad_output.get("ctc").get<bool>();
	if (ad_output.has("blank_label"))
	  blank_label = ad_output.get("blank_label").get<int>();
      }

    // gpu
#if !defined(CPU_ONLY) && !defined(USE_CAFFE_CPU_ONLY)
    bool gpu = _gpu;
    if (ad_mllib.has("gpu"))
      {
	gpu = ad_mllib.get("gpu").get<bool>();
	if (gpu)
	  {
	    set_gpuid(ad_mllib);
	  }
      }
    if (gpu)
      {
	for (auto i: _gpuid)
	  {
	    Caffe::SetDevice(i);
	    if (gpu != _gpu)
	      Caffe::DeviceQuery();
	  }
	Caffe::set_mode(Caffe::GPU);
      }
    else Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::CPU);
#endif

    APIData cad = ad;
    bool has_mean_file = this->_mlmodel._has_mean_file;
    if (_autoencoder)
      has_mean_file = false;
    cad.add("has_mean_file",has_mean_file);
    if (ad_output.has("measure"))
      {
	try
	  {
	    inputc.transform(cad);
	  }
	catch (std::exception &e)
	  {
	    throw;
	  }

	int batch_size = inputc.test_batch_size();
	if (ad_mllib.has("net"))
	  {
	    APIData ad_net = ad_mllib.getobj("net");
	    if (ad_net.has("test_batch_size"))
	      batch_size = ad_net.get("test_batch_size").get<int>();
	  }

	bool has_mean_file = this->_mlmodel._has_mean_file;
	test(_net,ad,inputc,batch_size,has_mean_file,1,out);
	APIData out_meas = out.getobj("measure");
	out_meas.erase("train_loss");
	out_meas.erase("iteration");
	out.add("measure",out_meas);
	return 0;
      }
    
    std::string extract_layer;
    if (ad_mllib.has("extract_layer"))
      extract_layer = ad_mllib.get("extract_layer").get<std::string>();
      
    try
      {
        inputc.transform(cad);
      }
    catch (std::exception &e)
      {
        throw;
      }
    int batch_size = inputc.test_batch_size();
    if (ad_mllib.has("net"))
      {
	APIData ad_net = ad_mllib.getobj("net");
	if (ad_net.has("test_batch_size"))
	  batch_size = ad_net.get("test_batch_size").get<int>();
      }
    inputc.reset_dv_test();
    std::vector<APIData> vrad;
    int nclasses = -1;
    int idoffset = 0;
    std::vector<APIData> series;
    int serieNum = 0;

    while(true)
      {
	try
	  {
           if (!inputc._sparse)
	      {
		std::vector<Datum> dv = inputc.get_dv_test(batch_size,has_mean_file);
		if (dv.empty())
                {
                  if (typeid(inputc) == typeid(CSVTSCaffeInputFileConn)) // timeseries
                    // in case of time series, need to output data of last serie at end
                    {
                      if (series.size() > 0)
                        {
                          APIData out;
                          if (!inputc._ids.empty())
                            out.add("uri",inputc._ids.at(serieNum++));
                          else out.add("uri",std::to_string(serieNum++));
                          out.add("series", series);
                          out.add("probs",std::vector<double>(series.size(),1.0));
                          out.add("loss",0.0);
                          vrad.push_back(out);
                        }
                      series.clear();
                    }
                  break;
                }
		batch_size = dv.size();
		if (boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0]) == 0)
		    {
		      this->_logger->info("deploy net's first layer is required to be of MemoryData type (predict)");
		      delete _net;
		      _net = nullptr;
		      throw MLLibBadParamException("deploy net's first layer is required to be of MemoryData type");
		    }
		boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->set_batch_size(batch_size);
		boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddDatumVector(dv);
	      }
	    else
	      {
		std::vector<caffe::SparseDatum> dv = inputc.get_dv_test_sparse(batch_size);
		if (dv.empty())
		  break;
		batch_size = dv.size();
		if (boost::dynamic_pointer_cast<caffe::MemorySparseDataLayer<float>>(_net->layers()[0]) == 0)
		  {
		    this->_logger->error("deploy net's first layer is required to be of MemoryData type (predict)");
		    delete _net;
		    _net = nullptr;
		    throw MLLibBadParamException("deploy net's first layer is required to be of MemorySparseData type");
		  }
		boost::dynamic_pointer_cast<caffe::MemorySparseDataLayer<float>>(_net->layers()[0])->set_batch_size(batch_size);
		boost::dynamic_pointer_cast<caffe::MemorySparseDataLayer<float>>(_net->layers()[0])->AddDatumVector(dv);
	      }
	  }
	catch(std::exception &e)
	  {
	    this->_logger->error("exception while filling up network for prediction");
	    delete _net;
	    _net = nullptr;
	    throw;
	  }

       this->_mem_used_test = _net->memory_used();

	float loss = 0.0;
	if (extract_layer.empty() || inputc._segmentation) // supervised or segmentation
	  {
	    std::vector<Blob<float>*> results;
	    
	    if (rois)
	      {
		std::map<std::string,int> n_layer_names_index = _net->layer_names_index();
		std::map<std::string,int>::const_iterator lit;
		if ((lit=n_layer_names_index.find(roi_layer))==n_layer_names_index.end())
		  throw MLLibBadParamException("unknown rois layer " + roi_layer);
		int li = (*lit).second;
		try
		  {
		    loss = _net->ForwardFromTo(0,li);
		  }
		catch(std::exception &e)
		  {
		    this->_logger->error("Error while proceeding with supervised prediction forward pass, not enough memory? {}",e.what());
		    delete _net;
		    _net = nullptr;
		    throw;
		  }
		const std::vector<std::vector<Blob<float>*>>& rresults = _net->top_vecs();
		results = rresults.at(li);
	      }
	    else
	      {
		try
		  {
		    results = _net->Forward(&loss);
		  }
		catch(std::exception &e)
		  {
		    this->_logger->error("Error while proceeding with supervised prediction forward pass, not enough memory? {}",e.what());
		    delete _net;
		    _net = nullptr;
		    throw;
		  }
	      }
	    
	    if (inputc._segmentation)
	      {
               int slot = results.size() - 1;
               nclasses = _nclasses;

               bool conf_best = false;
               std::vector<bool> confidences;
               if (_nclasses == 1)
                 confidences = std::vector<bool>(2,false);
               else
                 confidences = std::vector<bool>(_nclasses,false);
               if (ad_output.has("confidences"))
                 {
                   std::vector<std::string> smaps =
                     ad_output.get("confidences").get<std::vector<std::string>>();
                   for (std::string s: smaps)
                     if (s == "all")
                       for (int ci = 0; ci<_nclasses; ++ci)
                         confidences[ci] = true;
                     else if (s=="best")
                       conf_best = true;
                     else
                       confidences[std::stoi(s)]= true;
                 }

		for (int j=0;j<batch_size;j++)
		  {
		    APIData rad;
		    std::string uri;
		    if (!inputc._ids.empty())
		      {
			uri = inputc._ids.at(idoffset+j);
			rad.add("uri",uri);
		      }
		    else
		      {
			uri = std::to_string(idoffset+j);
			rad.add("uri",uri);
		      }
		    rad.add("loss",loss);
		    std::vector<double> vals;
                  std::map<int,std::vector<double>> confidence_maps;
                  std::vector<double> conf_map_best;
		    int imgsize = inputc.width()*inputc.height();
		    for (int i=0;i<imgsize;i++)
		      {
			double max_prob = -1.0;
			double best_cat = -1.0;
			if (results[slot]->shape(1) != 1)
			  {
			    for (int k=0;k<nclasses;k++)
			      {
				double prob = results[slot]->cpu_data()[(j*nclasses+k)*imgsize+i];
                            if (confidences[k])
                              confidence_maps[k].push_back(prob);
				if (prob > max_prob)
				  {
				    max_prob = prob;
				    best_cat = static_cast<double>(k);
				  }
			      }
                         if (conf_best)
                           conf_map_best.push_back(max_prob);
			  }
			else
			  {
			    double prob = results[slot]->cpu_data()[(j)*imgsize+i];
                         if (confidences[1])
                           confidence_maps[1].push_back(prob);
                         if (confidences[0])
                           confidence_maps[0].push_back(1.0-prob);
			    if (prob > 0.5)
                           {
                             if (conf_best)
                               conf_map_best.push_back(prob);
                             best_cat = 1;
                           }
			    else
                           {
                             if (conf_best)
                               conf_map_best.push_back(1.0-prob);
                             best_cat = 0;
                           }
			  }
			vals.push_back(best_cat);
		      }
		    auto bit = inputc._imgs_size.find(uri);
		    APIData ad_imgsize;
		    ad_imgsize.add("height",(*bit).second.first);
		    ad_imgsize.add("width",(*bit).second.second);
		    rad.add("imgsize",ad_imgsize);
		    if (imgsize != (*bit).second.first*(*bit).second.second) // resizing output segmentation array
		      {
                      vals =img_resize(vals,inputc.height(),inputc.width(),(*bit).second.first, (*bit).second.second,true);
                     if (conf_best)
                       conf_map_best = img_resize(conf_map_best, inputc.height(), inputc.width(),
                                                  (*bit).second.first, (*bit).second.second,false);
                     for (int ci=0; ci <_nclasses; ++ci)
                       if (confidences[ci])
                       confidence_maps[ci] =
                         img_resize(confidence_maps[ci],
                                    inputc.height(), inputc.width(),
                                     (*bit).second.first, (*bit).second.second, false);
		      }
		    rad.add("vals",vals);
                  if (conf_best || !confidence_maps.empty())
                    {
                      APIData confs;
                      if (conf_best)
                        confs.add("best", conf_map_best);
                      for (int ci=0; ci < _nclasses; ++ci)
                        if (confidences[ci])
                          confs.add(std::to_string(ci), confidence_maps[ci]);
                      rad.add("confidences",confs);
                    }
		    vrad.push_back(rad);
		  }
	      }
	    else if (bbox) // in-image object detection
	      {
		int results_height = results[0]->height();
		const int det_size = 7;
		const float *outr = results[0]->cpu_data();
		for (int j=0;j<batch_size;j++)
		  {
		    int k = 0;
		    std::vector<double> probs;
		    std::vector<std::string> cats;
		    std::vector<APIData> bboxes;
		    APIData rad;
		    std::string uri = inputc._ids.at(idoffset+j);
		    auto bit = inputc._imgs_size.find(uri);
		    int rows = 1;
		    int cols = 1;
		    if (bit != inputc._imgs_size.end())
		      {
			// original image size
			rows = (*bit).second.first;
			cols = (*bit).second.second;
		      }
		    else
		      {
			this->_logger->error("couldn't find original image size for {}",uri);
		      }
		    bool leave = false;
		    int curi = -1;
		    while(true && k<results_height)
		      {
			if (outr[0] == -1)
			  {
			    // skipping invalid detection
			    this->_logger->error("skipping invalid detection");
			    outr += det_size;
			    leave = true;
			    break;
			  }
			std::vector<float> detection(outr, outr + det_size);
			if (curi == -1)
			  curi = detection[0]; // first pass
			else if (curi != detection[0])
			  break; // this belongs to next image
			++k;
			outr += det_size;
			if (detection[2] < confidence_threshold)
			  continue;
			probs.push_back(detection[2]);
			cats.push_back(this->_mlmodel.get_hcorresp(detection[1]));
			APIData ad_bbox;
			ad_bbox.add("xmin",detection[3]*cols);
			ad_bbox.add("ymax",detection[4]*rows);
			ad_bbox.add("xmax",detection[5]*cols);
			ad_bbox.add("ymin",detection[6]*rows);
			bboxes.push_back(ad_bbox);
		      }
		    if (leave)
		      continue;
		    rad.add("uri",uri);
		    rad.add("loss",0.0); // XXX: unused
		    rad.add("probs",probs);
		    rad.add("cats",cats);
		    rad.add("bboxes",bboxes); 
		    vrad.push_back(rad);
		  }
	      }
	    else if (rois) {
	      // adhoc code for roi extractions
	      // nblobs should be 5: 0 is id (in batch)
	      // 1 is label
	      // 2 is confidence
	      // 3 is coord
	      // 4 is pooled_data
	      int nroi = results[0]->shape(0);
	      // 1 et un seul val par reply
	      int max_id = -1;
	      for (int iroi=0; iroi<nroi; ++iroi)
		if (std::round(results[0]->cpu_data()[iroi]) > max_id)
		  max_id = std::round(results[0]->cpu_data()[iroi]);
	      
	      // loop over images
	      for (int iid=0; iid<=max_id; ++iid) {
		APIData rad;
		std::vector<APIData> rois;
		// loop overs rois
		std::vector<double> probs;
		std::vector<std::string> cats;
		std::vector<APIData> bboxes;
		std::vector<APIData> vals;
		std::string uri = inputc._ids.at(idoffset+iid);
		auto bit = inputc._imgs_size.find(uri);
		int rows = 1;
		int cols = 1;
		if (bit != inputc._imgs_size.end())
		  {
		    // original image size
		    rows = (*bit).second.first;
		    cols = (*bit).second.second;
		  }
		else
		  {
		    this->_logger->error("couldn't find original image size for {}",uri);
		  }
		
		for (int iroi=0; iroi<nroi; ++iroi) {
		  // if cat == -1 no detection has been done on image
		  if (results[1]->cpu_data()[iroi] == -1)
		    continue;
		  // check if current roi belongs to current image
		  if (std::round(results[0]->cpu_data()[iroi]) == iid) {
		    if (results[2]->cpu_data()[iroi] < confidence_threshold)
		      continue;
		    APIData roi;
		    //roi.add("cat",results[1]->cpu_data()[iroi]);
		    cats.push_back(this->_mlmodel.get_hcorresp(results[1]->cpu_data()[iroi]));
		    probs.push_back(results[2]->cpu_data()[iroi]);
		    APIData ad_bbox;
		    ad_bbox.add("xmin",results[3]->cpu_data()[iroi*4]*cols);
		    ad_bbox.add("ymax",results[3]->cpu_data()[iroi*4+1]*rows);
		    ad_bbox.add("xmax",results[3]->cpu_data()[iroi*4+2]*cols);
		    ad_bbox.add("ymin",results[3]->cpu_data()[iroi*4+3]*rows);
		    bboxes.push_back(ad_bbox);
		    std::vector<double> pooled_data;
		    int poolsize = results.at(4)->count()/nroi;
		    for (int idata = 0; idata < poolsize; ++idata)
		      pooled_data.push_back(results.at(4)->cpu_data()[iroi*poolsize+idata]);
		    APIData rval;
		    rval.add("vals",pooled_data);
		    vals.push_back(rval);
		  }//end if roi in image
		} // end loop over all rois
		rad.add("vals",vals);
		rad.add("bboxes", bboxes);
		rad.add("uri",inputc._ids.at(idoffset+iid));
		rad.add("loss",0.0); // XXX: unused
		rad.add("probs",probs);
		rad.add("cats",cats);
		vrad.push_back(rad);
	      }
	    }
	    else if (ctc)
	    {
	      // input is time_step x batch_size x alphabet_size
	      int slot = results.size() - 1;
	      const int alphabet_size = results[slot]->shape(2);
	      const int time_step = results[slot]->shape(0);
	      const float *pred_data = results[slot]->cpu_data();
	      for (int j=0;j<batch_size;j++)
		{
		  std::vector<int> pred_label_seq_with_blank(time_step);
		  std::vector<std::vector<float>> pred_sample;

		  const float *pred_cur = pred_data;
		  pred_cur += j*alphabet_size;
		  for (int t=0;t<time_step;t++)
		    {
		      pred_label_seq_with_blank[t] = std::max_element(pred_cur, pred_cur + alphabet_size) - pred_cur;
		      pred_cur += batch_size * alphabet_size;
		    }

		  // get labels seq
		  std::vector<int> pred_label_seq;
		  int prev = blank_label;
		  for(int l = 0; l < time_step; ++l)
		    {
		      int cur = pred_label_seq_with_blank[l];
		      if(cur != prev && cur != blank_label)
			pred_label_seq.push_back(cur);
		      prev = cur;
		    }
		  APIData outseq;
		  std::string outstr;
		  std::ostringstream oss;
		  for (auto l: pred_label_seq)
		    {
		      outstr += char(std::atoi(this->_mlmodel.get_hcorresp(l).c_str()));
		      //utf8::append(this->_mlmodel.get_hcorresp(l),outstr);
		    }
		  std::vector<std::string> cats;
		  cats.push_back(outstr);
		  if (!inputc._ids.empty())
		    outseq.add("uri",inputc._ids.at(idoffset+j));
		  else outseq.add("uri",std::to_string(idoffset+j));
		  outseq.add("cats",cats);
		  outseq.add("probs",std::vector<double>(1,1.0)); //XXX: in raw pred_label_seq_with_blank
		  outseq.add("loss",0.0);
		  vrad.push_back(outseq);
		}
	    }
           else if (typeid(inputc) == typeid(CSVTSCaffeInputFileConn)) // timeseries
             {
               int slot = findOutputSlotNumberByBlobName(_net,"rnn_pred");
               //results[slot] is TxNxDataDim , N is batchsize ...
               int nout = _ntargets;

               const boost::shared_ptr<Blob<float>> contseq = _net->blob_by_name("cont_seq");
               //cont_seq is TxN

               CSVTSCaffeInputFileConn* ic =
                 reinterpret_cast<CSVTSCaffeInputFileConn*>(&inputc);

               for (int j=0; j<batch_size; ++j)
                 {
                   for (int t=0; t<ic->_timesteps; ++t)
                     {
                       std::vector<int> loc0 = {t,j};
                       if (contseq->data_at(loc0) == 0 && series.size() != 0)
                         // new seq -> push old one
                         {
                           if (series.size() > 0)
                             {
                               APIData out;
                               if (!inputc._ids.empty())
                                 out.add("uri",inputc._ids.at(serieNum++));
                               else out.add("uri",std::to_string(serieNum++));
                               out.add("series", series);
                               out.add("probs",std::vector<double>(series.size(),1.0));
                               out.add("loss",0.0);
                               vrad.push_back(out);
                             }
                           series.clear();
                         }

                       std::vector<double> predictions;
                       for (int k=0; k<nout; ++k)
                         {
                           std::vector<int> loc = {t,j,k};
                           if (ic->_min_vals.empty() || ic->_max_vals.empty())
                             {
                               this->_logger->info("not unscaling output because no bounds data found");
                               predictions.push_back(results[slot]->data_at(loc));
                             }
                           else
                             {
                               double res = results[slot]->data_at(loc);
                               double max = ic->_max_vals[ic->_label_pos[k]];
                               double min = ic->_min_vals[ic->_label_pos[k]];
                               double unscaled_res = res * (max - min) + min;
                               predictions.push_back(unscaled_res);
                             }
                         }
                       APIData ts;
                       ts.add("out", predictions);
                       series.push_back(ts);
                     }
                 }
             }
	    else // classification
	      {
		int slot = results.size() - 1;
		if (_regression)
		  {
		    if (_ntargets > 1)
		      slot = 1;
		    else slot = 0; // XXX: more in-depth testing required
		  }
		int scount = results[slot]->count();
		int scperel = scount / batch_size;
		nclasses = scperel;
		if (_autoencoder)
		  nclasses = scperel = 1;
		for (int j=0;j<batch_size;j++)
		  {
		    APIData rad;
		    if (!inputc._ids.empty())
		      rad.add("uri",inputc._ids.at(idoffset+j));
		    else rad.add("uri",std::to_string(idoffset+j));
		    rad.add("loss",loss);
		    std::vector<double> probs;
		    std::vector<std::string> cats;
		    for (int i=0;i<nclasses;i++)
		      {
			double prob = results[slot]->cpu_data()[j*scperel+i];
			if (prob < confidence_threshold)
			  continue;
			probs.push_back(prob);
			cats.push_back(this->_mlmodel.get_hcorresp(i));
		      }
		    rad.add("probs",probs);
		    rad.add("cats",cats);
		    vrad.push_back(rad);
		  }
	      }
	  }
	else // unsupervised
	  {
	    std::map<std::string,int> n_layer_names_index = _net->layer_names_index();
	    std::map<std::string,int>::const_iterator lit;
	    if ((lit=n_layer_names_index.find(extract_layer))==n_layer_names_index.end())
	      throw MLLibBadParamException("unknown extract layer " + extract_layer);
	    int li = (*lit).second;
	    try
	      {
		loss = _net->ForwardFromTo(0,li);
	      }
	    catch(std::exception &e)
	      {
		this->_logger->error("Error while proceeding with unsupervised prediction forward pass, not enough memory? {}",e.what());
		delete _net;
		_net = nullptr;
		throw;
	      }

	    const std::vector<std::vector<Blob<float>*>>& rresults = _net->top_vecs();
	    std::vector<Blob<float>*> results = rresults.at(li);

	    int slot = 0;
	    int scount = results[slot]->count();
	    int scperel = scount / batch_size;
	    std::vector<int> vshape = {batch_size,scperel};
	    results[slot]->Reshape(vshape); // reshaping into a rectangle, first side = batch size
	    for (int j=0;j<batch_size;j++)
	      {
		APIData rad;
		rad.add("uri",inputc._ids.at(idoffset+j));
		rad.add("loss",loss);
		std::vector<double> vals;
		int cpos = 0;
		for (int c=0;c<results.at(slot)->shape(1);c++)
		  {
		    vals.push_back(results.at(slot)->cpu_data()[j*scperel+cpos]);
		    ++cpos;
		  }
		rad.add("vals",vals);
		vrad.push_back(rad);
	      }
	  }
	idoffset += batch_size;
      } // end prediction loop over batches

    if (!inputc._segmentation)
      tout.add_results(vrad);
    if (extract_layer.empty())
      {
	if (_regression)
	  {
	    out.add("regression",true);
	  }
	else if (_autoencoder)
	  {
	    out.add("autoencoder",true);
	  }
       if (typeid(inputc) == typeid(CSVTSCaffeInputFileConn))
         {
           out.add("timeseries",true);
         }
      }
    
    out.add("nclasses",nclasses);
    out.add("bbox",bbox);
    out.add("roi",rois);
    out.add("multibox_rois",multibox_rois);
    if (!inputc._segmentation)
      tout.finalize(ad.getobj("parameters").getobj("output"),out,static_cast<MLModel*>(&this->_mlmodel));
    else // segmentation returns an array, best dealt with an unsupervised connector
      {
	UnsupervisedOutput unsupo;
	unsupo.add_results(vrad);
	unsupo.finalize(ad.getobj("parameters").getobj("output"),out,static_cast<MLModel*>(&this->_mlmodel));
      }
    out.add("status",0);
    
    return 0;
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_in_memory_net_and_solver(caffe::SolverParameter &sp,
													    const APIData &ad,
													    const TInputConnectorStrategy &inputc,
													    bool &has_mean_file,
													    int &user_batch_size,
													    int &batch_size,
													    int &test_batch_size,
													    int &test_iter)
  {
    // fix net model path.
    sp.set_net(this->_mlmodel._repo + "/" + sp.net());
    
    // fix net snapshot path.
    sp.set_snapshot_prefix(this->_mlmodel._repo + "/model");
    
    // acquire custom batch size if any
    user_batch_size = batch_size = inputc.batch_size();
    test_batch_size = inputc.test_batch_size();
    test_iter = -1;
    fix_batch_size(ad,inputc,user_batch_size,batch_size,test_batch_size,test_iter);
    if (test_iter != -1) // has changed
	sp.set_test_iter(0,test_iter);
    
    // fix source paths in the model.
    caffe::NetParameter *np = sp.mutable_net_param();
    if (!caffe::ReadProtoFromTextFile(sp.net().c_str(),np))
      throw MLLibInternalException("failed opening train prototxt file " + sp.net());

    this->_logger->info("input db = {}",inputc._db);
    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
    if (!inputc._db && !inputc._bbox && typeid(inputc) == typeid(ImgCaffeInputFileConn))
      {
	caffe::LayerParameter *lparam = np->mutable_layer(0);
        caffe::ImageDataParameter* image_data_parameter = lparam->mutable_image_data_param();
        image_data_parameter->set_batch_size(user_batch_size);
	if (ad_mllib.has("resume") && ad_mllib.get("resume").get<bool>())
	  {
	    if (ad_mllib.getobj("solver").has("rand_skip"))
	      {
		image_data_parameter->set_rand_skip(ad_mllib.getobj("solver").get("rand_skip").get<int>());
	      }
	  }
      }

    for (int i=0;i<2;i++)
      {
	caffe::LayerParameter *lp = np->mutable_layer(i);
	if (lp->has_data_param())
	  {
	    caffe::DataParameter *dp = lp->mutable_data_param();
	    if (dp->has_source())
	      {
		if (i == 0 && ad.has("db")) // training
		  {
		    dp->set_source(ad.getobj("db").get("train_db").get<std::string>());
		  }
		else if (i == 1 && ad.has("db"))
		  {
		    dp->set_source(ad.getobj("db").get("test_db").get<std::string>());
		  }
		else 
		  {
		    dp->set_source(this->_mlmodel._repo + "/" + dp->source()); // this updates in-memory net
		  }
	      }
	    if (dp->has_batch_size() && batch_size != inputc.batch_size() && batch_size > 0)
	      {
		dp->set_batch_size(user_batch_size); // data params seem to handle batch_size that are no multiple of the training set
		batch_size = user_batch_size;
	      }
	  }
	#ifdef USE_HDF5
	else if (lp->has_hdf5_data_param())
	  {
	    caffe::HDF5DataParameter *dp = lp->mutable_hdf5_data_param();
	    if (dp->has_source())
	      {
		if (i == 0)
		  {
		    dp->set_source(ad.getobj("db").get("train_db").get<std::string>());
		    if (dp->has_batch_size() && batch_size != inputc.batch_size() && batch_size > 0)
		      dp->set_batch_size(user_batch_size);
		  }
		else if (i == 1)
		  {
		    dp->set_source(ad.getobj("db").get("test_db").get<std::string>());
		    dp->set_batch_size(test_batch_size);
		  }
	      }
	    dp->set_image(true);
	  }
	#endif // USE_HDF5
	else if (lp->has_dense_image_data_param())
	  {
	    caffe::DenseImageDataParameter *dp = lp->mutable_dense_image_data_param();
	    if (dp->has_source())
	      {
		if (i == 0)
		  dp->set_source(ad.getobj("source").get("source_train").get<std::string>());
		else dp->set_source(ad.getobj("source").get("source_test").get<std::string>());
	      }
	    if (dp->has_batch_size() && batch_size != inputc.batch_size() && batch_size > 0)
	      {
		dp->set_batch_size(user_batch_size);
		batch_size = user_batch_size;
	      }
	    if (ad_mllib.has("resume") && ad_mllib.get("resume").get<bool>())
	      {
		if (ad_mllib.getobj("solver").has("rand_skip"))
		  {
		    dp->set_rand_skip(ad_mllib.getobj("solver").get("rand_skip").get<int>());
		  }
	      }
	  }
	else if (lp->has_memory_data_param())
	  {
	    caffe::MemoryDataParameter *mdp = lp->mutable_memory_data_param();
	    if (mdp->has_batch_size() && batch_size > 0 &&
               ( (batch_size != inputc.batch_size()) || (typeid(inputc) == typeid(CSVTSCaffeInputFileConn)) ))
	      {
		if (i == 0) // training
		  mdp->set_batch_size(batch_size);
		else mdp->set_batch_size(test_batch_size);
	      }
	  }
	if (!_autoencoder && (lp->has_transform_param() || inputc._has_mean_file || !inputc._mean_values.empty()))
	  {
	    caffe::TransformationParameter *tp = lp->mutable_transform_param();
	    has_mean_file = tp->has_mean_file();
	    if (tp->has_mean_file() || !inputc._mean_values.empty())
	      {
		if (tp->crop_size() == 0) // beware if no mean file available, though should not happen
		  {
		    if (inputc._mean_values.empty())
		      {
			if (ad.has("db"))
			  tp->set_mean_file(ad.getobj("db").get("meanfile").get<std::string>());
			else tp->set_mean_file(this->_mlmodel._repo + "/" + tp->mean_file());
		      }
		    else
		      {
			for (size_t d=0;d<inputc._mean_values.size();d++)
			  tp->add_mean_value(inputc._mean_values.at(d));
			tp->clear_mean_file();
		      }
		  }
		else
		  {
		    for (size_t d=0;d<inputc._mean_values.size();d++)
		      tp->add_mean_value(inputc._mean_values.at(d));
		    tp->clear_mean_file();
		  }
	      }
	  }
      }

    if (typeid(inputc) == typeid(CSVTSCaffeInputFileConn))
      {
        caffe::LayerParameter *loss_scale_layer_param = find_layer_by_name(*np,"Loss_Scale");
        if (loss_scale_layer_param != NULL)
          {
            loss_scale_layer_param->mutable_scale_param()->mutable_filler()->
              set_value(1.0/(float)_ntargets/(float)batch_size);
          }
      }


    //caffe::WriteProtoToTextFile(*np,sp.net().c_str());
    sp.clear_net();
  }


  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  bool CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_timesteps(int timesteps)
  {
    std::string deploy_file = this->_mlmodel._repo + "/deploy.prototxt";
    caffe::NetParameter deploy_net_param;
    caffe::ReadProtoFromTextFile(deploy_file,&deploy_net_param);

    caffe::LayerParameter *lparam = deploy_net_param.mutable_layer(0);
    int orig_timesteps = lparam->memory_data_param().channels();
    if (orig_timesteps == timesteps)
      return false;
    lparam->mutable_memory_data_param()->set_channels(timesteps);
    caffe::WriteProtoToTextFile(deploy_net_param,deploy_file);
    return true;
  }


  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_deploy_protofile_softmax(const APIData &ad)
  {
    if (ad.has("scaling_temperature"))
      {
	double temperature = ad.get("scaling_temperature").get<double>();
	std::string deploy_file = this->_mlmodel._repo + "/deploy.prototxt";
	caffe::NetParameter deploy_net_param;
	caffe::ReadProtoFromTextFile(deploy_file,&deploy_net_param);
	int k = deploy_net_param.layer_size();
	for (int l=k-1;l>0;l--)
	  {
	    caffe::LayerParameter *lparam = deploy_net_param.mutable_layer(l);
	    if (lparam->type() == "Softmax")
	      {
		lparam->mutable_softmax_param()->set_scaling_temperature(temperature);
	      }
	    // we don't break in case multiple softmax heads.
	  }
	caffe::WriteProtoToTextFile(deploy_net_param,deploy_file);
      }
    else return;
  }
  
  // XXX: we are no more pre-setting the batch_size to the data set values (e.g. number of samples)
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_protofile_net(const std::string &net_file,
												 const std::string &deploy_file,
												 const TInputConnectorStrategy &inputc,
												 const bool &has_class_weights,
												 const int &ignore_label,
												 const int &timesteps)
  {
    caffe::NetParameter net_param;
    caffe::ReadProtoFromTextFile(net_file,&net_param); //TODO: catch parsing error (returns bool true on success)
    int width = inputc.width();
    int height = inputc.height();
    bool has_embed = false;
    if (_crop_size > 0)
      width = height = _crop_size;

    if (!inputc._db && !inputc._bbox && !inputc._ctc && typeid(this->_inputc) == typeid(ImgCaffeInputFileConn))
      {
            caffe::LayerParameter *lparam = net_param.mutable_layer(0);
            caffe::ImageDataParameter* image_data_parameter = lparam->mutable_image_data_param();
            image_data_parameter->set_source(inputc._uris.at(0));
	    image_data_parameter->set_root_folder(inputc._root_folder);
            image_data_parameter->set_batch_size(inputc.batch_size());
            image_data_parameter->set_new_height(inputc.height());
            image_data_parameter->set_new_width(inputc.width());
      }

    if (this->_inputc._ctc) // crnn
      {
	int k = net_param.layer_size();
	for (int l=0;l<k;l++)
	  {
	    caffe::LayerParameter *lparam = net_param.mutable_layer(l);
	    if (lparam->type() == "Reshape")
	      {
		lparam->mutable_reshape_param()->mutable_shape()->set_dim(2,timesteps);
	      }
	    else if (lparam->type() == "ContinuationIndicator")
	      {
		lparam->mutable_continuation_indicator_param()->set_time_step(timesteps);
	      }
	    else if (lparam->type() == "CtcLoss")
	      {
		lparam->mutable_ctc_loss_param()->set_time_step(timesteps);
		lparam->mutable_ctc_loss_param()->set_alphabet_size(inputc._alphabet_size);
	      }
	    else if (lparam->type() == "InnerProduct") // last layer before ctcloss
	      {
		lparam->mutable_inner_product_param()->set_num_output(inputc._alphabet_size);
	      }
	  }
      }

    if (net_param.mutable_layer(0)->has_memory_data_param()
	|| net_param.mutable_layer(1)->has_memory_data_param())
      {
        if (_ntargets == 0 || _ntargets == 1 || (typeid(this->_inputc) == typeid(CSVTSCaffeInputFileConn)))
	  {
	    if (net_param.mutable_layer(0)->has_memory_data_param())
	      {
		net_param.mutable_layer(0)->mutable_memory_data_param()->set_channels(inputc.channels());
		net_param.mutable_layer(0)->mutable_memory_data_param()->set_width(width);
		net_param.mutable_layer(0)->mutable_memory_data_param()->set_height(height);
	      }
	    if (net_param.mutable_layer(1)->has_memory_data_param())
	      {
		net_param.mutable_layer(1)->mutable_memory_data_param()->set_channels(inputc.channels()); // test layer
		net_param.mutable_layer(1)->mutable_memory_data_param()->set_width(width);
		net_param.mutable_layer(1)->mutable_memory_data_param()->set_height(height);
	      }
	  }
	else
	  {
	    if (net_param.mutable_layer(0)->has_memory_data_param())
	      net_param.mutable_layer(0)->mutable_memory_data_param()->set_channels(inputc.channels()+_ntargets);
	    if (net_param.mutable_layer(1)->has_memory_data_param())
	      net_param.mutable_layer(1)->mutable_memory_data_param()->set_channels(inputc.channels()+_ntargets);
	    net_param.mutable_layer(2)->mutable_slice_param()->set_slice_point(0,inputc.channels());
	  }
      }
    
    if (net_param.mutable_layer(3)->type() == "Embed") // embed is preceded by a flatten layer
      {
	has_embed = true;
	net_param.mutable_layer(3)->mutable_embed_param()->set_input_dim(inputc._max_embed_id);
      }

    if (net_param.mutable_layer(4)->type() == "Reshape" && has_embed)
      {
	net_param.mutable_layer(4)->mutable_reshape_param()->mutable_shape()->set_dim(2,inputc._sequence_txt);
      }
    
    // if autoencoder, set the last inner product layer output number to input size (i.e. inputc.channels())
    if (_autoencoder && typeid(this->_inputc) == typeid(CSVCaffeInputFileConn))
      {
	int k = net_param.layer_size();
	std::string bottom;
	for (int l=k-1;l>0;l--)
	  {
	    caffe::LayerParameter *lparam = net_param.mutable_layer(l);
	    if (lparam->type() == "SigmoidCrossEntropyLoss")
	      {
		bottom = lparam->bottom(0);
	      }
	    if (!bottom.empty() && lparam->type() == "InnerProduct")
	      {
		lparam->mutable_inner_product_param()->set_num_output(inputc.channels());
		break;
	      }
	  }
      }

    if (has_class_weights)
      {
	int k = net_param.layer_size();
	for (int l=k-1;l>0;l--)
	  {
	    caffe::LayerParameter *lparam = net_param.mutable_layer(l);
	    if (lparam->type() == "SoftmaxWithLoss")
	      {
		lparam->set_type("SoftmaxWithInfogainLoss");
		lparam->mutable_infogain_loss_param()->set_source(this->_mlmodel._repo + "/class_weights.binaryproto");
		break;
	      }
           else if (lparam->type().compare("DiceCoefLoss") == 0)
             {
               lparam->mutable_dice_coef_loss_param()->set_weights(this->_mlmodel._repo + "/class_weights.binaryproto");
               break;
             }
	  }
      }

    if (ignore_label >= 0)
      {
	int k = net_param.layer_size();
	for (int l=k-1;l>0;l--)
	  {
	    caffe::LayerParameter *lparam = net_param.mutable_layer(l);
	    if (lparam->type() == "SoftmaxWithLoss" || lparam->type() == "SoftmaxWithInfogainLoss")
	      {
		lparam->mutable_loss_param()->set_ignore_label(ignore_label);
	      }
	  }
      }

    caffe::NetParameter deploy_net_param;
    caffe::ReadProtoFromTextFile(deploy_file,&deploy_net_param);    

    if (this->_inputc._ctc) // crnn
      {
	int k = deploy_net_param.layer_size();
	for (int l=0;l<k;l++)
	  {
	    caffe::LayerParameter *lparam = deploy_net_param.mutable_layer(l);
	    if (lparam->type() == "Reshape")
	      {
		lparam->mutable_reshape_param()->mutable_shape()->set_dim(2,timesteps);
	      }
	    else if (lparam->type() == "ContinuationIndicator")
	      {
		lparam->mutable_continuation_indicator_param()->set_time_step(timesteps);
	      }
	    else if (lparam->type() == "InnerProduct") // last layer before ctcloss
	      {
		lparam->mutable_inner_product_param()->set_num_output(inputc._alphabet_size);
	      }
	  }
      }
    
    if (deploy_net_param.mutable_layer(2)->type() == "Embed")
      {
	deploy_net_param.mutable_layer(2)->mutable_embed_param()->set_input_dim(inputc._max_embed_id);
      }
    if (deploy_net_param.mutable_layer(3)->type() == "Reshape" && has_embed)
      {
	deploy_net_param.mutable_layer(3)->mutable_reshape_param()->mutable_shape()->set_dim(2,inputc._sequence_txt);
      }
    
    if (_autoencoder && typeid(this->_inputc) == typeid(CSVCaffeInputFileConn))
      {
	int k = deploy_net_param.layer_size();
	std::string bottom = "";
	for (int l=k-1;l>0;l--)
	  {
	    caffe::LayerParameter *lparam = deploy_net_param.mutable_layer(l);
	    if (lparam->type() == "InnerProduct")
	      {
		lparam->mutable_inner_product_param()->set_num_output(inputc.channels());
		break;
	      }
	  }
      }

    if (deploy_net_param.mutable_layer(0)->has_memory_data_param())
      {
	// no batch size set on deploy model since it is adjusted for every prediction batch
        if (_ntargets == 0 || _ntargets == 1  || (typeid(this->_inputc) == typeid(CSVTSCaffeInputFileConn)))
	  {
	    deploy_net_param.mutable_layer(0)->mutable_memory_data_param()->set_channels(inputc.channels());
	    deploy_net_param.mutable_layer(0)->mutable_memory_data_param()->set_width(width);
	    deploy_net_param.mutable_layer(0)->mutable_memory_data_param()->set_height(height);
	  }
	else
	  {
	    deploy_net_param.mutable_layer(0)->mutable_memory_data_param()->set_channels(inputc.channels()+_ntargets);
	    deploy_net_param.mutable_layer(1)->mutable_slice_param()->set_slice_point(0,inputc.channels());
	  }
	if (!_autoencoder)
	  for (size_t d=0;d<inputc._mean_values.size();d++)
	    deploy_net_param.mutable_layer(0)->mutable_transform_param()->add_mean_value(inputc._mean_values.at(d));
	if (_scale != 1.0)
	  deploy_net_param.mutable_layer(0)->mutable_transform_param()->set_scale(_scale);
	if (_crop_size > 0)
	  deploy_net_param.mutable_layer(0)->mutable_transform_param()->set_crop_size(_crop_size);
      }

    caffe::WriteProtoToTextFile(net_param,net_file);
    caffe::WriteProtoToTextFile(deploy_net_param,deploy_file);
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_protofile_classes(caffe::NetParameter &net_param)
  {
    // fix class numbers
    // this procedure looks for the first bottom layer with a 'num_output' field and
    // set it to the number of classes defined by the supervised service.
    if (this->_inputc._bbox)
      return;
    bool last_layer = true;
    for (int l=net_param.layer_size()-1;l>0;l--)
      {
	caffe::LayerParameter *lparam = net_param.mutable_layer(l);
	if (lparam->type() == "Convolution" || lparam->type() == "Deconvolution")
	  {
	    if (lparam->has_convolution_param())
	      {
            int num_output = lparam->mutable_convolution_param()->num_output();
            if (_loss == "dice" && _nclasses <= 2 && lparam->name() == "linear_aggregation" && num_output == 1)
              continue;
            if (last_layer || num_output == 0)
              lparam->mutable_convolution_param()->set_num_output(_nclasses);
            if (last_layer && num_output != 0)
              break;
            else last_layer = false;
	      }
	  }
	else if (lparam->type() == "InnerProduct" || lparam->type() == "SparseInnerProduct")
	  {
	    if (lparam->has_inner_product_param())
	      {
            if (!_regression || _ntargets == 0)
                lparam->mutable_inner_product_param()->set_num_output(_nclasses);
            else lparam->mutable_inner_product_param()->set_num_output(_ntargets);
            break;
	      }
	  }
	/*else if (lparam->type() == "DetectionOutput")
	  {
	    if (lparam->has_detection_output_param())
	      {
		lparam->mutable_detection_output_param()->set_num_classes(_nclasses);
		break;
		}
	      }*/
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_protofile_finetune(caffe::NetParameter &net_param)
  {
    std::string ft_lname, ft_oldname;

    if (net_param.name() == "deeplab_vgg16")
      // special case with 4 ends to change
      {
        for (int l=net_param.layer_size()-1;l>0;l--)
          {
            caffe::LayerParameter *lparam = net_param.mutable_layer(l);
            if (lparam->type() == "Convolution" &&
                (lparam->name() == "fc8_vgg16_1" ||
                 lparam->name() == "fc8_vgg16_2" ||
                 lparam->name() == "fc8_vgg16_3" ||
                 lparam->name() == "fc8_vgg16_4" ))
              {
                ft_oldname = lparam->top(0);
                ft_lname = lparam->name() + "_ftune";
                lparam->set_name(ft_lname);
                lparam->set_top(0,ft_lname);
                continue;
              }
            if (lparam->name() == "fc8_vgg16")
              {
                ft_lname = lparam->name() + "_ftune";
                lparam->set_name(ft_lname);
                lparam->set_top(0,ft_lname);
                for (int i=0; i<4; ++i) {
                  std::string ft_lname_b = lparam->bottom(i) + "_ftune";
                  lparam->set_bottom(i,ft_lname_b);
                }
                continue;
              }
            if (lparam->name() == "normalization" && lparam->bottom(0) == "fc8_vgg16")
              {
                lparam->set_bottom(0, lparam->bottom(0) + "_ftune");
                continue;
              }
          }
        return;
      }
    // fix class numbers
    // this procedure looks for the first bottom layer with a 'num_output' field and
    // rename the layer so that its weights can be reinitialized and the net finetuned
    int k = net_param.layer_size();
    for (int l=net_param.layer_size()-1;l>0;l--)
      {
	caffe::LayerParameter *lparam = net_param.mutable_layer(l);
	if (lparam->type() == "Convolution")
	  {
	    ft_oldname = lparam->top(0);
	    ft_lname = lparam->name() + "_ftune";
	    lparam->set_name(ft_lname);
	    lparam->set_top(0,ft_lname);
	    k = l;
	    break;
	  }
	else if (lparam->type() == "InnerProduct")
	  {
	    ft_oldname = lparam->top(0);
	    ft_lname = lparam->name() + "_ftune";
	    lparam->set_name(ft_lname);
	    lparam->set_top(0,ft_lname);
	    k = l;
	    break;
	  }
      }
    // update relations from other layers
    for (int l=net_param.layer_size()-1;l>k;l--)
      {
	caffe::LayerParameter *lparam = net_param.mutable_layer(l);
	if (lparam->top(0) == ft_oldname)
	  lparam->set_top(0,ft_lname);
	if (lparam->bottom(0) == ft_oldname)
	  lparam->set_bottom(0,ft_lname);
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::fix_batch_size(const APIData &ad,
											   const TInputConnectorStrategy &inputc,
											   int &user_batch_size,
											   int &batch_size,
											   int &test_batch_size,
											   int &test_iter)
  {
    // acquire custom batch size if any
    APIData ad_net = ad.getobj("parameters").getobj("mllib").getobj("net");
    if (ad_net.has("batch_size"))
      {
	// adjust batch size so that it is a multiple of the number of training samples (Caffe requirement)
	user_batch_size = batch_size = test_batch_size = ad_net.get("batch_size").get<int>();
	if (ad_net.has("test_batch_size"))
	  test_batch_size = ad_net.get("test_batch_size").get<int>();
	if (batch_size == 0)
	  throw MLLibBadParamException("batch size set to zero");
	this->_logger->info("user batch_size={} / inputc batch_size=",batch_size,inputc.batch_size());

	// code below is required when Caffe (weirdly) requires the batch size 
	// to be a multiple of the training dataset size.
	if (!inputc._ctc && !inputc._segmentation && !(!inputc._db && typeid(inputc) == typeid(ImgCaffeInputFileConn)))
	  {
	    if (batch_size < inputc.batch_size())
	      {
		int min_batch_size = 0;
		for (int i=batch_size;i>=1;i--)
		  if (inputc.batch_size() % i == 0)
		    {
		      min_batch_size = i;
		      break;
		    }
		int max_batch_size = 0;
		for (int i=batch_size;i<inputc.batch_size();i++)
		  {
		    if (inputc.batch_size() % i == 0)
		      {
			max_batch_size = i;
			break;
		      }
		  }
		if (std::abs(batch_size-min_batch_size) < std::abs(max_batch_size-batch_size))
		  batch_size = min_batch_size;
		else batch_size = max_batch_size;
		for (int i=test_batch_size;i>1;i--)
		  if (inputc.test_batch_size() % i == 0)
		    {
		      test_batch_size = i;
		      break;
		    }
		test_iter = inputc.test_batch_size() / test_batch_size;
	      }
	    else batch_size = inputc.batch_size();
	    test_iter = inputc.test_batch_size() / test_batch_size;
	  }
	else
	  {
	    batch_size = user_batch_size;
	    test_iter = std::max(inputc.test_batch_size() / test_batch_size,1);
	  }
	    
	//debug
	this->_logger->info("batch_size={} / test_batch_size={} / test_iter={}", batch_size, test_batch_size, test_iter);
	//debug

	if (batch_size == 0)
	  throw MLLibBadParamException("auto batch size set to zero: MemoryData input requires batch size to be a multiple of training set");
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::model_complexity(long int &flops,
                                                                                             long int &params)
  {
    for (size_t l=0;l<_net->layers().size();l++)
      {
        const boost::shared_ptr<caffe::Layer<float>> &layer = _net->layers().at(l);
        std::string lname = layer->layer_param().name();
        std::string ltype = layer->layer_param().type();
        std::vector<boost::shared_ptr<Blob<float>>> blblobs = layer->blobs();
        const std::vector<caffe::Blob<float>*> &tlblobs = _net->top_vecs().at(l);
        if (blblobs.empty())
          continue;
        long int lcount = blblobs.at(0)->count();
        long int lflops = 0;
        if (ltype == "Convolution")
          {
            int dwidth = tlblobs.at(0)->width();
            int dheight = tlblobs.at(0)->height();
            lflops = lcount * dwidth * dheight;
          }
        else
          {
            lflops = lcount;
          }
        flops += lflops;
        params += lcount;
      }
    this->_logger->info("Net total flops={} / total params={}",flops,params);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::model_type(caffe::Net<float> *net,
										       std::string &mltype)
  {
    // XXX: using deploy.prototxt to detect network's task type.
    if (_autoencoder)
      {
	mltype = "autoencoder";
	return;
      }
    bool has_deconv = false;
    for (size_t l=0;l<net->layers().size();l++)
      {
	const boost::shared_ptr<caffe::Layer<float>> &layer = net->layers().at(l);
	std::string lname = layer->layer_param().name();
	std::string ltype = layer->layer_param().type();

	if (ltype == "DetectionOutput" || ltype == "MultiBoxLoss" || ltype == "PriorBox")
	  {
	    mltype = "detection";
	    const boost::shared_ptr<caffe::Layer<float>> &final_layer = net->layers().at(net->layers().size()-1);
	    if (final_layer->layer_param().type() == "Slice")
	      mltype = "rois";
	    break;
	  }
	if (ltype == "ContinuationIndicator") // XXX: CTC layer does not appear in deploy file, this is a hack used by our LSTMs
	  {
	    mltype = "ctc";
	    break;
	  }
	if (ltype == "Interp" || ltype == "Deconvolution") // XXX: using interpolation and deconvolution as proxy to segmentation
	  {
	    mltype = "segmentation";
	    has_deconv = true;
	    // we don't break since some detection tasks may use deconvolutions
	  }
	if (!has_deconv && ltype == "Sigmoid" && l == net->layers().size()-1)
	  {
	    mltype = "regression";
	    break;
	  }
	
      }
    if (mltype.empty())
      mltype = "classification";
    this->_logger->info("detected network type is {}",mltype);
  }
  

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_protofiles_dice_deeplab_vgg16(caffe::NetParameter &net_param,caffe::NetParameter &deploy_net_param, const APIData& ad)
  {
    caffe::LayerParameter *shrink_param = find_layer_by_name(net_param,"label_shrink");
    shrink_param->add_include();
    caffe::NetStateRule *nsr = shrink_param->mutable_include(0);
    nsr->set_phase(caffe::TRAIN);
    caffe::InterpParameter *ip = shrink_param->mutable_interp_param();
    ip->set_mode(caffe::InterpParameter::NEAREST);


    int softml_pos = find_index_layer_by_type(net_param,"SoftmaxWithLoss");
    std::string logits = net_param.layer(softml_pos).bottom(0);
    std::string loss_input = logits + "_prob";

    std::string logits_norm = logits + "_norm";
    std::string agg_output = logits_norm + "_agg";

    caffe::LayerParameter* mvn_param = insert_layer_before(net_param, softml_pos++);
    *mvn_param->mutable_name() = "normalization";
    *mvn_param->mutable_type() = "MVN";
    mvn_param->mutable_mvn_param()->set_across_channels(true);
    mvn_param->add_bottom(logits);
    mvn_param->add_top(logits_norm);

    if (_nclasses <= 2)
      {
        caffe::LayerParameter* conv_param = insert_layer_before(net_param, softml_pos++);
        *conv_param->mutable_name() = "linear_aggregation";
        *conv_param->mutable_type() = "Convolution";
        conv_param->add_bottom(logits_norm);
        conv_param->add_top(agg_output);
        caffe::ConvolutionParameter* conv_param_param = conv_param->mutable_convolution_param();
        conv_param_param->set_num_output(1);
        conv_param_param->set_axis(1);
        conv_param_param->add_kernel_size(1);
        caffe::FillerParameter* filler_param = conv_param_param->mutable_weight_filler();
        filler_param->set_std(0.5);
        filler_param->set_type("gaussian");

        caffe::LayerParameter* sigmoid_loss_param = insert_layer_before(net_param, softml_pos++);
        sigmoid_loss_param->set_name("probabilizer");
        sigmoid_loss_param->set_type("Sigmoid");
        sigmoid_loss_param->add_bottom(agg_output);
        sigmoid_loss_param->add_top(loss_input);
        sigmoid_loss_param->add_include();
        nsr = sigmoid_loss_param->mutable_include(0);
        nsr->set_phase(caffe::TRAIN);
      }
    else
      {
        caffe::LayerParameter* sigmoid_loss_param = insert_layer_before(net_param, softml_pos++);
        sigmoid_loss_param->set_name("probabilizer");
        sigmoid_loss_param->set_type("Softmax");
        sigmoid_loss_param->add_bottom(logits_norm);
        sigmoid_loss_param->add_top(loss_input);
        sigmoid_loss_param->add_include();
        nsr = sigmoid_loss_param->mutable_include(0);
        nsr->set_phase(caffe::TRAIN);
      }


    caffe::LayerParameter* final_interp_param = find_layer_by_name(net_param,"final_interp");
    if (_nclasses <= 2)
      {
        *final_interp_param->mutable_bottom(0) = agg_output;
        caffe::LayerParameter* probt_param = find_layer_by_name(net_param, "probt");
        *probt_param->mutable_type() = "Sigmoid";
      }
    else
      {
        *final_interp_param->mutable_bottom(0) = logits_norm;
        caffe::LayerParameter* probt_param = find_layer_by_name(net_param, "probt");
        *probt_param->mutable_type() = "Softmax";
      }



    caffe::LayerParameter *lossparam = net_param.mutable_layer(softml_pos);
    *lossparam->mutable_type() = "DiceCoefLoss";
    *lossparam->mutable_bottom(0) = loss_input;
    caffe::DiceCoefLossParameter* dclp = lossparam->mutable_dice_coef_loss_param();

    update_protofiles_dice_params(dclp, ad);

    // now work on deploy.txt
    int final_interp_pos = find_index_layer_by_type(deploy_net_param, "Interp");
    final_interp_param = deploy_net_param.mutable_layer(final_interp_pos);
    if (_nclasses <= 2)
      *final_interp_param->mutable_bottom(0) = agg_output;
    else
      *final_interp_param->mutable_bottom(0) = logits_norm;

    mvn_param = insert_layer_before(deploy_net_param, final_interp_pos++);
    *mvn_param->mutable_name() = "normalization";
    *mvn_param->mutable_type() = "MVN";
    mvn_param->mutable_mvn_param()->set_across_channels(true);
    mvn_param->add_bottom(logits);
    mvn_param->add_top(logits_norm);


    if (_nclasses <= 2)
      {
        caffe::LayerParameter *conv_param = insert_layer_before(deploy_net_param, final_interp_pos);
        *conv_param->mutable_name() = "linear_aggregation";
        *conv_param->mutable_type() = "Convolution";
        conv_param->add_bottom(logits_norm);
        conv_param->add_top(agg_output);
        caffe::ConvolutionParameter*conv_param_param = conv_param->mutable_convolution_param();
        conv_param_param->set_num_output(1);
        conv_param_param->set_axis(1);
        conv_param_param->add_kernel_size(1);
        caffe::FillerParameter*filler_param = conv_param_param->mutable_weight_filler();
        filler_param->set_std(0.5);
        filler_param->set_type("gaussian");
        caffe::LayerParameter *probt_param = find_layer_by_name(deploy_net_param, "probt");
        *probt_param->mutable_type() = "Sigmoid";
      }
    else
      {
        caffe::LayerParameter *probt_param = find_layer_by_name(deploy_net_param, "probt");
        *probt_param->mutable_type() = "Softmax";
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_protofiles_one_hot(caffe::NetParameter &net_param)
  {
    caffe::LayerParameter* denseImageDataLayer = find_layer_by_type(net_param,"DenseImageData");
    caffe::DenseImageDataParameter *dp = denseImageDataLayer->mutable_dense_image_data_param();
    dp->set_one_hot_nclasses(_nclasses);
  }


    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_protofiles_dice_params(caffe::DiceCoefLossParameter*dclp,  const APIData &ad)
  {

    caffe::DiceCoefLossParameter::GeneralizationMode genMode =
      caffe::DiceCoefLossParameter::MULTICLASS;
    caffe::DiceCoefLossParameter::WeightMode weightMode =
      caffe::DiceCoefLossParameter::INVERSE_VOLUME;
    caffe::DiceCoefLossParameter::ContourShape contours =
      caffe::DiceCoefLossParameter::NO;
    int contour_size = 7;
    float contour_amplitude = 5.0;
    if (ad.has("dice_param"))
      {
        APIData diceParams = ad.getobj("dice_param");
        if (diceParams.has("class_weighting"))
          {
            APIData cwd = diceParams.getobj("class_weighting");
            if (cwd.has("compute_on"))
              {
                std::string wm = cwd.get("compute_on").get<std::string>();
                if (wm == "image")
                  genMode = caffe::DiceCoefLossParameter::MULTICLASS_WEIGHTED;
                else if (wm == "batch")
                  genMode = caffe::DiceCoefLossParameter::MULTICLASS_WEIGHTED_BATCH;
                else if (wm == "all")
                  genMode = caffe::DiceCoefLossParameter::MULTICLASS_WEIGHTED_ALL;
                else
                  {
                    this->_logger->warn("dice class weighting on {} unrecognized , setting to all",wm);
                    genMode = caffe::DiceCoefLossParameter::MULTICLASS_WEIGHTED_ALL;
                  }
              }
            if (cwd.has("weight"))
              {
                std::string wm = cwd.get("weight").get<std::string>();
                if (wm == "equalize_classes")
                  weightMode = caffe::DiceCoefLossParameter::EQUALIZE_CLASSES;
                else if (wm == "extra_small_volumes")
                  weightMode = caffe::DiceCoefLossParameter::EXTRA_SMALL_VOLUMES;
                else if (wm == "inverse_volume")
                  weightMode = caffe::DiceCoefLossParameter::INVERSE_VOLUME;
                else
                  {
                    this->_logger->warn("dice weight mode {} unrecognized , setting to inverse volume",wm);
                    weightMode = caffe::DiceCoefLossParameter::INVERSE_VOLUME;
                  }
              }
          }


        if (diceParams.has("contour"))
          {
            APIData contourdata = diceParams.getobj("contour");

            if (contourdata.has("shape"))
              {
                std::string cs = contourdata.get("shape").get<std::string>();
                if (cs == "simple")
                  contours = caffe::DiceCoefLossParameter::SIMPLE;
                else if (cs == "sharp")
                  contours = caffe::DiceCoefLossParameter::SHARP;
                else
                  {
                    this->_logger->warn("dice contour shape {} unrecognized , setting to sharp",cs);
                    contours = caffe::DiceCoefLossParameter::SHARP;
                  }
              }
            if (contourdata.has("size"))
              {
                try
                  {
                    contour_size = contourdata.get("size").get<int>();
                  }
                catch (std::exception &e)
                  {
                    this->_logger->warn("dice contour size unrecognized, (odd) int expected");
                  }

              }
            if (contourdata.has("amplitude"))
              {
                try
                  {
                    contour_amplitude = contourdata.get("amplitude").get<double>();
                  }
                catch (std::exception &e)
                  {
                    this->_logger->warn("dice contour amplitude unrecognized, float expected");
                  }
              }
          }
      }


    if (_nclasses <= 2)
      dclp->set_generalization(caffe::DiceCoefLossParameter::NONE);
    else
      {
        dclp->set_generalization(genMode);
        if (genMode != caffe::DiceCoefLossParameter::MULTICLASS)
          dclp->set_weight_mode(weightMode);
      }

    dclp->set_contour_shape(contours);
    if (contours != caffe::DiceCoefLossParameter::NO)
      {
        dclp->set_contour_size(contour_size);
        dclp->set_contour_amplitude(contour_amplitude);
      }
  }


  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_protofiles_dice_unet(caffe::NetParameter &net_param,caffe::NetParameter &deploy_net_param,  const APIData&ad)
  {
    int softml_pos = find_index_layer_by_type(net_param,"SoftmaxWithLoss");

    std::string logits = net_param.layer(softml_pos).bottom(0);
    std::string loss_input = logits +  "_prob";
    std::string logits_norm  = logits + "_norm";
    std::string agg_output = logits + "_agg";

    caffe::LayerParameter* mvn_param = insert_layer_before(net_param, softml_pos++);
    *mvn_param->mutable_name() = "normalization";
    *mvn_param->mutable_type() = "MVN";
    mvn_param->mutable_mvn_param()->set_across_channels(true);
    mvn_param->add_bottom(logits);
    mvn_param->add_top(logits_norm);


    if (_nclasses <= 2)
      {
        caffe::LayerParameter* conv_param = insert_layer_before(net_param, softml_pos++);
        *conv_param->mutable_name() = "linear_aggregation";
        *conv_param->mutable_type() = "Convolution";
        conv_param->add_bottom(logits_norm);
        conv_param->add_top(agg_output);
        caffe::ConvolutionParameter* conv_param_param = conv_param->mutable_convolution_param();
        conv_param_param->set_num_output(1);
        conv_param_param->set_axis(1);
        conv_param_param->add_kernel_size(1);
        caffe::FillerParameter* filler_param = conv_param_param->mutable_weight_filler();
        filler_param->set_std(0.5);
        filler_param->set_type("gaussian");

        caffe::LayerParameter* sigmoid_loss_param = insert_layer_before(net_param, softml_pos++);
        sigmoid_loss_param->set_name("probabilizer");
        sigmoid_loss_param->set_type("Sigmoid");
        sigmoid_loss_param->add_bottom(agg_output);
        sigmoid_loss_param->add_top(loss_input);
        sigmoid_loss_param->add_include();
        caffe::NetStateRule *nsr = sigmoid_loss_param->mutable_include(0);
        nsr->set_phase(caffe::TRAIN);

        caffe::LayerParameter* probt_param = find_layer_by_name(net_param, "probt");
        *probt_param->mutable_type() = "Sigmoid";
        *probt_param->mutable_bottom(0) = agg_output;
      }
    else
      {
        caffe::LayerParameter* softmax_param = insert_layer_before(net_param, softml_pos++);
        softmax_param->set_name("probabilizer_multi");
        softmax_param->set_type("Softmax");
        softmax_param->add_bottom(logits_norm);
        softmax_param->add_top(loss_input);
        softmax_param->add_include();
        caffe::NetStateRule *nsr = softmax_param->mutable_include(0);
        nsr->set_phase(caffe::TRAIN);

        caffe::LayerParameter* probt_param = find_layer_by_name(net_param, "probt");
        *probt_param->mutable_bottom(0) = logits_norm;
      }

    caffe::LayerParameter *lossparam = net_param.mutable_layer(softml_pos);
    lossparam->clear_softmax_param();
    *lossparam->mutable_type() = "DiceCoefLoss";
    *lossparam->mutable_bottom(0) = loss_input;
    caffe::DiceCoefLossParameter* dclp = lossparam->mutable_dice_coef_loss_param();

    update_protofiles_dice_params(dclp, ad);

    // BELOW DEPLOY
    int final_pred = find_index_layer_by_name(deploy_net_param, "pred");

    mvn_param = insert_layer_before(deploy_net_param, final_pred++);
    *mvn_param->mutable_name() = "normalization";
    *mvn_param->mutable_type() = "MVN";
    mvn_param->mutable_mvn_param()->set_across_channels(true);
    mvn_param->add_bottom(logits);
    mvn_param->add_top(logits_norm);

    if (_nclasses <= 2)
      {
        caffe::LayerParameter* conv_param = insert_layer_before(deploy_net_param, final_pred);
        *conv_param->mutable_name() = "linear_aggregation";
        *conv_param->mutable_type() = "Convolution";
        conv_param->add_bottom(logits_norm);
        conv_param->add_top(agg_output);
        caffe::ConvolutionParameter*conv_param_param = conv_param->mutable_convolution_param();
        conv_param_param->set_num_output(1);
        conv_param_param->set_axis(1);
        conv_param_param->add_kernel_size(1);
        caffe::FillerParameter*filler_param = conv_param_param->mutable_weight_filler();
        filler_param->set_std(0.5);
        filler_param->set_type("gaussian");

        caffe::LayerParameter* probt_param = find_layer_by_name(deploy_net_param, "pred");
        *probt_param->mutable_type() = "Sigmoid";
        *probt_param->mutable_bottom(0) = agg_output;
      }
    else
      {
        caffe::LayerParameter* probt_param = find_layer_by_name(deploy_net_param, "pred");
        *probt_param->mutable_bottom(0) = logits_norm;
      }
  }


  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_protofile_imageDataLayer(caffe::NetParameter &net_param)
  {
    caffe::LayerParameter *lparam = net_param.mutable_layer(0);
    caffe::ImageDataParameter* image_data_parameter = lparam->mutable_image_data_param();
    lparam->set_type("ImageData");
    image_data_parameter->set_source(this->_inputc._root_folder);
    image_data_parameter->set_batch_size(this->_inputc.batch_size());
    image_data_parameter->set_shuffle(this->_inputc._shuffle);
    image_data_parameter->set_new_height(this->_inputc.height());
    image_data_parameter->set_new_width(this->_inputc.width());
    if (this->_inputc._multi_label)
      image_data_parameter->set_label_size(_nclasses);
    else
      image_data_parameter->set_label_size(1);
    if (this->_inputc._has_mean_file)
      image_data_parameter->set_mean_file("mean.binaryproto");
    lparam->clear_data_param();
    lparam->clear_transform_param();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  bool CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::is_better(double v1, double v2, std::string metric_name)
  {
    if (metric_name == "eucll" || metric_name == "delta_score_0.1" || metric_name == "L1_mean_error")
      return (v2 > v1);
    return (v1 > v2);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::save_if_best(APIData &meas_out, boost::shared_ptr<caffe::Solver<float>>solver, bool already_snapshoted)
  {
    double cur_meas = std::numeric_limits<double>::infinity();
    std::string meas;
    for (auto m: _best_metrics)
      {
        if (meas_out.has(m))
          {
            cur_meas = meas_out.get(m).get<double>();
            meas = m;
            break;
          }
      }
    if (cur_meas == std::numeric_limits<double>::infinity())
      {
        // could not find value for measuring best
        this->_logger->info("could not find any value for measuring best model");
        return;
      }
    if (_best_metric_value == std::numeric_limits<double>::infinity() ||
        is_better(cur_meas, _best_metric_value, meas))
      {
        _best_metric_value = cur_meas;
        if (!already_snapshoted)
          solver->Snapshot();
        try
          {
            std::ofstream bestfile;
            std::string bestfilename = this->_mlmodel._repo + this->_mlmodel._best_model_filename;
            bestfile.open(bestfilename,std::ios::out);
            bestfile << "iteration:" <<  solver->iter_ << std::endl;
            bestfile <<  meas << ":" << cur_meas << std::endl;
            bestfile.close();
          }
        catch(std::exception &e)
          {
            this->_logger->error("could not write best model file");
          }
      }

  }


  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::findOutputSlotNumberByBlobName(const caffe::Net<float> *net,
                                     const std::string blob_name)
  {
    const std::vector<std::string> blob_names = net->blob_names();
    const std::vector<int> output_blob_indices = net->output_blob_indices();
    for (int i =0; i<net->num_outputs(); ++i)
      {
        if (blob_names[output_blob_indices[i]] == blob_name)
          return i;
      }
    return -1;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  boost::shared_ptr<Blob<float>> CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::findOutputBlobByName(const caffe::Net<float> *net, const std::string blob_name)
  {
    const std::vector<std::string> blob_names = net->blob_names();
    const std::vector<int> output_blob_indices = net->output_blob_indices();
    for (int i =0; i<net->num_outputs(); ++i)
      {
        if (blob_names[output_blob_indices[i]] == blob_name)
          net->output_blobs().at(i);
      }
    return nullptr;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  std::vector<double> CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::img_resize(const std::vector<double>& vals, const int height_net, const int width_net, const int height_dest, const int width_dest, bool resize_nn)
  {
    cv::Mat segimg = cv::Mat(height_net,width_net, CV_64FC1);
    std::memcpy(segimg.data,vals.data(),vals.size()*sizeof(double));
    cv::Mat segimg_res;
    if (resize_nn)
      cv::resize(segimg,segimg_res,cv::Size(width_dest,height_dest),0,0,cv::INTER_NEAREST);
    else
      cv::resize(segimg,segimg_res,cv::Size(width_dest,height_dest),0,0,cv::INTER_LINEAR);
    return std::vector<double>((double*)segimg_res.data,(double*)segimg_res.data+segimg_res.rows*segimg_res.cols);
  }

  template class CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>;
  template class CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>;
  template class CaffeLib<CSVTSCaffeInputFileConn,SupervisedOutput,CaffeModel>;
  template class CaffeLib<TxtCaffeInputFileConn,SupervisedOutput,CaffeModel>;
  template class CaffeLib<SVMCaffeInputFileConn,SupervisedOutput,CaffeModel>;
  template class CaffeLib<ImgCaffeInputFileConn,UnsupervisedOutput,CaffeModel>;
  template class CaffeLib<CSVCaffeInputFileConn,UnsupervisedOutput,CaffeModel>;
  template class CaffeLib<CSVTSCaffeInputFileConn,UnsupervisedOutput,CaffeModel>;
  template class CaffeLib<TxtCaffeInputFileConn,UnsupervisedOutput,CaffeModel>;
  template class CaffeLib<SVMCaffeInputFileConn,UnsupervisedOutput,CaffeModel>;
}
