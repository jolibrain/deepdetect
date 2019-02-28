/**
 * DeepDetect
 * Copyright (c) 2015 Emmanuel Benazera
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
#include "caffeinputconns.h"
#include "outputconnectorstrategy.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace dd;

static std::string net_file = "../templates/caffe/mlp/mlp.prototxt";
static std::string dnet_file = "../templates/caffe/mlp/deploy.prototxt";
static std::string onet_file = "nmlp.prototxt";
static std::string donet_file = "nmlp_deploy.prototxt";

static std::string convnet_file = "../templates/caffe/convnet/convnet.prototxt";
static std::string dconvnet_file = "../templates/caffe/convnet/deploy.prototxt";
static std::string oconvnet_file = "nconvnet.prototxt";
static std::string doconvnet_file = "nconvnet_deploy.prototxt";
static std::string oresnet_file = "nresnet.prototxt";
static std::string doresnet_file = "nresnet_deploy.prototxt";

TEST(caffelib,configure_mlp_template_1_nt)
{
  int nclasses = 7;
  caffe::NetParameter net_param, deploy_net_param;

  std::vector<int> layers = {200};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.6);
  ad.add("nclasses",nclasses);
  
  CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel());
  caff->configure_mlp_template(ad,CSVCaffeInputFileConn(),net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,onet_file.c_str());
  caffe::WriteProtoToTextFile(deploy_net_param,donet_file);
  
  ASSERT_EQ(8,net_param.layer_size());
  caffe::LayerParameter *lparam = net_param.mutable_layer(2);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(layers.at(0),lparam->mutable_inner_product_param()->num_output());
  lparam = net_param.mutable_layer(3);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(4);
  ASSERT_EQ("Dropout",lparam->type());
  ASSERT_NEAR(0.6,lparam->mutable_dropout_param()->dropout_ratio(),1e-5); // near as there seems to be a slight conversion issue from protobufs
  lparam = net_param.mutable_layer(5);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(nclasses,lparam->mutable_inner_product_param()->num_output());
  
  ASSERT_EQ(5,deploy_net_param.layer_size());
  caffe::LayerParameter *dlparam = deploy_net_param.mutable_layer(1);
  ASSERT_EQ("InnerProduct",dlparam->type());
  ASSERT_EQ(layers.at(0),dlparam->mutable_inner_product_param()->num_output());
  dlparam = deploy_net_param.mutable_layer(2);
  ASSERT_EQ("PReLU",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(3);
  ASSERT_EQ("InnerProduct",dlparam->type());
  ASSERT_EQ(nclasses,dlparam->mutable_inner_product_param()->num_output());

  delete caff;
}

TEST(caffelib,configure_mlp_template_1_db)
{
  int nclasses = 7;
  caffe::NetParameter net_param, deploy_net_param;
  
  std::vector<int> layers = {200};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.2);
  ad.add("db",true);
  ad.add("nclasses",nclasses);
  
  CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel());
  caff->configure_mlp_template(ad,CSVCaffeInputFileConn(),net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,onet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,donet_file);

  ASSERT_EQ(8,net_param.layer_size());
  caffe::LayerParameter *lparam = net_param.mutable_layer(0);
  ASSERT_EQ("Data",lparam->type());
  ASSERT_EQ("train.lmdb",lparam->mutable_data_param()->source());
  ASSERT_EQ(caffe::DataParameter_DB_LMDB,lparam->mutable_data_param()->backend());
  lparam = net_param.mutable_layer(2);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(layers.at(0),lparam->mutable_inner_product_param()->num_output());
  lparam = net_param.mutable_layer(3);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(4);
  ASSERT_EQ("Dropout",lparam->type());
  ASSERT_NEAR(0.2,lparam->mutable_dropout_param()->dropout_ratio(),1e-5); // near as there seems to be a slight conversion issue from protobufs
  lparam = net_param.mutable_layer(5);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(nclasses,lparam->mutable_inner_product_param()->num_output());
  
  ASSERT_EQ(5,deploy_net_param.layer_size());
  caffe::LayerParameter *dlparam = deploy_net_param.mutable_layer(1);
  ASSERT_EQ("InnerProduct",dlparam->type());
  ASSERT_EQ(layers.at(0),dlparam->mutable_inner_product_param()->num_output());
  dlparam = deploy_net_param.mutable_layer(2);
  ASSERT_EQ("PReLU",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(3);
  ASSERT_EQ("InnerProduct",dlparam->type());
  ASSERT_EQ(nclasses,dlparam->mutable_inner_product_param()->num_output());

  delete caff;
}

TEST(caffelib,configure_mlp_template_n_nt)
{
  int nclasses = 7;
  caffe::NetParameter net_param, deploy_net_param;
    
  std::vector<int> layers = {200,150,75};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.6);
  ad.add("nclasses",nclasses);

  CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel());
  caff->configure_mlp_template(ad,CSVCaffeInputFileConn(),net_param,deploy_net_param);
    
  caffe::WriteProtoToTextFile(net_param,onet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,donet_file);
  
  ASSERT_EQ(14,net_param.layer_size());
  ASSERT_EQ(9,deploy_net_param.layer_size());
  int rl = 1;
  caffe::LayerParameter *lparam;
  for (size_t l=0;l<layers.size();l++)
    {
      lparam = net_param.mutable_layer(++rl);
      ASSERT_EQ("InnerProduct",lparam->type());
      ASSERT_EQ(layers.at(l),lparam->mutable_inner_product_param()->num_output());
      lparam = net_param.mutable_layer(++rl);
      ASSERT_EQ("PReLU",lparam->type());
      lparam = net_param.mutable_layer(++rl);
      ASSERT_EQ("Dropout",lparam->type());
      ASSERT_NEAR(0.6,lparam->mutable_dropout_param()->dropout_ratio(),1e-5); // near as there seems to be a slight conversion issue from protobufs
    }
  int drl = 0;
  caffe::LayerParameter *dlparam;
  for (size_t l=0;l<layers.size();l++)
    {
      dlparam = deploy_net_param.mutable_layer(++drl);
      ASSERT_EQ("InnerProduct",dlparam->type());
      ASSERT_EQ(layers.at(l),dlparam->mutable_inner_product_param()->num_output());
      dlparam = deploy_net_param.mutable_layer(++drl);
      ASSERT_EQ("PReLU",dlparam->type());
    }
  lparam = net_param.mutable_layer(rl+1);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(nclasses,lparam->mutable_inner_product_param()->num_output());
  dlparam = deploy_net_param.mutable_layer(drl+1);
  ASSERT_EQ("InnerProduct",dlparam->type());
  ASSERT_EQ(nclasses,dlparam->mutable_inner_product_param()->num_output());
  delete caff;
}

TEST(caffelib,configure_mlp_template_n_mt)
{
  int ntargets = 3;
  caffe::NetParameter net_param, deploy_net_param;
  
  std::vector<int> layers = {200,150,75};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.6);
  ad.add("ntargets",ntargets);

  CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel());
  caff->configure_mlp_template(ad,CSVCaffeInputFileConn(),net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,onet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,donet_file);
  
  ASSERT_EQ(15,net_param.layer_size());
  ASSERT_EQ(10,deploy_net_param.layer_size());
  int rl = 1;
  caffe::LayerParameter *lparam;
  lparam = net_param.mutable_layer(++rl);
  ASSERT_EQ("Slice",lparam->type());
  ASSERT_EQ(1,lparam->mutable_slice_param()->slice_point(0)); // 1 is a dummy slice point
  for (size_t l=0;l<layers.size();l++)
    {
      lparam = net_param.mutable_layer(++rl);
      ASSERT_EQ("InnerProduct",lparam->type());
      ASSERT_EQ(layers.at(l),lparam->mutable_inner_product_param()->num_output());
      lparam = net_param.mutable_layer(++rl);
      ASSERT_EQ("PReLU",lparam->type());
      lparam = net_param.mutable_layer(++rl);
      ASSERT_EQ("Dropout",lparam->type());
      ASSERT_NEAR(0.6,lparam->mutable_dropout_param()->dropout_ratio(),1e-5); // near as there seems to be a slight conversion issue from protobufs
    }
  int drl = 0;
  caffe::LayerParameter *dlparam;
  dlparam = deploy_net_param.mutable_layer(++drl);
  ASSERT_EQ("Slice",dlparam->type());
  ASSERT_EQ(1,dlparam->mutable_slice_param()->slice_point(0)); // 1 is a dummy slice point
  for (size_t l=0;l<layers.size();l++)
    {
      dlparam = deploy_net_param.mutable_layer(++drl);
      ASSERT_EQ("InnerProduct",dlparam->type());
      ASSERT_EQ(layers.at(l),dlparam->mutable_inner_product_param()->num_output());
      dlparam = deploy_net_param.mutable_layer(++drl);
      ASSERT_EQ("PReLU",dlparam->type());
    }
  lparam = net_param.mutable_layer(rl+1);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(ntargets,lparam->mutable_inner_product_param()->num_output());
  dlparam = deploy_net_param.mutable_layer(drl+1);
  ASSERT_EQ("InnerProduct",dlparam->type());
  ASSERT_EQ(ntargets,dlparam->mutable_inner_product_param()->num_output());
  delete caff;
}

TEST(caffelib,configure_convnet_template_1)
{
  int nclasses = 18;
  caffe::NetParameter net_param, deploy_net_param;
 
  std::vector<std::string> layers = {"1CR64"};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.2);
  ad.add("nclasses",nclasses);

  CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel());
  caff->configure_convnet_template(ad,ImgCaffeInputFileConn(),net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,oconvnet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,doconvnet_file);
 
  ASSERT_EQ(8,net_param.layer_size());
  caffe::LayerParameter *lparam = net_param.mutable_layer(2);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(64,lparam->mutable_convolution_param()->num_output());
  lparam = net_param.mutable_layer(3);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(4);
  ASSERT_EQ("Pooling",lparam->type());
  lparam = net_param.mutable_layer(5);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(nclasses,lparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("ip0_conv_0",lparam->bottom(0));

  //ASSERT_EQ(6,deploy_net_param.layer_size());
  caffe::LayerParameter *dlparam = deploy_net_param.mutable_layer(1);
  ASSERT_EQ("Convolution",dlparam->type());
  ASSERT_EQ(64,dlparam->mutable_convolution_param()->num_output());
  dlparam = deploy_net_param.mutable_layer(2);
  ASSERT_EQ("PReLU",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(3);
  ASSERT_EQ("Pooling",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(4);
  ASSERT_EQ("InnerProduct",dlparam->type());
  ASSERT_EQ(nclasses,dlparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("ip0_conv_0",dlparam->bottom(0));
  delete caff;
}

TEST(caffelib,configure_convnet_template_1_db)
{
  int nclasses = 18;
  caffe::NetParameter net_param, deploy_net_param;
 
  std::vector<std::string> layers = {"1CR64"};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.2);
  ad.add("db",true);
  ad.add("nclasses",nclasses);

  CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel());
  caff->configure_convnet_template(ad,ImgCaffeInputFileConn(),net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,oconvnet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,doconvnet_file);
  
  ASSERT_EQ(8,net_param.layer_size());
  caffe::LayerParameter *lparam = net_param.mutable_layer(0);
  ASSERT_EQ("Data",lparam->type());
  ASSERT_EQ("train.lmdb",lparam->mutable_data_param()->source());
  ASSERT_EQ(caffe::DataParameter_DB_LMDB,lparam->mutable_data_param()->backend());
  lparam = net_param.mutable_layer(2);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(64,lparam->mutable_convolution_param()->num_output());
  lparam = net_param.mutable_layer(3);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(4);
  ASSERT_EQ("Pooling",lparam->type());
  lparam = net_param.mutable_layer(5);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(nclasses,lparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("ip0_conv_0",lparam->bottom(0));

  ASSERT_EQ(6,deploy_net_param.layer_size());
  caffe::LayerParameter *dlparam = deploy_net_param.mutable_layer(1);
  ASSERT_EQ("Convolution",dlparam->type());
  ASSERT_EQ(64,dlparam->mutable_convolution_param()->num_output());
  dlparam = deploy_net_param.mutable_layer(2);
  ASSERT_EQ("PReLU",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(3);
  ASSERT_EQ("Pooling",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(4);
  ASSERT_EQ("InnerProduct",dlparam->type());
  ASSERT_EQ(nclasses,dlparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("ip0_conv_0",dlparam->bottom(0));
  delete caff;
}

TEST(caffelib,configure_convnet_template_2)
{
  int nclasses = 18;
  caffe::NetParameter net_param, deploy_net_param;
  
  std::vector<std::string> layers = {"1CR64","1CR128","1000"};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.2);
  ad.add("nclasses",nclasses);

  CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel());
  caff->configure_convnet_template(ad,ImgCaffeInputFileConn(),net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,oconvnet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,doconvnet_file);
  
  ASSERT_EQ(14,net_param.layer_size());
  caffe::LayerParameter *lparam = net_param.mutable_layer(2);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(64,lparam->mutable_convolution_param()->num_output());
  lparam = net_param.mutable_layer(3);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(4);
  ASSERT_EQ("Pooling",lparam->type());
  lparam = net_param.mutable_layer(5);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(128,lparam->mutable_convolution_param()->num_output());
  lparam = net_param.mutable_layer(6);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(7);
  ASSERT_EQ("Pooling",lparam->type());
  lparam = net_param.mutable_layer(8);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(1000,lparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("ip1_conv_0",lparam->bottom(0));
  lparam = net_param.mutable_layer(9);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(10);
  ASSERT_EQ("Dropout",lparam->type());
  ASSERT_NEAR(0.2,lparam->mutable_dropout_param()->dropout_ratio(),1e-5);
  lparam = net_param.mutable_layer(11);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(nclasses,lparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("fc1000_0",lparam->bottom(0));
  ASSERT_EQ("ip_losst",lparam->top(0));

  ASSERT_EQ(11,deploy_net_param.layer_size());
  caffe::LayerParameter *dlparam = deploy_net_param.mutable_layer(1);
  ASSERT_EQ("Convolution",dlparam->type());
  ASSERT_EQ(64,dlparam->mutable_convolution_param()->num_output());
  dlparam = deploy_net_param.mutable_layer(2);
  ASSERT_EQ("PReLU",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(3);
  ASSERT_EQ("Pooling",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(4);
  ASSERT_EQ("Convolution",dlparam->type());
  ASSERT_EQ(128,dlparam->mutable_convolution_param()->num_output());
  dlparam = deploy_net_param.mutable_layer(5);
  ASSERT_EQ("PReLU",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(6);
  ASSERT_EQ("Pooling",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(7);
  ASSERT_EQ("InnerProduct",dlparam->type());
  ASSERT_EQ(1000,dlparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("ip1_conv_0",dlparam->bottom(0));
  dlparam = deploy_net_param.mutable_layer(8);
  ASSERT_EQ("PReLU",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(9);
  ASSERT_EQ("InnerProduct",dlparam->type());
  ASSERT_EQ(nclasses,dlparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("fc1000_0",dlparam->bottom(0)); 
  ASSERT_EQ("ip_loss",dlparam->top(0));
  delete caff;
}

TEST(caffelib,configure_convnet_template_3)
{
  int nclasses = 18;
  caffe::NetParameter net_param, deploy_net_param;
  
  std::vector<std::string> layers = {"2CR64"};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.2);
  ad.add("nclasses",nclasses);

  CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel());
  caff->configure_convnet_template(ad,ImgCaffeInputFileConn(),net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,oconvnet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,doconvnet_file);

  ASSERT_EQ(10,net_param.layer_size());
  caffe::LayerParameter *lparam = net_param.mutable_layer(2);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(64,lparam->mutable_convolution_param()->num_output());
  lparam = net_param.mutable_layer(3);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(4);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(64,lparam->mutable_convolution_param()->num_output());
  lparam = net_param.mutable_layer(5);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(6);
  ASSERT_EQ("Pooling",lparam->type());
  lparam = net_param.mutable_layer(7);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(nclasses,lparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("ip0_conv_1",lparam->bottom(0));
  
  ASSERT_EQ(8,deploy_net_param.layer_size());
  caffe::LayerParameter *dlparam = deploy_net_param.mutable_layer(1);
  ASSERT_EQ("Convolution",dlparam->type());
  ASSERT_EQ(64,dlparam->mutable_convolution_param()->num_output());
  dlparam = deploy_net_param.mutable_layer(2);
  ASSERT_EQ("PReLU",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(3);
  ASSERT_EQ("Convolution",dlparam->type());
  ASSERT_EQ(64,dlparam->mutable_convolution_param()->num_output());
  dlparam = deploy_net_param.mutable_layer(4);
  ASSERT_EQ("PReLU",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(5);
  ASSERT_EQ("Pooling",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(6);
  ASSERT_EQ("InnerProduct",dlparam->type());
  ASSERT_EQ(nclasses,dlparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("ip0_conv_1",dlparam->bottom(0));
  delete caff;
}

TEST(caffelib,configure_convnet_template_n)
{
  int nclasses = 18;
  caffe::NetParameter net_param, deploy_net_param;
  
  std::vector<std::string> layers = {"2CR32","2CR64","2CR128","4096","1024"};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.2);
  ad.add("nclasses",nclasses);
  ad.add("db",true);

  CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel());
  caff->configure_convnet_template(ad,ImgCaffeInputFileConn(),net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,oconvnet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,doconvnet_file);
  
  ASSERT_EQ(26,net_param.layer_size());
  caffe::LayerParameter *lparam = net_param.mutable_layer(0);
  ASSERT_EQ("Data",lparam->type());
  //ASSERT_EQ("mean.binaryproto",lparam->mutable_transform_param()->mean_file());
  ASSERT_EQ(22,deploy_net_param.layer_size());
  delete caff;
}

TEST(caffelib,configure_convnet_template_n_1D)
{
  int nclasses = 18;
  caffe::NetParameter net_param, deploy_net_param;
  
  std::vector<std::string> layers = {"1CR256","1CR256","4CR256","1024","1024"};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.2);
  ad.add("db",true);
  ad.add("nclasses",nclasses);
  
  TxtCaffeInputFileConn tcif;
  tcif._characters = true;
  tcif._flat1dconv = true;
  tcif._db = true;
  tcif.build_alphabet();
  tcif._sequence = 1014;
  CaffeLib<TxtCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<TxtCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel());
  caff->configure_convnet_template(ad,tcif,net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,oconvnet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,doconvnet_file);
  
  ASSERT_EQ(27,net_param.layer_size());
  caffe::LayerParameter *lparam = net_param.mutable_layer(0);
  ASSERT_EQ("Data",lparam->type());
  ASSERT_TRUE(lparam->mutable_transform_param()->mean_file().empty());
  lparam = net_param.mutable_layer(1);
  ASSERT_EQ("MemoryData",lparam->type());
  ASSERT_EQ(1,lparam->mutable_memory_data_param()->channels());
  ASSERT_EQ(tcif._sequence,lparam->mutable_memory_data_param()->height());
  ASSERT_EQ(tcif._alphabet.size(),lparam->mutable_memory_data_param()->width());
  lparam = net_param.mutable_layer(2);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(256,lparam->mutable_convolution_param()->num_output());
  ASSERT_EQ(7,lparam->mutable_convolution_param()->kernel_h());
  ASSERT_EQ(69,lparam->mutable_convolution_param()->kernel_w());
  lparam = net_param.mutable_layer(4);
  ASSERT_EQ("Pooling",lparam->type());
  ASSERT_EQ(3,lparam->mutable_pooling_param()->stride_h());
  ASSERT_EQ(1,lparam->mutable_pooling_param()->stride_w());
  ASSERT_EQ(3,lparam->mutable_pooling_param()->kernel_h());
  ASSERT_EQ(1,lparam->mutable_pooling_param()->kernel_w());
  lparam = net_param.mutable_layer(8);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(256,lparam->mutable_convolution_param()->num_output());
  ASSERT_EQ(3,lparam->mutable_convolution_param()->kernel_h());
  ASSERT_EQ(1,lparam->mutable_convolution_param()->kernel_w());
  
  ASSERT_EQ(23,deploy_net_param.layer_size());
  lparam = deploy_net_param.mutable_layer(0);
  ASSERT_EQ("MemoryData",lparam->type());
  ASSERT_EQ(1,lparam->mutable_memory_data_param()->channels());
  ASSERT_EQ(tcif._sequence,lparam->mutable_memory_data_param()->height());
  ASSERT_EQ(tcif._alphabet.size(),lparam->mutable_memory_data_param()->width());
  lparam = deploy_net_param.mutable_layer(1);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(256,lparam->mutable_convolution_param()->num_output());
  ASSERT_EQ(7,lparam->mutable_convolution_param()->kernel_h());
  ASSERT_EQ(69,lparam->mutable_convolution_param()->kernel_w());
  lparam = deploy_net_param.mutable_layer(3);
  ASSERT_EQ("Pooling",lparam->type());
  ASSERT_EQ(3,lparam->mutable_pooling_param()->stride_h());
  ASSERT_EQ(1,lparam->mutable_pooling_param()->stride_w());
  ASSERT_EQ(3,lparam->mutable_pooling_param()->kernel_h());
  ASSERT_EQ(1,lparam->mutable_pooling_param()->kernel_w());
  lparam = deploy_net_param.mutable_layer(7);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(256,lparam->mutable_convolution_param()->num_output());
  ASSERT_EQ(3,lparam->mutable_convolution_param()->kernel_h());
  ASSERT_EQ(1,lparam->mutable_convolution_param()->kernel_w());
  delete caff;
}

//TODO: lregression template

//TODO: regression one target template

//TODO: resnets

TEST(caffelib,configure_resnet_template_n_nt)
{
  int nclasses = 7;
  caffe::NetParameter net_param, deploy_net_param;
  std::vector<std::string> layers = {"Res50"};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","relu");
  //ad.add("dropout",0.6);
  ad.add("nclasses",nclasses);
  
  CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel());
  caff->_logger = spdlog::stdout_logger_mt("test");
  caff->configure_resnet_template(ad,ImgCaffeInputFileConn(),net_param,deploy_net_param);
  
  caffe::WriteProtoToTextFile(net_param,oresnet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,doresnet_file);
  bool succ = caffe::ReadProtoFromTextFile(oresnet_file,&net_param);
  ASSERT_TRUE(succ);

  ASSERT_EQ(278,net_param.layer_size());
  ASSERT_EQ(276,deploy_net_param.layer_size());
  delete caff;
}

TEST(caffelib, configure_deeplabvgg16_diceloss)
{
  APIData ad;
  ad.add("template","deeplab_vgg16");
  ad.add("loss","dice");
  APIData dice_param;
  APIData dice_class_weighting;
  dice_class_weighting.add("compute_on","batch");
  dice_class_weighting.add("weight","equalize_classes");
  dice_param.add("class_weighting",dice_class_weighting);
  APIData dice_contour;
  dice_contour.add("shape","simple");
  dice_contour.add("size",3);
  dice_contour.add("amplitude",20.5);
  dice_param.add("contour",dice_contour);
  ad.add("dice_param",dice_param);
  ad.add("ignore_label",0);
  ad.add("templates","../templates/caffe");
  ad.add("repository","./");
  ad.add("nclasses",2);
  ad.add("segmentation",true);
  CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel(ad));
  caff->_logger = spdlog::stdout_logger_mt("UT-deeplab_vgg16");
  caff->_inputc.init(ad);
  caff->init_mllib(ad);
  caff->instantiate_template(ad);
  caff->_loss = 1;

  caffe::NetParameter net_param, deploy_net_param;
  bool succ = caffe::ReadProtoFromTextFile("./deeplab_vgg16.prototxt",&net_param);
  ASSERT_TRUE(succ);

  bool found = false;
  int k = net_param.layer_size();
  caffe::LayerParameter *lparam;
  for (int l=k-1;l>0;l--)
    {
      lparam = net_param.mutable_layer(l);
      if (lparam->type() == "DiceCoefLoss")
        {
          found = true;
          break;
        }
    }
  ASSERT_TRUE(found);

  const caffe::DiceCoefLossParameter &clp = lparam->dice_coef_loss_param();
  ASSERT_TRUE(clp.generalization() == caffe::DiceCoefLossParameter::NONE);
  ASSERT_TRUE(clp.contour_shape() == caffe::DiceCoefLossParameter::SIMPLE);
  ASSERT_TRUE(clp.contour_size() == 3);
  ASSERT_TRUE(clp.contour_amplitude() == 20.5);


  remove("./deeplab_vgg16.prototxt");
  remove("./deeplab_vgg16_solver.prototxt");
  remove("./deploy.prototxt");
}

TEST(caffelib,configure_service_images_autoenc_geometry)
{
  APIData adg;
  adg.add("all_effects",false);
  adg.add("persp_horizontal",true);
  adg.add("persp_vertical",false);
  adg.add("zoom_in",true);
  adg.add("zoom_out",true);
  adg.add("pad_mode","mirrored");
  adg.add("prob",0.1);
  APIData ad;
  ad.add("template","convnet");
  std::vector<std::string> net = {"1CR32","1CR64","1CR128","DR128","1CR128","DR64","1CR64","DR32","1CR32"};
  ad.add("layers", net);
  ad.add("activation","relu");
  ad.add("autoencoder",true);
  ad.add("geometry",adg);
  ad.add("templates","../templates/caffe");
  ad.add("repository","./");
  ad.add("width",224);
  ad.add("height",224);

  CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel(ad));
  caff->_logger = spdlog::stdout_logger_mt("UT-geom");
  caff->_inputc.init(ad);
  //caff->init_mllib(ad);
  caff->instantiate_template(ad);


  caffe::NetParameter net_param;
  std::string prototxt = "convnet.prototxt";;
  bool succ = caffe::ReadProtoFromTextFile(prototxt,&net_param);
  ASSERT_TRUE(succ);


  caffe::LayerParameter *lparam = net_param.mutable_layer(0);
  caffe::GeometryParameter gparam = lparam->transform_param().geometry_param();
  ASSERT_FLOAT_EQ(gparam.prob(), 0.1);
  ASSERT_EQ(gparam.persp_horizontal(),true);
  ASSERT_EQ(gparam.persp_vertical(),false);
  ASSERT_EQ(gparam.zoom_out(),true);
  ASSERT_EQ(gparam.zoom_in(),true);
  ASSERT_EQ(gparam.pad_mode(),caffe::GeometryParameter_Pad_mode_MIRRORED);
  remove("./convnet.prototxt");
  remove("./convnet_solver.prototxt");
  remove("./deploy.prototxt");

}

TEST(caffelib, configure_unet_diceloss)
{
  APIData ad;
  ad.add("template","unet");
  ad.add("loss","dice");
  ad.add("templates","../templates/caffe");
  ad.add("repository","./");
  ad.add("nclasses",2);
  ad.add("segmentation",true);
  CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel> *caff = new CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>(CaffeModel(ad));
  caff->_logger = spdlog::stdout_logger_mt("UT-unet");
  caff->_inputc.init(ad);
  caff->init_mllib(ad);
  caff->instantiate_template(ad);
  caff->_loss = 1;

  caffe::NetParameter net_param, deploy_net_param;
  bool succ = caffe::ReadProtoFromTextFile("./unet.prototxt",&net_param);
  ASSERT_TRUE(succ);

  bool found = false;
  int k = net_param.layer_size();

  for (int l=k-1;l>0;l--)
    {
      caffe::LayerParameter *lparam = net_param.mutable_layer(l);
      if (lparam->type() == "DiceCoefLoss")
        found = true;
    }
  ASSERT_TRUE(found);

}
