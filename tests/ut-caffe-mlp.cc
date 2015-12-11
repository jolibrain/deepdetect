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

TEST(caffelib,configure_mlp_template_1)
{
  int nclasses = 7;
  caffe::NetParameter net_param, deploy_net_param;
  bool succ = caffe::ReadProtoFromTextFile(net_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(dnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);

  std::vector<int> layers = {200};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.6);
  
  CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>::configure_mlp_template(ad,false,0,nclasses,net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,onet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,donet_file);
  succ = caffe::ReadProtoFromTextFile(onet_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(donet_file,&deploy_net_param);
  ASSERT_TRUE(succ);
  
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
}

TEST(caffelib,configure_mlp_template_n)
{
  int nclasses = 7;
  caffe::NetParameter net_param, deploy_net_param;
  bool succ = caffe::ReadProtoFromTextFile(net_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(dnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);
  
  std::vector<int> layers = {200,150,75};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.6);
  
  CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>::configure_mlp_template(ad,false,0,nclasses,net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,onet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,donet_file);
  succ = caffe::ReadProtoFromTextFile(onet_file,&net_param);
  ASSERT_TRUE(succ);

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
}

TEST(caffelib,configure_mlp_template_n_mt)
{
  int nclasses = 7;
  int targets = 3;
  caffe::NetParameter net_param, deploy_net_param;
  bool succ = caffe::ReadProtoFromTextFile(net_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(dnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);
  
  std::vector<int> layers = {200,150,75};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.6);
  
  CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>::configure_mlp_template(ad,false,targets,nclasses,net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,onet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,donet_file);
  succ = caffe::ReadProtoFromTextFile(onet_file,&net_param);
  ASSERT_TRUE(succ);

  ASSERT_EQ(15,net_param.layer_size());
  ASSERT_EQ(10,deploy_net_param.layer_size());
  int rl = 1;
  caffe::LayerParameter *lparam;
  lparam = net_param.mutable_layer(++rl);
  ASSERT_EQ("Slice",lparam->type());
  ASSERT_EQ(nclasses,lparam->mutable_slice_param()->slice_point(0));
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
  ASSERT_EQ(nclasses,dlparam->mutable_slice_param()->slice_point(0));
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
}

TEST(caffelib,configure_convnet_template_1)
{
  int nclasses = 18;
  caffe::NetParameter net_param, deploy_net_param;
  bool succ = caffe::ReadProtoFromTextFile(convnet_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(dconvnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);

  std::vector<std::string> layers = {"1CR64"};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.2);
  
  CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>::configure_convnet_template(ad,false,0,nclasses,CSVCaffeInputFileConn(),net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,oconvnet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,doconvnet_file);
  succ = caffe::ReadProtoFromTextFile(oconvnet_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(doconvnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);

  ASSERT_EQ(9,net_param.layer_size());
  caffe::LayerParameter *lparam = net_param.mutable_layer(2);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(64,lparam->mutable_convolution_param()->num_output());
  lparam = net_param.mutable_layer(3);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(4);
  ASSERT_EQ("Pooling",lparam->type());
  lparam = net_param.mutable_layer(5);
  ASSERT_EQ("Dropout",lparam->type());
  ASSERT_NEAR(0.2,lparam->mutable_dropout_param()->dropout_ratio(),1e-5); // near as there seems to be a slight conversion issue from protobufs
  lparam = net_param.mutable_layer(6);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(nclasses,lparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("pool0",lparam->bottom(0));

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
  ASSERT_EQ("pool0",dlparam->bottom(0));
}

TEST(caffelib,configure_convnet_template_2)
{
  int nclasses = 18;
  caffe::NetParameter net_param, deploy_net_param;
  bool succ = caffe::ReadProtoFromTextFile(convnet_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(dconvnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);

  std::vector<std::string> layers = {"1CR64","1CR128","1000"};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.2);
  
  CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>::configure_convnet_template(ad,false,0,nclasses,CSVCaffeInputFileConn(),net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,oconvnet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,doconvnet_file);
  succ = caffe::ReadProtoFromTextFile(oconvnet_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(doconvnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);

  ASSERT_EQ(16,net_param.layer_size());
  caffe::LayerParameter *lparam = net_param.mutable_layer(2);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(64,lparam->mutable_convolution_param()->num_output());
  lparam = net_param.mutable_layer(3);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(4);
  ASSERT_EQ("Pooling",lparam->type());
  lparam = net_param.mutable_layer(5);
  ASSERT_EQ("Dropout",lparam->type());
  ASSERT_NEAR(0.2,lparam->mutable_dropout_param()->dropout_ratio(),1e-5); // near as there seems to be a slight conversion issue from protobufs
  lparam = net_param.mutable_layer(6);
  ASSERT_EQ("Convolution",lparam->type());
  ASSERT_EQ(128,lparam->mutable_convolution_param()->num_output());
  lparam = net_param.mutable_layer(7);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(8);
  ASSERT_EQ("Pooling",lparam->type());
  lparam = net_param.mutable_layer(9);
  ASSERT_EQ("Dropout",lparam->type());
  ASSERT_NEAR(0.2,lparam->mutable_dropout_param()->dropout_ratio(),1e-5);
  lparam = net_param.mutable_layer(10);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(1000,lparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("pool1",lparam->bottom(0));
  lparam = net_param.mutable_layer(11);
  ASSERT_EQ("PReLU",lparam->type());
  lparam = net_param.mutable_layer(12);
  ASSERT_EQ("Dropout",lparam->type());
  ASSERT_NEAR(0.2,lparam->mutable_dropout_param()->dropout_ratio(),1e-5);
  lparam = net_param.mutable_layer(13);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(nclasses,lparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("ip2",lparam->bottom(0));
  ASSERT_EQ("ip3",lparam->top(0));

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
  ASSERT_EQ("pool1",dlparam->bottom(0));
  dlparam = deploy_net_param.mutable_layer(8);
  ASSERT_EQ("PReLU",dlparam->type());
  dlparam = deploy_net_param.mutable_layer(9);
  ASSERT_EQ("InnerProduct",dlparam->type());
  ASSERT_EQ(nclasses,dlparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("ip2",dlparam->bottom(0)); 
  ASSERT_EQ("ip3",dlparam->top(0));
}

TEST(caffelib,configure_convnet_template_3)
{
  int nclasses = 18;
  caffe::NetParameter net_param, deploy_net_param;
  bool succ = caffe::ReadProtoFromTextFile(convnet_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(dconvnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);

  std::vector<std::string> layers = {"2CR64"};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.2);
  
  CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>::configure_convnet_template(ad,false,0,nclasses,CSVCaffeInputFileConn(),net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,oconvnet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,doconvnet_file);
  succ = caffe::ReadProtoFromTextFile(oconvnet_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(doconvnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);

  ASSERT_EQ(11,net_param.layer_size());
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
  ASSERT_EQ("Dropout",lparam->type());
  lparam = net_param.mutable_layer(8);
  ASSERT_EQ("InnerProduct",lparam->type());
  ASSERT_EQ(nclasses,lparam->mutable_inner_product_param()->num_output());
  ASSERT_EQ("pool0",lparam->bottom(0));
  
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
  ASSERT_EQ("pool0",dlparam->bottom(0));
}

TEST(caffelib,configure_convnet_template_n)
{
  int nclasses = 18;
  caffe::NetParameter net_param, deploy_net_param;
  bool succ = caffe::ReadProtoFromTextFile(convnet_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(dconvnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);

  std::vector<std::string> layers = {"2CR32","2CR64","2CR128","4096","1024"};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.2);
  
  CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>::configure_convnet_template(ad,false,0,nclasses,CSVCaffeInputFileConn(),net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,oconvnet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,doconvnet_file);
  succ = caffe::ReadProtoFromTextFile(oconvnet_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(doconvnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);

  ASSERT_EQ(29,net_param.layer_size());
  ASSERT_EQ(22,deploy_net_param.layer_size());
}

TEST(caffelib,configure_convnet_template_n_1D)
{
  int nclasses = 18;
  caffe::NetParameter net_param, deploy_net_param;
  bool succ = caffe::ReadProtoFromTextFile(convnet_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(dconvnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);

  std::vector<std::string> layers = {"1CR256","1CR256","4CR256","1024","1024"};
  APIData ad;
  ad.add("layers",layers);
  ad.add("activation","prelu");
  ad.add("dropout",0.2);

  TxtCaffeInputFileConn tcif;
  tcif._characters = true;
  tcif.build_alphabet();
  CaffeLib<TxtCaffeInputFileConn,SupervisedOutput,CaffeModel>::configure_convnet_template(ad,false,0,nclasses,tcif,net_param,deploy_net_param);

  caffe::WriteProtoToTextFile(net_param,oconvnet_file);
  caffe::WriteProtoToTextFile(deploy_net_param,doconvnet_file);
  succ = caffe::ReadProtoFromTextFile(oconvnet_file,&net_param);
  ASSERT_TRUE(succ);
  succ = caffe::ReadProtoFromTextFile(doconvnet_file,&deploy_net_param);
  ASSERT_TRUE(succ);

  ASSERT_EQ(29,net_param.layer_size());
  ASSERT_EQ(22,deploy_net_param.layer_size());
}
