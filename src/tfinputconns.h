/**
 * DeepDetect
 * Copyright (c) 2016 Emmanuel Benazera
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

#ifndef TFINPUTCONNS_H
#define TFINPUTCONNS_H

#include "imginputfileconn.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include <opencv2/opencv.hpp>

 namespace dd
 {
  class TFInputInterface
  {
  public:
    TFInputInterface() {}
    TFInputInterface(const TFInputInterface &tii)
    :_dv(tii._dv),_ids(tii._ids){}
    ~TFInputInterface() {}

  public:
    //TODO: parameters common to all TF input connectors
    std::vector<tensorflow::Tensor> _dv; // main tensor for prediction.
    std::vector<std::string> _ids; // input ids (eg. Image Ids).


  };

  class ImgTFInputFileConn : public ImgInputFileConn, public TFInputInterface
  {
  public:
    ImgTFInputFileConn()
    :ImgInputFileConn() {}
    ImgTFInputFileConn(const ImgTFInputFileConn &i)
    :ImgInputFileConn(i),TFInputInterface(i) {}
    ~ImgTFInputFileConn() {}

  // size of each element in tensorflow 
    int channels() const
    {
      if (_bw) return 1;
      else return 3; // RGB
    }
    
    int height() const
    {
      return _height;
    }
    
    int width() const
    {
      return _width;
    }

    int batch_size() const
    {
     if (!_dv.empty())
      return _dv.size();
    else return ImgInputFileConn::batch_size();
  }


  void init(const APIData &ad)
  {
    ImgInputFileConn::init(ad);
  }

  void transform(const APIData &ad)
  { 
    try
    {
      ImgInputFileConn::transform(ad);
    }
    catch(InputConnectorBadParamException &e)
    {
      throw;
    }
      // converting the input dataFrame into Tensor a Tensorflow DataStructure 
    _model_repo = ad.get("model_repo").get<std::string>();
    try{
      ImgInputFileConn::transform(ad);      
    }
    catch (InputConnectorBadParamException &e){
      throw;
    }

        // parameter for doing the Image Manipulation

    std::cout << "size of the _image is " << this->_images.size()<< std::endl;
    for (int i=0; i<(int)this->_images.size()-1;i++){
      tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,_height,_width,channels()}));
      auto input_tensor_mapped = input_tensor.tensor<float, 4>();

      cv::Mat readImage;

      readImage = this->_images.at(i);
      //std::cout <<"Image is empty or not ?? " <<Image.empty()<<std::endl;
      cv::Size s(_height,_width);
      cv::Mat Image;
      cv::resize(readImage,Image,s,0,0,cv::INTER_CUBIC);
      cv::Mat Image2;
      Image.convertTo(Image2, CV_32FC1);
      Image = Image2;
      Image = Image-_mean;
      Image = Image/_std;

      const float * source_data = (float*) Image.data;

                // copying the data into the corresponding tensor
      for (int y = 0; y < height(); ++y) {
        const float* source_row = source_data + (y * width()  * channels());
        for (int x = 0; x < width(); ++x) {
          const float* source_pixel = source_row + (x * channels());
          for (int c = 0; c < channels(); ++c) {
            const float* source_value = source_pixel + c;
            input_tensor_mapped(0, y, x, c) = *source_value;
          }
        }
      }
      std::cout << "I am here !!"<<std::endl;
      _dv.push_back(input_tensor);
      _ids.push_back(this->_uris.at(i));
      std::cout << "size of _dv in tfinput is " <<_dv.size()<< std::endl;
    }
    
  }

public:
    //TODO: image connector parameters
  int _mean = 128;
  int _std = 128;
  int _height = 299;
  int _width = 299;
  std::string _graphFile;
  std:: string _model_repo;
  std::vector<tensorflow::Tensor> _dv;
  std::vector<std::string> _ids;
};

}

#endif
