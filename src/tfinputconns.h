/*copyright (c) 2016 Emmanuel Benazera
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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"


#include "inputconnectorstrategy.h"
#include "ext/base64/base64.h"
#include <glog/logging.h>

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

  class ImgTFInputFileConn : public InputConnectorStrategy, public TFInputInterface

 {
  public:
    ImgTFInputFileConn()
    :InputConnectorStrategy() {}
    ImgTFInputFileConn(const ImgTFInputFileConn &i)
    :InputConnectorStrategy(),TFInputInterface(i) {}
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
    else return 0;
  }

void init(const APIData &ad)
    {
      fillup_parameters(ad);
    }
  // COde for reading an Image file and converting it into tensor
int ReadTensorFromImageFile(std::string file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<tensorflow::Tensor>* out_tensors){
  tensorflow::GraphDefBuilder b;
  std::string input_name = "file_reader";
  std::string output_name = "normalized";
  tensorflow::Node* file_render;

   
tensorflow::Node* file_reader =
      tensorflow::ops::ReadFile(tensorflow::ops::Const(file_name, b.opts()),
                                b.opts().WithName(input_name));
  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = channels() ;
  tensorflow::Node* image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = tensorflow::ops::DecodePng(
        file_reader,
        b.opts().WithAttr("channels", wanted_channels).WithName("png_reader"));
  } else {
    // Assume if it's not a PNG then it must be a JPEG.
    image_reader = tensorflow::ops::DecodeJpeg(
        file_reader,
        b.opts().WithAttr("channels", wanted_channels).WithName("jpeg_reader"));
  }
  // Now cast the image data to float so we can do normal math on it.
  tensorflow::Node* float_caster = tensorflow::ops::Cast(
      image_reader, tensorflow::DT_FLOAT, b.opts().WithName("float_caster"));
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  tensorflow::Node* dims_expander = tensorflow::ops::ExpandDims(
      float_caster, tensorflow::ops::Const(0, b.opts()), b.opts());
  // Bilinearly resize the image to fit the required dimensions.
  tensorflow::Node* resized = tensorflow::ops::ResizeBilinear(
      dims_expander, tensorflow::ops::Const({input_height, input_width},
                                            b.opts().WithName("size")),
      b.opts());
  // Subtract the mean and divide by the scale.
  tensorflow::ops::Div(
      tensorflow::ops::Sub(
          resized, tensorflow::ops::Const({input_mean}, b.opts()), b.opts()),
      tensorflow::ops::Const({input_std}, b.opts()),
      b.opts().WithName(output_name));

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  tensorflow::Status toGraph_status = b.ToGraphDef(&graph);
  if(!toGraph_status.ok()){
    return 1;
  }
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  tensorflow::Status createGraph_status = session->Create(graph);
  if(!createGraph_status.ok()){
    return 1;
  }
  tensorflow::Status run_status = session->Run({}, {output_name}, {}, out_tensors);
  if (!run_status.ok()){
  return 1;
  }
  return 0;
}

//file to fill Parameter
void fillup_parameters(const APIData &ad)
    {
      // optional parameters.
      if (ad.has("width"))
  _width = ad.get("width").get<int>();
      if (ad.has("height"))
  _height = ad.get("height").get<int>();
      if (ad.has("bw"))
  _bw = ad.get("bw").get<bool>();
      if (ad.has("shuffle"))
  _shuffle = ad.get("shuffle").get<bool>();
      if (ad.has("seed"))
  _seed = ad.get("seed").get<int>();
      if (ad.has("test_split"))
  _test_split = ad.get("test_split").get<double>();
    }
  void transform(const APIData &ad)
  { 
    // try
    // {
    //   ImgInputFileConn::transform(ad);
    // }
    // catch(InputConnectorBadParamException &e)
    // {
    //   throw;
    // }

    //File to make tfindependent of opencv files
  InputConnectorStrategy::get_data(ad);
  




      // converting the input dataFrame into Tensor a Tensorflow DataStructure 
    _model_repo = ad.get("model_repo").get<std::string>();
    
        // parameter for doing the Image Manipulation

    std::cout << "size of the _image is " << _uris.size()<< std::endl;
    for (int i=0; i<(int)_uris.size();i++){
      // tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,_height,_width,channels()}));
      // auto input_tensor_mapped = input_tensor.tensor<float, 4>();

      // cv::Mat readImage;

      // readImage = this->_images.at(i);
      // //std::cout <<"Image is empty or not ?? " <<Image.empty()<<std::endl;
      // cv::Size s(_height,_width);
      // cv::Mat Image;
      // cv::resize(readImage,Image,s,0,0,cv::INTER_CUBIC);
      // cv::Mat Image2;
      // Image.convertTo(Image2, CV_32FC1);
      // Image = Image2;
      // Image = Image-_mean;
      // Image = Image/_std;

      // const float * source_data = (float*) Image.data;

      //           // copying the data into the corresponding tensor
      // for (int y = 0; y < height(); ++y) {
      //   const float* source_row = source_data + (y * width()  * channels());
      //   for (int x = 0; x < width(); ++x) {
      //     const float* source_pixel = source_row + (x * channels());
      //     for (int c = 0; c < channels(); ++c) {
      //       const float* source_value = source_pixel + c;
      //       input_tensor_mapped(0, y, x, c) = *source_value;
      //     }
      //   }
      // }
      // std::cout << "I am here !!"<<std::endl;

      std::vector<tensorflow::Tensor> input_tensor_vector;
      int readTensorStatus = ReadTensorFromImageFile(_uris.at(i), _height,
                               _width, _mean,
                               _std,
                                &input_tensor_vector);
      
      _dv.push_back(input_tensor_vector[0]);
      _ids.push_back(_uris.at(i));
      std::cout << "size of _dv in tfinput is " <<_dv.size()<< std::endl;
    }
    
  }

public:
    //TODO: image connector parameters
  int _mean = 128;
  int _std = 128;
  int _height = 299;
  int _width = 299;
  bool _bw = false; /**< whether to convert to black & white. */
  double _test_split = 0.0; /**< auto-split of the dataset. */
  bool _shuffle = false; /**< whether to shuffle the dataset, usually before splitting. */
  int _seed = -1; /**< shuffling seed. */
  std::string _graphFile;
  std:: string _model_repo;
  std::vector<tensorflow::Tensor> _dv;
  std::vector<std::string> _ids;
};

}

#endif

