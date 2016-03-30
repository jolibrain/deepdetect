/*

  Following file take opencv mat file as an input and convert it to proper tensor object 
  Created by : Kumar Shubham
  Date : 27-03-2016
*/

//Loading Opencv fIles for processing

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <iostream>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"



int main(int argc, char** argv)
{
  
  // Loading the file path provided in the arg into a mat objects
  std::string path = argv[1];
  cv::Mat Image = cv::imread(path);

  // Getting the size and other details from image
  cv::Size s = Image.size();
  int height = s.height;
  int width = s.width;
  int depth = Image.channels();

  std::cerr << "height=" << height << " / width=" << width << " / depth=" << depth << std::endl;
  
  // creating a Tensor for storing the data
  tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,height,width,depth}));
  auto input_tensor_mapped = input_tensor.tensor<float, 4>();
  
  // Data pointer to enumerate the pixel and its corresponding value
  std::vector<float> array;
  if (Image.isContinuous()) {
    array.assign((float*)Image.datastart, (float*)Image.dataend);
  } else {
    for (int i = 0; i < Image.rows; ++i) {
      array.insert(array.end(), (float*)Image.ptr<uchar>(i), (float*)Image.ptr<uchar>(i)+Image.cols);
    }
  }
  //const float * source_data = (float*) Image.data;
  const float * source_data = &array[0];
  
  // copying the data into the corresponding tensor

  for (int y = 0; y < height; ++y) {
    const float* source_row = source_data + (y * width * depth);
    for (int x = 0; x < width; ++x) {
      const float* source_pixel = source_row + (x * depth);
      for (int c = 0; c < depth; ++c) {
	const float* source_value = source_pixel + c;
	input_tensor_mapped(0, y, x, c) = *source_value;
	/*std::cerr << "y=" << y << " / x=" << x << " / c=" << c << std::endl;
	std::cerr << "source_value=" << *source_value << std::endl; 
	input_tensor_mapped(0, 0, 0, 0) = *source_value;*/
      }
    }
  }
  /*cv::namedWindow("imageOpencv",CV_WINDOW_KEEPRATIO);
  cv::imshow("imgOpencv",Image);
  cv::waitKey(0);*/
}
  
