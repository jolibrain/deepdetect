
/*
  Following file take opencv mat file as an input and convert it to proper
  tensor object Created by : Kumar Shubham Date : 27-03-2016
*/

// Loading Opencv fIles for processing

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <iostream>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include <fstream>

int main(int argc, char **argv)
{

  // Loading the file path provided in the arg into a mat objects
  std::string path = argv[1];
  cv::Mat readImage = cv::imread(path);
  std::cerr << "read image=" << path << std::endl;

  // converting the image to the necessary dim and for normalization
  int height = 299;
  int width = 299;
  int mean = 128;
  int std = 128;

  cv::Size s(height, width);
  cv::Mat Image;
  std::cerr << "resizing\n";
  cv::resize(readImage, Image, s, 0, 0, cv::INTER_CUBIC);
  std::cerr << "success resizing\n";

  int depth = Image.channels();

  // std::cerr << "height=" << height << " / width=" << width << " / depth=" <<
  // depth << std::endl;

  // creating a Tensor for storing the data
  tensorflow::Tensor input_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({ 1, height, width, depth }));
  auto input_tensor_mapped = input_tensor.tensor<float, 4>();

  cv::Mat Image2;
  Image.convertTo(Image2, CV_32FC1);
  Image = Image2;
  Image = Image - mean;
  Image = Image / std;
  const float *source_data = (float *)Image.data;

  // copying the data into the corresponding tensor
  for (int y = 0; y < height; ++y)
    {
      const float *source_row = source_data + (y * width * depth);
      for (int x = 0; x < width; ++x)
        {
          const float *source_pixel = source_row + (x * depth);
          for (int c = 0; c < depth; ++c)
            {
              const float *source_value = source_pixel + c;
              input_tensor_mapped(0, y, x, c) = *source_value;
            }
        }
    }

  // initializing the graph
  tensorflow::GraphDef graph_def;

  // Name of the folder in which inception graph is present
  std::string graphFile = "../../model/tensorflow_inception_graph.pb";

  // Loading the graph to the given variable
  tensorflow::Status graphLoadedStatus
      = ReadBinaryProto(tensorflow::Env::Default(), graphFile, &graph_def);
  if (!graphLoadedStatus.ok())
    {
      std::cout << graphLoadedStatus.ToString() << std::endl;
      return 1;
    }

  // creating a session with the grap
  std::unique_ptr<tensorflow::Session> session_inception(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  // session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  tensorflow::Status session_create_status
      = session_inception->Create(graph_def);

  if (!session_create_status.ok())
    {
      return 1;
    }
  // running the loaded graph
  std::vector<tensorflow::Tensor> finalOutput;

  std::string InputName = "Mul";
  std::string OutputName = "softmax";
  tensorflow::Status run_status = session_inception->Run(
      { { InputName, input_tensor } }, { OutputName }, {}, &finalOutput);

  // finding the labels for prediction
  std::cerr << "final output size=" << finalOutput.size() << std::endl;
  tensorflow::Tensor output = std::move(finalOutput.at(0));
  auto scores = output.flat<float>();
  std::cerr << "scores size=" << scores.size() << std::endl;

  // Label File Name
  std::string labelfile = "../../model/imagenet_comp_graph_label_strings.txt";
  std::ifstream label(labelfile);
  std::string line;

  // sorting the file to find the top labels
  std::vector<std::pair<float, std::string>> sorted;

  for (unsigned int i = 0; i <= 1000; ++i)
    {
      std::getline(label, line);
      sorted.emplace_back(scores(i), line);
      // std::cout << scores(i) << " / line=" << line << std::endl;
    }

  std::sort(sorted.begin(), sorted.end());
  std::reverse(sorted.begin(), sorted.end());
  std::cout << "size of the sorted file is " << sorted.size() << std::endl;
  for (unsigned int i = 0; i < 5; ++i)
    {

      std::cout << "The output of the current graph has category  "
                << sorted[i].second << " with probability " << sorted[i].first
                << std::endl;
    }

  /*cv::namedWindow("imageOpencv",CV_WINDOW_KEEPRATIO);
  cv::imshow("imgOpencv",Image);

  */
}
