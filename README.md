## DeepDetect : Open Source Deep Learning Server & API

[![Join the chat at https://gitter.im/beniz/deepdetect](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/beniz/deepdetect?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

DeepDetect (http://www.deepdetect.com/) is a machine learning API and server written in C++11. It makes state of the art machine learning easy to work with and integrate into existing applications.

DeepDetect relies on external machine learning libraries through a very generic and flexible API. At the moment it has support for:

- the deep learning library [Caffe](https://github.com/BVLC/caffe)
- distributed gradient boosting library [XGBoost](https://github.com/dmlc/xgboost)
- the deep learning and other usages library [Tensorflow](https://tensorflow.org)
- clustering with [T-SNE](https://github.com/DmitryUlyanov/Multicore-TSNE)

#### Machine Learning functionalities per library (current):

|            | Training | Prediction | Classification | Object Detection | Regression | Autoencoder |
|------------|----------|------------|----------------|-----------|------------|-------------|
| Caffe      | Y        | Y          | Y              | Y         |   Y        | Y           |
| XGBoost    | Y        | Y          | Y              | N         |   Y        | N/A         |
| Tensorflow | N        | Y          | Y              | N         |   N        | N           |
| T-SNE      | Y        | N/A        | N/A            | N/A       |   N/A      | N/A         |


#### GPU support per library

|            | Training | Prediction |
|------------|----------|------------|
| Caffe      | Y        | Y          |
| XGBoost    | Y        | N          |
| Tensorflow | Y        | Y          |
| T-SNE      | N        | N          |

#### Input data support per library (current):

|            | CSV | SVM | Text words | Text characters | Images |
|------------|-----|-----|------------|-----------------|--------|
| Caffe      | Y   | Y   | Y          | Y               | Y      |
| XGBoost    | Y   | Y   | Y          | N               | N      |
| Tensorflow | N   | N   | N          | N               | Y      |
| T-SNE      | Y   | N   | N          | N               | Y      | (*)
(*) more input support for T-SNE is pending

#### Main functionalities

DeepDetect implements support for supervised and unsupervised deep learning of images, text and other data, with focus on simplicity and ease of use, test and connection into existing applications. It supports classification, object detection, regression, autoencoders, ...

#### Support

Please join either the community on [Gitter](https://gitter.im/beniz/deepdetect) or on IRC Freenode #deepdetect, where we help users get through with installation, API, neural nets and connection to external applications.

#### Supported Platforms

The reference platforms with support are **Ubuntu 14.04 LTS** and **Ubuntu 16.04 LTS**.

Supported images that come with pre-trained image classification deep (residual) neural nets:

- **docker images** for CPU and GPU machines are available at https://hub.docker.com/r/beniz/deepdetect_cpu/ and https://hub.docker.com/r/beniz/deepdetect_gpu/ respectively. See https://github.com/beniz/deepdetect/tree/master/docker/README.md for details on how to use them.

- For **Amazon AMI** see official builds documentation at https://deepdetect.com/products/ami/, and direct links to [GPU AMI](https://aws.amazon.com/marketplace/pp/B01N4D483M) and [CPU AMI](https://aws.amazon.com/marketplace/pp/B01N1RGWQZ).

#### Quickstart
Setup an image classifier API service in a few minutes:
http://www.deepdetect.com/tutorials/imagenet-classifier/

#### Tutorials
List of tutorials, training from text, data and images, setup of prediction services, and export to external software (e.g. ElasticSearch): http://www.deepdetect.com/tutorials/tutorials/

#### Features and Documentation
Current features include:

- high-level API for machine learning and deep learning
- Support for Caffe, Tensorflow, XGBoost and T-SNE
- classification, regression, autoencoders, object detection
- JSON communication format
- remote Python client library
- dedicated server with support for asynchronous training calls
- high performances, benefit from multicore CPU and GPU
- connector to handle large collections of images with on-the-fly data augmentation (e.g. rotations, mirroring)
- connector to handle CSV files with preprocessing capabilities
- connector to handle text files, sentences, and character-based models
- connector to handle SVM file format for sparse data
- range of built-in model assessment measures (e.g. F1, multiclass log loss, ...)
- no database dependency and sync, all information and model parameters organized and available from the filesystem
- flexible template output format to simplify connection to external applications
- templates for the most useful neural architectures (e.g. Googlenet, Alexnet, ResNet, convnet, character-based convnet, mlp, logistic regression)
- support for sparse features and computations on both GPU and CPU

##### Documentation

- Full documentation is available from http://www.deepdetect.com/overview/introduction/
- API documentation is available from http://www.deepdetect.com/api/
- FAQ is available from http://www.deepdetect.com/overview/faq/

##### Clients

- Python client:
  - REST client: https://github.com/beniz/deepdetect/tree/master/clients/python
  - 'a la scikit' bindings: https://github.com/ArdalanM/pyDD
- Java client: https://github.com/kfadhel/deepdetect-api-java
- Early C# client: https://github.com/beniz/deepdetect/pull/98

##### Dependencies

- C++, gcc >= 4.8 or clang with support for C++11 (there are issues with Clang + Boost)
- [eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for all matrix operations;
- [glog](https://code.google.com/p/google-glog/) for logging events and debug;
- [gflags](https://code.google.com/p/gflags/) for command line parsing;
- OpenCV >= 2.4
- [cppnetlib](http://cpp-netlib.org/)
- Boost
- [curl](http://curl.haxx.se/)
- [curlpp](http://www.curlpp.org/)
- [utfcpp](http://utfcpp.sourceforge.net/)
- [gtest](https://code.google.com/p/googletest/) for unit testing (optional);

##### Caffe Dependencies

- CUDA 8 or 7.5 is recommended for GPU mode.
- BLAS via ATLAS, MKL, or OpenBLAS.
- [protobuf](https://github.com/google/protobuf)
- IO libraries hdf5, leveldb, snappy, lmdb

##### XGBoost Dependencies

None outside of C++ compiler and make
- CUDA 8 or 7.5 is recommended for GPU mode.

#### Tensorflow Dependencies

- Cmake > 3
- [Bazel](https://www.bazel.io/versions/master/docs/install.html#install-on-ubuntu)

##### Caffe version

By default DeepDetect automatically relies on a modified version of Caffe, https://github.com/beniz/caffe/tree/master
This version includes many improvements over the original Caffe, such as sparse input data support, exception handling, class weights, object detection, and various additional losses and layers.

##### Implementation

The code makes use of C++ policy design for modularity, performance and putting the maximum burden on the checks at compile time. The implementation uses many features from C++11.

##### Demo

- Image classification Web interface:
HTML and javascript classification image demo in [demo/imgdetect](https://github.com/beniz/deepdetect/tree/master/demo/imgdetect)

- Image similarity search:
Python script for indexing and searching images is in [demo/imgsearch](https://github.com/beniz/deepdetect/tree/master/demo/imgsearch)

- Image object detection:
Python script for object detection within images is in [demo/objdetect](https://github.com/beniz/deepdetect/tree/master/demo/objdetect)

##### Examples

- List of examples, from MLP for data, text, multi-target regression to CNN and GoogleNet, finetuning, etc...:
http://www.deepdetect.com/overview/examples/

##### Models

|                          | Caffe | Tensorflow | Source        | Top-1 Accuracy (ImageNet) |
|--------------------------|-------|------------|---------------|---------------------------|
| AlexNet                  | Y     | N          | BVLC          |          57.1%                 |
| SqueezeNet               | Y     | N          | DeepScale              |       59.5%                    | 
| Inception v1 / GoogleNet | [Y](https://deepdetect.com/models/ggnet/bvlc_googlenet.caffemodel)     | [Y](https://deepdetect.com/models/tf/inception_v1.pb)          | BVLC / Google |             67.9%              |
| Inception v2             | N     | [Y](https://deepdetect.com/models/tf/inception_v2.pb)          | Google        |     72.2%                      |
| Inception v3             | N     | [Y](https://deepdetect.com/models/tf/inception_v3.pb)          | Google        |         76.9%                  |
| Inception v4             | N     | [Y](https://deepdetect.com/models/tf/inception_v4.pb)          | Google        |         80.2%                  |
| ResNet 50                | [Y](https://deepdetect.com/models/resnet/ResNet-50-model.caffemodel)     | [Y](https://deepdetect.com/models/tf/resnet_v1_50/resnet_v1_50.pb)          | MSR           |      75.3%                     |
| ResNet 101               | [Y](https://deepdetect.com/models/resnet/ResNet-101-model.caffemodel)     | [Y](https://deepdetect.com/models/tf/resnet_v1_101/resnet_v1_101.pb)          | MSR           |        76.4%                   |
| ResNet 152               | [Y](https://deepdetect.com/models/resnet/ResNet-152-model.caffemodel)     | [Y](https://deepdetect.com/models/tf/resnet_v1_152/resnet_v1_152.pb)         | MSR           |               77%            |
| Inception-ResNet-v2      | N     | [Y](https://deepdetect.com/models/tf/inception_resnet_v2.pb)          | Google        |       79.79%                    |
| VGG-16                   | [Y](https://deepdetect.com/models/vgg_16/VGG_ILSVRC_16_layers.caffemodel)     | [Y](https://deepdetect.com/models/tf/vgg_16/vgg_16.pb)          | Oxford        |               70.5%            |
| VGG-19                   | [Y](https://deepdetect.com/models/vgg_16/VGG_ILSVRC_16_layers.caffemodel)     | [Y](https://deepdetect.com/models/tf/vgg_19/vgg_19.pb)          | Oxford        |               71.3%            |
| ResNext 50                | [Y](https://deepdetect.com/models/resnext/resnext_50)     | N          | https://github.com/terrychenism/ResNeXt           |      76.9%                     |
| ResNext 101                | [Y](https://deepdetect.com/models/resnext/resnext_101)     | N          | https://github.com/terrychenism/ResNeXt           |      77.9%                     |
| ResNext 152               | [Y](https://deepdetect.com/models/resnext/resnext_152)     | N          | https://github.com/terrychenism/ResNeXt           |      78.7%                     |
| DenseNet-121                   | [Y](https://deepdetect.com/models/densenet/densenet_121_32/)     | N          | https://github.com/shicai/DenseNet-Caffe        |               74.9%            |
| DenseNet-161                   | [Y](https://deepdetect.com/models/densenet/densenet_161_48/)     | N          | https://github.com/shicai/DenseNet-Caffe        |               77.6%            |
| DenseNet-169                   | [Y](https://deepdetect.com/models/densenet/densenet_169_32/)     | N          | https://github.com/shicai/DenseNet-Caffe        |               76.1%            |
| DenseNet-201                   | [Y](https://deepdetect.com/models/densenet/densenet_201_32/)     | N          | https://github.com/shicai/DenseNet-Caffe        |               77.3%            |
| VOC0712 (object detection) | [Y](https://deepdetect.com/models/voc0712_dd.tar.gz) | N | https://github.com/weiliu89/caffe/tree/ssd | 71.2 mAP |

More models:

- List of free, even for commercial use, deep neural nets for image classification, and character-based convolutional nets for text classification: http://www.deepdetect.com/applications/list_models/

#### Templates

DeepDetect comes with a built-in system of neural network templates (Caffe backend only at the moment). This allows the creation of custom networks based on recognized architectures, for images, text and data, and with much simplicity.

Usage:
- specify `template` to use, from `mlp`, `convnet` and `resnet`
- specify the architecture with the `layers` parameter:
  - for `mlp`, e.g. `[300,100,10]`
  - for `convnet`, e.g. `["1CR64","1CR128","2CR256","1024","512"], where the main pattern is `xCRy` where `y` is the number of outputs (feature maps), `CR` stands for Convolution + Activation (with `relu` as default), and `x` specifies the number of chained `CR` blocks without pooling. Pooling is applied between all `xCRy`
- for `resnets`:
   - with images, e.g. `["Res50"]` where the main pattern is `ResX` with X the depth of the Resnet
   - with character-based models (text), use the `xCRy` pattern of convnets instead, with the main difference that `x` now specifies the number of chained `CR` blocks within a resnet block
   - for Resnets applied to CSV or SVM (sparse data), use the `mlp` pattern. In this latter case, at the moment, the `resnet` is built with blocks made of two layers for each specified layer after the first one. Here is an example: `[300,100,10]` means that a first hidden layer of size `300` is applied followed by a `resnet` block made of two `100` fully connected layer, and another block of two `10` fully connected layers. This is subjected to future changes and more control.

### Authors
DeepDetect is designed and implemented by Emmanuel Benazera <beniz@droidnik.fr>.

### Build

Below are instructions for Ubuntu 14.04 LTS. For other Linux and Unix systems, steps may differ, CUDA, Caffe and other libraries may prove difficult to setup. If you are building on 16.04 LTS, look at https://github.com/beniz/deepdetect/issues/126 that tells you how to proceed.

Beware of dependencies, typically on Debian/Ubuntu Linux, do:
```
sudo apt-get install build-essential libgoogle-glog-dev libgflags-dev libeigen3-dev libopencv-dev libcppnetlib-dev libboost-dev libboost-iostreams-dev libcurlpp-dev libcurl4-openssl-dev protobuf-compiler libopenblas-dev libhdf5-dev libprotobuf-dev libleveldb-dev libsnappy-dev liblmdb-dev libutfcpp-dev cmake libgoogle-perftools-dev unzip
```

#### Default build with Caffe
For compiling along with Caffe:
```
mkdir build
cd build
cmake ..
make
```

If you are building for one or more GPUs, you may need to add CUDA to your ld path:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

If you would like to build with cuDNN, your `cmake` line should be:
```
cmake .. -DUSE_CUDNN=ON
```

To target the build of underlying Caffe to a specific CUDA architecture (e.g. Pascal), you can use:
```
cmake .. -DCUDA_ARCH="-gencode arch=compute_61,code=sm_61"
```

If you would like a CPU only build, use:
```
cmake .. -DUSE_CPU_ONLY=ON
```

#### Build with XGBoost support

If you would like to build with XGBoost, include the `-DUSE_XGBOOST=ON` parameter to `cmake`:
```
cmake .. -DUSE_XGBOOST=ON
```

#### Build with Tensorflow support
First you must install [Bazel](https://www.bazel.io/versions/master/docs/install.html#install-on-ubuntu) and Cmake with version > 3.

And other dependencies:
```
sudo apt-get install python-numpy swig python-dev python-wheel unzip
```

If you would like to build with Tensorflow, include the `-DUSE_TF=ON` paramter to `cmake`:
```
cmake .. -DUSE_TF=ON
```

You can combine with XGBoost support with:
```
cmake .. -DUSE_TF=on -DUSE_XGBOOST=ON
```

#### Build with T-SNE support

Simply specify the option via cmake command line:
```
cmake .. -DUSE_TSNE=ON
```

### Run tests

Note: running tests requires the automated download of ~75Mb of datasets, and computations may take around thirty minutes on a CPU-only machines.

To prepare for tests, compile with:
```
cmake -DBUILD_TESTS=ON ..
make
```
Run tests with:
```
ctest
```

### Start the server

```
cd build/main
./dede

DeepDetect [ commit 73d4e638498d51254862572fe577a21ab8de2ef1 ]
Running DeepDetect HTTP server on localhost:8080
```

Main options are:
- `-host` to select which host to run on, default is `localhost`, use `0.0.0.0` to listen on all interfaces
- `-port` to select which port to listen to, default is `8080`
- `-nthreads` to select the number of HTTP threads, default is `10`

To see all options, do:
```
./dede --help
```

### Run examples

See tutorials from http://www.deepdetect.com/tutorials/tutorials/

### References

- DeepDetect (http://www.deepdetect.com/)
- Caffe (https://github.com/BVLC/caffe)
- XGBoost (https://github.com/dmlc/xgboost)
- T-SNE (https://github.com/DmitryUlyanov/Multicore-TSNE)
