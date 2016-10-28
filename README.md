## DeepDetect : Open Source Deep Learning Server & API

[![Join the chat at https://gitter.im/beniz/deepdetect](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/beniz/deepdetect?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

DeepDetect (http://www.deepdetect.com/) is a machine learning API and server written in C++11. It makes state of the art machine learning easy to work with and integrate into existing applications.

DeepDetect relies on external machine learning libraries through a very generic and flexible API. At the moment it has support for the deep learning library [Caffe](https://github.com/BVLC/caffe) and distributed gradient boosting library [XGBoost](https://github.com/dmlc/xgboost).

#### Main functionalities

DeepDetect implements support for supervised deep learning of images, text and other data, with focus on simplicity and ease of use, test and connection into existing applications.

#### Support

Please join either the community on [Gitter](https://gitter.im/beniz/deepdetect) or on IRC Freenode #deepdetect, where we help users get through with installation, API, neural nets and connection to external applications.

#### Supported Platforms

The reference platform with support is **Ubuntu 14.04 LTS**.

Supported images that come with pre-trained image classification deep (residual) neural nets:

- **docker images** for CPU and GPU machines are available at https://hub.docker.com/r/beniz/deepdetect_cpu/ and https://hub.docker.com/r/beniz/deepdetect_gpu/ respectively. See https://github.com/beniz/deepdetect/tree/master/docker/README.md for details on how to use them.

- For **Amazon AMI** see https://github.com/beniz/deepdetect/issues/5#issuecomment-199952341 and [performance report](https://github.com/beniz/deepdetect/issues/5#issuecomment-200806618)

#### Quickstart
Setup an image classifier API service in a few minutes:
http://www.deepdetect.com/tutorials/imagenet-classifier/

#### Tutorials
List of tutorials, training from text, data and images, setup of prediction services, and export to external software (e.g. ElasticSearch): http://www.deepdetect.com/tutorials/tutorials/

#### Features and Documentation
Current features include:

- high-level API for machine learning
- Support for Caffe and XGBoost
- JSON communication format
- remote Python client library
- dedicated server with support for asynchronous training calls
- high performances, benefit from multicores and GPU
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

- CUDA 7 or 6.5 is required for GPU mode.
- BLAS via ATLAS, MKL, or OpenBLAS.
- [protobuf](https://github.com/google/protobuf)
- IO libraries hdf5, leveldb, snappy, lmdb

##### XGBoost Dependencies

None outside of C++ compiler and make

##### Caffe version

By default DeepDetect automatically relies on a modified version of Caffe, https://github.com/beniz/caffe/tree/master

##### Implementation

The code makes use of C++ policy design for modularity, performance and putting the maximum burden on the checks at compile time. The implementation uses many features from C++11.

##### Demo

- Image classification Web interface:
HTML and javascript classification image demo in [demo/imgdetect](https://github.com/beniz/deepdetect/tree/master/demo/imgdetect)

- Image similarity search:
Python script for indexing and searching images is in [demo/imgsearch](https://github.com/beniz/deepdetect/tree/master/demo/imgsearch)

##### Examples

- List of examples, from MLP for data, text, multi-target regression to CNN and GoogleNet, finetuning, etc...:
http://www.deepdetect.com/overview/examples/

##### Models

- List of free, even for commercial use, deep neural nets for image classification, and character-based convolutional nets for text classification: http://www.deepdetect.com/applications/list_models/

### Authors
DeepDetect is designed and implemented by Emmanuel Benazera <beniz@droidnik.fr>.

### Build

Below are instructions for Ubuntu 14.04 LTS. For other Linux and Unix systems, steps may differ, CUDA, Caffe and other libraries may prove difficult to setup.

Beware of dependencies, typically on Debian/Ubuntu Linux, do:
```
sudo apt-get install build-essential libgoogle-glog-dev libgflags-dev libeigen3-dev libopencv-dev libcppnetlib-dev libboost-dev libboost-iostreams-dev libcurlpp-dev libcurl4-openssl-dev protobuf-compiler libopenblas-dev libhdf5-dev libprotobuf-dev libleveldb-dev libsnappy-dev liblmdb-dev libutfcpp-dev cmake
```

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

If you would like a CPU only build, use:
```
cmake .. -DUSE_CPU_ONLY=ON
```

If you would like to build with XGBoost, include the `-DUSE_XGBOOST=ON` parameter to `cmake`:
```
cmake .. -DUSE_XGBOOST=ON
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
