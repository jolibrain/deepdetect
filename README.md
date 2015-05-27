## deepdetect
DeepDetect (http://www.deepdetect.com/) is a machine learning API and server written in C++11. It makes state of the art machine learning easy to work with and integrate into existing applications.

DeepDetect relies on external machine learning libraries through a very generic and flexible API. At the moment it has support for the deep learning library [Caffe](https://github.com/BVLC/caffe).

Main functionalities:
DeepDetect implements support for supervised deep learning of images and other data, with focus on simplicity and ease of use, test and connection into existing applications.

Current features include:

- high-level API for machine learning
- JSON commnunication format
- dedicated server with support for asynchronous training calls
- high performances, benefit from multicores and GPU
- connector to handle large collections of images
- connector to handle CSV files with preprocessing capabilities
- range of built-in model assessment measures (e.g. F1, multiclass log loss, ...)
- no database dependency and sync, all information organized on the filesystem
- flexible template output format to simplify connection to external applications
- templates for the most useful neural architectures (e.g. Googlenet, Alexnet, mlp, logistic regression)

Documentation:
- Full documentation is available from http://www.deepdetect.com/overview/introduction/
- API documentation is available from http://www.deepdetect.com/api/

Dependencies:

- C++, gcc >= 4.9 or clang with support for C++11 (there are issues with Clang + Boost)
- [eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for all matrix operations;
- [glog](https://code.google.com/p/google-glog/) for logging events and debug;
- [gflags](https://code.google.com/p/gflags/) for command line parsing;
- OpenCV >= 2.4
- [cppnetlib](http://cpp-netlib.org/)
- Boost
- [curl](http://curl.haxx.se/)
- [curlpp](http://www.curlpp.org/)
- [gtest](https://code.google.com/p/googletest/) for unit testing (optional);

Caffe Dependencies:

- CUDA 7 or 6.5 is required for GPU mode.
- BLAS via ATLAS, MKL, or OpenBLAS.
- [protobuf](https://github.com/google/protobuf)
- IO libraries hdf5, leveldb, snappy, lmdb


Implementation:
The code makes use of C++ policy design for modularity, performance and putting the maximum burden on the checks at compile time. The implementation uses many features from C++11.

### Authors
DeepDetect is designed and implemented by Emmanuel Benazera <beniz@droidnik.fr>.

### Build

Below are instructions for Linux systems.

Beware of dependencies, typically on Debian/Ubuntu Linux, do:
```
sudo apt-get install build-essential libgoogle-glog-dev libgflags-dev libeigen3-dev libopencv-dev libcppnetlib-dev libboost-dev libcurlpp-dev libcurl4-openssl-dev protobuf-compiler libopenblas-dev libhdf5-dev libprotobuf-dev libleveldb-dev libsnappy-dev liblmdb-dev
```

For compiling along with Caffe:
```
mkdir build
cmake ..
make
```

### Run tests

Note: running tests requires the automated download of ~75Mb of datasets, and computations may take around thirty minutes on a CPU-only machines.

To prepare for tests, compile with:
```
cmake -DBUILD_TESTS=ON ..
make
```

### Start the server

```
cd build/main
./dede

DeepDetect [ commit 73d4e638498d51254862572fe577a21ab8de2ef1 ]
Running DeepDetect HTTP server on localhost:8080
```

### Run examples

See tutorials from 

### References

- DeepDetect (http://www.deepdetect.com/)
- Caffe (https://github.com/BVLC/caffe)