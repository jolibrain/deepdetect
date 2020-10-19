# Dependencies

- C++, gcc >= 4.8 or clang with support for C++11 (there are issues with Clang + Boost)
- [eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for all matrix operations;
- [glog](https://code.google.com/p/google-glog/) for logging events and debug;
- [gflags](https://code.google.com/p/gflags/) for command line parsing;
- OpenCV >= 2.4
- [cppnetlib](http://cpp-netlib.org/)
- Boost , Boost::graph
- [curl](http://curl.haxx.se/)
- [curlpp](http://www.curlpp.org/)
- [utfcpp](http://utfcpp.sourceforge.net/)
- [gtest](https://code.google.com/p/googletest/) for unit testing (optional);

## Caffe Dependencies

- CUDA 9 or 8 is recommended for GPU mode.
- BLAS via ATLAS, MKL, or OpenBLAS.
- [protobuf](https://github.com/google/protobuf)
- IO libraries hdf5, leveldb, snappy, lmdb

## XGBoost Dependencies

None outside of C++ compiler and make
- CUDA 8 is recommended for GPU mode.

## Tensorflow Dependencies

- Cmake > 3
- [Bazel 0.8.x](https://www.bazel.io/versions/master/docs/install.html#install-on-ubuntu)

## Dlib Dependencies

- CUDA 9 or 8 and cuDNN 7 for GPU mode. CUDA 10 for Ubuntu 18.04
  **Note:** The version of OpenBLAS (v0.2.20) shipped with Ubuntu 18.04 is not up to date and includes a bug. You must install a later version of OpenBLAS >= v0.3.0 to use Dlib on Ubuntu 18.04.

  The easiest way currently is to manually install the Ubuntu 19.10 `libopenblas-base` and `libopenblas-dev` packages. You may download them here:
  http://launchpadlibrarian.net/410583809/libopenblas-base_0.3.5+ds-2_amd64.deb
  http://launchpadlibrarian.net/410583808/libopenblas-dev_0.3.5+ds-2_amd64.deb
  and install them with `sudo apt-get install ./package-name.deb` to automatically handle dependencies.

## Caffe version

By default DeepDetect automatically relies on a modified version of Caffe, https://github.com/jolibrain/caffe/tree/master
This version includes many improvements over the original Caffe, such as sparse input data support, exception handling, class weights, object detection, segmentation, and various additional losses and layers.

We use Caffe as default since it has excellent conversion to production inference libraries such as TensorRT and NCNN, that are also supported by DeepDetect.

## Implementation

The code makes use of C++ policy design for modularity, performance and putting the maximum burden on the checks at compile time. The implementation uses many features from C++11.
