FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN echo 'building GPU DeepDetect image'

MAINTAINER Emmanuel Benazera "beniz@droidnik.fr"

LABEL description="DeepDetect deep learning server & API / GPU version"

RUN ln -sf /dev/stdout /var/log/deepdetect.log
RUN ln -sf /dev/stderr /var/log/deepdetect.log

RUN useradd -ms /bin/bash dd

RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  cmake \
  automake \
  build-essential \
  openjdk-8-jdk \
  pkg-config \
  zip \
  g++ \
  zlib1g-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libeigen3-dev \
  libopencv-dev \
  libcppnetlib-dev \
  libboost-dev \
  libboost-iostreams-dev \
  libcurlpp-dev \
  libcurl4-openssl-dev \
  protobuf-compiler \
  libopenblas-dev \
  libhdf5-dev \
  libprotobuf-dev \
  libleveldb-dev \
  libsnappy-dev \
  liblmdb-dev \
  libutfcpp-dev \
  wget \
  autoconf \
  libtool-bin \
  python-numpy \
  swig \
  python-dev \
  python-wheel \
  unzip \
  libgoogle-perftools-dev \
  screen \
  vim \
  strace \
  curl \
  bash-completion \
  ca-certificates && \
  wget -O /tmp/bazel.deb https://github.com/bazelbuild/bazel/releases/download/0.4.3/bazel_0.4.3-linux-x86_64.deb && \
  dpkg -i /tmp/bazel.deb && \
  apt-get remove -y libcurlpp0 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /opt
RUN wget -O /opt/curlpp-v0.8.1.zip https://github.com/jpbarrette/curlpp/archive/v0.8.1.zip && \
    unzip curlpp-v0.8.1.zip && \
    cd curlpp-0.8.1 && \
    cmake . && \
    make install && \
    cp /usr/local/lib/libcurlpp.* /usr/lib/ && \
    cd /opt && \
    rm -rf curlpp-0.8.1 curlpp-v0.8.1.zip

# Temporary work around to fix https://github.com/tensorflow/tensorflow/issues/8264
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64/ && \
    ln -s /usr/include/cudnn.h /usr/local/cuda/include/

WORKDIR /opt
RUN git clone https://github.com/beniz/deepdetect.git && cd deepdetect && mkdir build

WORKDIR /opt/deepdetect/build
RUN export TF_CUDA_VERSION=8.0 && \
    export TF_CUDA_COMPUTE_CAPABILITIES=5.2,6.1 && \
    export TF_CUDNN_VERSION=5 && \
    export TF_NEED_HDFS=0 && \
    export TF_NEED_GCP=0 && \
    export TF_NEED_OPENCL=0 && \
    export TF_NEED_CUDA=1 && \   
    export PYTHON_BIN_PATH=/usr/bin/python && \
    export PYTHON_LIB_PATH=/usr/local/lib/python2.7/dist-packages && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 && \
    export GCC_HOST_COMPILER_PATH=$(which gcc) && \
    export CUDA_TOOLKIT_PATH=/usr/local/cuda && \
    export CUDNN_INSTALL_PATH=/usr/local/cuda && \
    export TF_NEED_JEMALLOC=1 && \
    export TF_ENABLE_XLA=0 && \
    export CC_OPT_FLAGS="-march=native" && \
    cmake .. -DUSE_TF=ON -DUSE_CUDNN=ON -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN=52,61 -DCUDA_ARCH_PTX=52,61 && \
    make

# external volume to be mapped, e.g. for models or training data
RUN mkdir /data
VOLUME ["/data"]

# include a few image models within the image
RUN mkdir /opt/models
WORKDIR /opt/models
RUN mkdir ggnet && cd ggnet && wget http://www.deepdetect.com/models/ggnet/bvlc_googlenet.caffemodel
RUN mkdir resnet_50 && cd resnet_50 && wget http://www.deepdetect.com/models/resnet/ResNet-50-model.caffemodel && wget http://www.deepdetect.com/models/resnet/ResNet_mean.binaryproto
RUN cp /opt/deepdetect/datasets/imagenet/corresp_ilsvrc12.txt /opt/models/ggnet/corresp.txt
RUN cp /opt/deepdetect/datasets/imagenet/corresp_ilsvrc12.txt /opt/models/resnet_50/corresp.txt
RUN cp /opt/deepdetect/templates/caffe/googlenet/*prototxt /opt/models/ggnet/
RUN cp /opt/deepdetect/templates/caffe/resnet_50/*prototxt /opt/models/resnet_50/

# permissions
RUN chown -R dd:dd /opt/deepdetect
RUN chown -R dd:dd /opt/models

WORKDIR /opt/deepdetect/build/main
CMD ./dede -host 0.0.0.0
EXPOSE 8080