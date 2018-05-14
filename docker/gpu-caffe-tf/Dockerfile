FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

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
  libspdlog-dev \
  bash-completion \
  ca-certificates && \
  wget -O /tmp/bazel.deb https://github.com/bazelbuild/bazel/releases/download/0.8.1/bazel_0.8.1-linux-x86_64.deb && \
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
RUN if [ ! -f /usr/local/cuda/include/cudnn.h -a -f /usr/include/cudnn.h ]; then ln -s /usr/include/cudnn.h /usr/local/cuda/include/; fi && \
    if [ ! -f /usr/local/cuda/lib64/libcudnn.so -a -f /usr/lib/x86_64-linux-gnu/libcudnn.so ]; then ln -s /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64/; fi

WORKDIR /opt
RUN git clone https://github.com/beniz/deepdetect.git && cd deepdetect && mkdir build

WORKDIR /opt/deepdetect/build
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
RUN echo "/usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs" > /etc/ld.so.conf.d/cuda-8.0-stubs.conf && ldconfig
RUN cmake .. -DUSE_TF=ON -DUSE_CUDNN=ON && \ 
    make

# external volume to be mapped, e.g. for models or training data
RUN mkdir /data
VOLUME ["/data"]

# include a few image models within the image
RUN mkdir /opt/models
WORKDIR /opt/models
RUN mkdir ggnet && cd ggnet && wget http://www.deepdetect.com/models/ggnet/bvlc_googlenet.caffemodel
RUN mkdir resnet_50 && cd resnet_50 && wget http://www.deepdetect.com/models/resnet/ResNet-50-model.caffemodel && wget http://www.deepdetect.com/models/resnet/ResNet_mean.binaryproto
RUN mv /opt/models/resnet_50/ResNet_mean.binaryproto /opt/models/resnet_50/mean.binaryproto
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