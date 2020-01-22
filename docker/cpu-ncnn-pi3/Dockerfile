FROM armv7/armhf-ubuntu:16.04

RUN echo 'Building CPU NCNN RPi3 DeepDetect image'

MAINTAINER Emmanuel Benazera "emmanuel.benazera@jolibrain.com"
LABEL description="DeepDetect deep learning server & API / CPU NCNN-only RPi3 version"

RUN ln -sf /dev/stdout /var/log/deepdetect.log
RUN ln -sf /dev/stderr /var/log/deepdetect.log

RUN useradd -ms /bin/bash dd

RUN apt-get update

RUN apt-get install -y git cmake build-essential libgoogle-glog-dev libgflags-dev libeigen3-dev libopencv-dev libcppnetlib-dev libboost-dev libboost-iostreams-dev libcurl4-openssl-dev protobuf-compiler libopenblas-dev libprotobuf-dev libleveldb-dev libsnappy-dev liblmdb-dev libutfcpp-dev wget unzip libspdlog-dev python-setuptools python-dev libhdf5-dev libarchive-dev

WORKDIR /opt
RUN git clone https://github.com/jpbarrette/curlpp.git
WORKDIR /opt/curlpp
RUN cmake .
RUN make install
RUN cp /usr/local/lib/libcurlpp.* /usr/lib/

WORKDIR /opt
RUN echo 'Building DeepDetect'
RUN git clone https://github.com/jolibrain/deepdetect.git && cd deepdetect && mkdir build

WORKDIR /opt/deepdetect/build
RUN cmake .. -DUSE_NCNN=ON -DRPI3=ON -DUSE_HDF5=OFF -DUSE_CAFFE=OFF
RUN make

# external volume to be mapped, e.g. for models or training data
RUN mkdir /data
RUN mkdir /opt/models
VOLUME ["/data"]

# permissions
RUN chown -R dd:dd /opt/deepdetect
RUN chown -R dd:dd /opt/models

USER dd
WORKDIR /opt/deepdetect/build/main
CMD ./dede -host 0.0.0.0
EXPOSE 8080