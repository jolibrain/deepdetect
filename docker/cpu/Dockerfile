FROM ubuntu:16.04

RUN echo 'building CPU DeepDetect image'

MAINTAINER Emmanuel Benazera "beniz@droidnik.fr"
LABEL description="DeepDetect deep learning server & API / CPU version"

RUN ln -sf /dev/stdout /var/log/deepdetect.log
RUN ln -sf /dev/stderr /var/log/deepdetect.log

RUN useradd -ms /bin/bash dd

RUN apt-get update && apt-get install -y git cmake build-essential libgoogle-glog-dev libgflags-dev libeigen3-dev libopencv-dev libcppnetlib-dev libboost-dev libboost-iostreams-dev libcurl4-openssl-dev protobuf-compiler libopenblas-dev libhdf5-dev libprotobuf-dev libleveldb-dev libsnappy-dev liblmdb-dev libutfcpp-dev wget unzip libspdlog-dev python-setuptools python-dev libarchive-dev

WORKDIR /opt
RUN git clone https://github.com/jpbarrette/curlpp.git
WORKDIR /opt/curlpp
RUN cmake .
RUN make install
RUN cp /usr/local/lib/libcurlpp.* /usr/lib/

WORKDIR /opt
RUN git clone https://github.com/beniz/deepdetect.git && cd deepdetect && mkdir build

WORKDIR /opt/deepdetect/build
RUN cmake .. -DUSE_XGBOOST=ON -DUSE_SIMSEARCH=ON -DUSE_TSNE=ON -DUSE_NCNN=ON
RUN make

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

USER dd
WORKDIR /opt/deepdetect/build/main
CMD ./dede -host 0.0.0.0
EXPOSE 8080