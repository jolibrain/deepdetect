FROM nvidia/cuda:8.0-cudnn7-devel

RUN echo 'building GPU DeepDetect image'

MAINTAINER Emmanuel Benazera "beniz@droidnik.fr"

LABEL description="DeepDetect deep learning server & API / GPU version"

RUN ln -sf /dev/stdout /var/log/deepdetect.log
RUN ln -sf /dev/stderr /var/log/deepdetect.log

RUN useradd -ms /bin/bash dd

RUN apt-get update && apt-get install -y git cmake build-essential libgoogle-glog-dev libgflags-dev libeigen3-dev libopencv-dev libcppnetlib-dev libboost-dev libboost-iostreams-dev libcurl4-openssl-dev protobuf-compiler libopenblas-dev libhdf5-dev libprotobuf-dev libleveldb-dev libsnappy-dev liblmdb-dev libutfcpp-dev wget unzip libspdlog-dev python-setuptools python-dev python-six python-enum34

RUN git config --global user.email "you@example.com" && git config --global user.name "Your Name"

WORKDIR /opt
RUN git clone https://github.com/jpbarrette/curlpp.git
WORKDIR /opt/curlpp
RUN cmake .
RUN make install
RUN cp /usr/local/lib/libcurlpp.* /usr/lib/

WORKDIR /opt
RUN git clone https://github.com/beniz/deepdetect.git && cd deepdetect && mkdir build

WORKDIR /opt/deepdetect/build
RUN cmake .. -DUSE_CUDNN=ON -DUSE_XGBOOST=ON -DUSE_SIMSEARCH=ON -DUSE_CAFFE2=ON -DCUDA_ARCH="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_61,code=sm_61"
RUN make

# external volume to be mapped, e.g. for models or training data
RUN mkdir /data
VOLUME ["/data"]

# include a few image models within the image
RUN mkdir /opt/models && mkdir /opt/models/caffe && mkdir /opt/models/caffe2
WORKDIR /opt/models/caffe
RUN mkdir ggnet && cd ggnet && wget http://www.deepdetect.com/models/ggnet/bvlc_googlenet.caffemodel
RUN mkdir resnet_50 && cd resnet_50 && wget http://www.deepdetect.com/models/resnet/ResNet-50-model.caffemodel && wget http://www.deepdetect.com/models/resnet/ResNet_mean.binaryproto
RUN mv /opt/models/caffe/resnet_50/ResNet_mean.binaryproto /opt/models/caffe/resnet_50/mean.binaryproto
RUN cp /opt/deepdetect/datasets/imagenet/corresp_ilsvrc12.txt /opt/models/caffe/ggnet/corresp.txt
RUN cp /opt/deepdetect/datasets/imagenet/corresp_ilsvrc12.txt /opt/models/caffe/resnet_50/corresp.txt
RUN cp /opt/deepdetect/templates/caffe/googlenet/*prototxt /opt/models/caffe/ggnet/
RUN cp /opt/deepdetect/templates/caffe/resnet_50/*prototxt /opt/models/caffe/resnet_50/
WORKDIR /opt/models/caffe2
RUN mkdir resnet_50 && cd resnet_50 && wget http://www.deepdetect.com/dd/examples/caffe2/resnet_50_imagenet/init_net.pb && wget http://www.deepdetect.com/dd/examples/caffe2/resnet_50_imagenet/predict_net.pb && wget http://www.deepdetect.com/dd/examples/caffe2/resnet_50_imagenet/corresp.txt && cd ..
RUN mkdir detectron && cd detectron && wget http://www.deepdetect.com/dd/examples/caffe2/detectron/init_net.pb && wget http://www.deepdetect.com/dd/examples/caffe2/detectron/predict_net.pb && wget http://www.deepdetect.com/dd/examples/caffe2/detectron/corresp.txt

# permissions
RUN chown -R dd:dd /opt/deepdetect
RUN chown -R dd:dd /opt/models

WORKDIR /opt/deepdetect/build/main
CMD ./dede -host 0.0.0.0
EXPOSE 8080