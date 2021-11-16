# syntax = docker/dockerfile:1.0-experimental
FROM nvcr.io/nvidia/tensorrt:21.07-py3 AS build

ARG DEEPDETECT_RELEASE=OFF
ARG DEEPDETECT_ARCH=gpu
ARG DEEPDETECT_BUILD=tensorrt
ARG DEEPDETECT_DEFAULT_MODELS=true
ARG DEEPDETECT_OPENCV4_BUILD_PATH=/opt/deepdetect/opencv/opencv-4.5.3/build
#ARG DEEPDETECT_OPENCV4_BUILD_PATH=/tmp

# Install build dependencies
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,id=dede_cache_lib,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=dede_apt_lib,sharing=locked,target=/var/lib/apt \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y python-dev apt-transport-https ca-certificates gnupg software-properties-common wget curl

# CMake
RUN curl https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'

# python2 pycompile + docker-buildkit is a bit buggy, it's slow as hell, so disable it for dev
# bug closed as won't fix as python2 is eol: https://github.com/docker/for-linux/issues/502
RUN cp /bin/true /usr/bin/pycompile

# Don't install opencv-ml-dev, it will install libprotobuf dans link dede to 2 versions of protobuf
RUN --mount=type=cache,id=dede_cache_lib,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=dede_apt_lib,sharing=locked,target=/var/lib/apt \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y \
    git \
    ccache \
    automake \
    build-essential \
    openjdk-8-jdk \
    pkg-config \
    cmake \
    zip \
    g++ \
    gcc-7 g++-7 \
    zlib1g-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libeigen3-dev \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-thread-dev \
    libboost-system-dev \
    libboost-iostreams-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libboost-regex-dev \
    libboost-date-time-dev \
    libboost-chrono-dev \
    libboost-stacktrace-dev \
    libssl-dev \
    libcurlpp-dev \
    libcurl4-openssl-dev \
    libopenblas-dev \
    libhdf5-dev \
    libleveldb-dev \
    libsnappy-dev \
    liblmdb-dev \
    libutfcpp-dev \
    rapidjson-dev \
    libmapbox-variant-dev \
    autoconf \
    libtool-bin \
    python-numpy \
    python-yaml \
    python-typing \
    swig \
    curl \
    unzip \
    python-setuptools \
    python-dev \
    python3-dev \
    python3-pip \
    python-six \
    python-enum34 \
    python3-yaml \
    unzip \
    libgoogle-perftools-dev \
    curl \
    libarchive-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-gl \
    bash-completion

RUN for url in \
        https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel_0.24.1-linux-x86_64.deb \
        ; do curl -L -s -o /tmp/p.deb $url && dpkg -i /tmp/p.deb && rm -rf /tmp/p.deb; done

# Fix "ImportError: No module named builtins"
# RUN pip install future pyyaml typing

# Fix  ModuleNotFoundError: No module named 'dataclasses', 'typing_extensions' for torch 1.8.0
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install dataclasses typing_extensions

ADD . /opt/deepdetect
WORKDIR /opt/deepdetect/

ENV CCACHE_DIR=/ccache
ENV PATH=/usr/lib/ccache:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Install NVidia video codec
RUN wget http://www.deepdetect.com/stuff/Video_Codec_SDK_11.1.5.zip && unzip Video_Codec_SDK_11.1.5.zip
RUN cd Video_Codec_SDK_11.1.5 && cp Interface/* /usr/local/cuda/targets/x86_64-linux/include/ && cp Lib/linux/stubs/x86_64/* /usr/local/cuda/targets/x86_64-linux/lib/stubs/

# Build OpenCV 4 with CUDA
RUN mkdir opencv && cd opencv && wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.5.3.zip && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.3.zip && unzip opencv.zip && unzip opencv_contrib.zip
RUN cd /opt/deepdetect/opencv/opencv-4.5.3 && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/tmp/ \
-D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
-D CMAKE_CXX_FLAGS="-Wl,--allow-shlib-undefined" \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=OFF \
-D CUDA_ARCH_BIN=6.1 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D WITH_NVCUVID=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_opencv_python2=OFF \
-D BUILD_opencv_python3=OFF \
-D OPENCV_EXTRA_MODULES_PATH=/opt/deepdetect/opencv/opencv_contrib-4.5.3/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF ..

WORKDIR /opt/deepdetect/opencv/opencv-4.5.3/build
RUN make -j20
RUN make install

# Build Deepdetect
WORKDIR /opt/deepdetect
ENV TERM=xterm
RUN --mount=type=cache,target=/ccache/ mkdir build && cd build && ../build.sh

# Copy libs to /tmp/libs for next build stage
RUN ./docker/get_libs.sh

# Build final Docker image
FROM nvcr.io/nvidia/tensorrt:21.07-py3 AS runtime

ARG DEEPDETECT_ARCH=gpu

LABEL description="DeepDetect deep learning server & API / ${DEEPDETECT_ARCH} version"
LABEL maintainer="emmanuel.benazera@jolibrain.com"

# Install tools and dependencies
RUN --mount=type=cache,id=dede_cache_lib,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=dede_apt_lib,sharing=locked,target=/var/lib/apt \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y \
    wget \
    curl \
	libopenblas-base \
	liblmdb0 \
	libleveldb1d \
    libboost-regex1.71.0 \
	libgoogle-glog0v5 \
    libopencv4.2 \
	libgflags2.2 \
	libcurl4 \
	libcurlpp0 \
	libhdf5-cpp-103 \
    libboost-atomic1.71.0 \
    libboost-chrono1.71.0 \
    libboost-date-time1.71.0 \
	libboost-filesystem1.71.0 \
	libboost-thread1.71.0 \
	libboost-iostreams1.71.0 \
    libboost-regex1.71.0 \
    libboost-stacktrace1.71.0 \
    libboost-system1.71.0 \
	libarchive13 \
	libgstreamer1.0-0

# Fix permissions
RUN ln -sf /dev/stdout /var/log/deepdetect.log && \
    ln -sf /dev/stderr /var/log/deepdetect.log

RUN useradd -ms /bin/bash dd && \
    chown -R dd:dd /opt
USER dd

# Copy Deepdetect binaries from previous step
COPY --from=build /opt/deepdetect/build/main /opt/deepdetect/build/main
COPY --from=build /opt/deepdetect/build/oatpp-swagger/src/oatpp-swagger/res/ /opt/deepdetect/build/oatpp-swagger/src/oatpp-swagger/res/
COPY --from=build --chown=dd /opt/deepdetect/datasets/imagenet/corresp_ilsvrc12.txt /opt/models/ggnet/corresp.txt
COPY --from=build --chown=dd /opt/deepdetect/datasets/imagenet/corresp_ilsvrc12.txt /opt/models/resnet_50/corresp.txt
COPY --from=build --chown=dd /opt/deepdetect/templates/caffe/googlenet/*prototxt /opt/models/ggnet/
COPY --from=build --chown=dd /opt/deepdetect/templates/caffe/resnet_50/*prototxt /opt/models/resnet_50/
COPY --from=build /tmp/lib/* /opt/deepdetect/build/lib/
COPY --from=build /opt/deepdetect/templates /opt/deepdetect/build/templates

COPY --from=build /opt/deepdetect/get_models.sh /opt/deepdetect/
COPY --from=build /opt/deepdetect/docker/check-dede-deps.sh /opt/deepdetect/
COPY --from=build /opt/deepdetect/docker/start-dede.sh /opt/deepdetect/

# External volume to be mapped, e.g. for models or training data
WORKDIR /opt/models

USER root
RUN chown -R dd:dd /opt/models

USER dd
RUN /opt/deepdetect/get_models.sh

# Ensure all libs are presents
RUN /opt/deepdetect/check-dede-deps.sh

WORKDIR /opt/deepdetect/build/main
CMD /opt/deepdetect/start-dede.sh -host 0.0.0.0
VOLUME ["/data"]
EXPOSE 8080
