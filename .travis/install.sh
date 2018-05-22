#!/bin/bash
set -e
set -x

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$(dirname "$LOCAL_DIR")
cd "$ROOT_DIR"

APT_INSTALL_CMD='sudo apt-get install -y --no-install-recommends'

if [ "$TRAVIS_OS_NAME" = 'linux' ]; then
    ####################
    # apt dependencies #
    ####################
    sudo apt-get update
    $APT_INSTALL_CMD \
	build-essential \
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
	libutfcpp-dev

        # Install ccache symlink wrappers
        pushd /usr/local/bin
        sudo ln -sf "$(which ccache)" gcc
        sudo ln -sf "$(which ccache)" g++
        popd

        ################
        # Install GCC5 #
        ################
        sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
        sudo apt-get update
        $APT_INSTALL_CMD g++-4.9
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 60 \
            --slave /usr/bin/g++ g++ /usr/bin/g++-4.9

	##################
        # Install spdlog #
        ##################
        # Not available on 14.04 trusty via package
	git clone https://github.com/gabime/spdlog.git
	sudo cp -r spdlog/include/spdlog /usr/include/
	
        if [ "$BUILD_CUDA" = 'true' ]; then
        ##################
        # Install ccache #
        ##################
        # Needs specific branch to work with nvcc (ccache/ccache#145)
        if [ -e "${BUILD_CCACHE_DIR}/ccache" ]; then
            echo "Using cached ccache build at \"$BUILD_CCACHE_DIR\" ..."
        else
            git clone https://github.com/colesbury/ccache -b ccbin "$BUILD_CCACHE_DIR"
            pushd "$BUILD_CCACHE_DIR"
            ./autogen.sh
            ./configure
            make "-j$(nproc)"
            popd
        fi

        # Overwrite ccache symlink wrappers
        pushd /usr/local/bin
        sudo ln -sf "${BUILD_CCACHE_DIR}/ccache" gcc
        sudo ln -sf "${BUILD_CCACHE_DIR}/ccache" g++
        sudo ln -sf "${BUILD_CCACHE_DIR}/ccache" nvcc
        popd

        #################
        # Install CMake #
        #################
        # Newer version required to get cmake+ccache+nvcc to work
        _cmake_installer=/tmp/cmake.sh
        wget -O "$_cmake_installer" https://cmake.org/files/v3.8/cmake-3.8.2-Linux-x86_64.sh
        sudo bash "$_cmake_installer" --prefix=/usr/local --skip-license
        rm -rf "$_cmake_installer"
	
        ################
        # Install CUDA #
        ################
        CUDA_REPO_PKG='cuda-repo-ubuntu1404_8.0.44-1_amd64.deb'
        CUDA_PKG_VERSION='8-0'
        CUDA_VERSION='8.0'
        wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/${CUDA_REPO_PKG}"
        sudo dpkg -i "$CUDA_REPO_PKG"
        rm -f "$CUDA_REPO_PKG"
        sudo apt-get update
        $APT_INSTALL_CMD \
            "cuda-core-${CUDA_PKG_VERSION}" \
            "cuda-cublas-dev-${CUDA_PKG_VERSION}" \
            "cuda-cudart-dev-${CUDA_PKG_VERSION}" \
            "cuda-curand-dev-${CUDA_PKG_VERSION}" \
            "cuda-driver-dev-${CUDA_PKG_VERSION}" \
            "cuda-nvrtc-dev-${CUDA_PKG_VERSION}" \
	    "cuda-cusparse-dev-${CUDA_PKG_VERSION}"
        # Manually create CUDA symlink
        sudo ln -sf /usr/local/cuda-$CUDA_VERSION /usr/local/cuda

        #################
        # Install cuDNN #
        #################
        CUDNN_REPO_PKG='nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb'
        CUDNN_PKG_VERSION='6.0.20-1+cuda8.0'
        wget "https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/${CUDNN_REPO_PKG}"
        sudo dpkg -i "$CUDNN_REPO_PKG"
        rm -f "$CUDNN_REPO_PKG"
        sudo apt-get update
        $APT_INSTALL_CMD \
            "libcudnn6=${CUDNN_PKG_VERSION}" \
            "libcudnn6-dev=${CUDNN_PKG_VERSION}"
    fi
else
    echo "OS \"$TRAVIS_OS_NAME\" is unknown"
    exit 1
fi

####################
# pip dependencies #
####################
## As needed
