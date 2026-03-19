#!/bin/bash

set -e

# Deepdetect architecture and build profiles
deepdetect_arch=(cpu gpu jetson)
deepdetect_cpu_build_profiles=(default tf armv7)
deepdetect_gpu_build_profiles=(default tf caffe2 tensorrt)
deepdetect_gpu_variants=(default legacy61)

DEEPDETECT_GPU_VARIANT=${DEEPDETECT_GPU_VARIANT:-default}
DEEPDETECT_JETSON_ARCH="8.7" # Orin only

DEEPDETECT_RELEASE=${DEEPDETECT_RELEASE:-OFF}
BUILD_TESTS=${BUILD_TESTS:-OFF}

build_cuda_arch_flags() {
    local flags=
    local card=

    for card in "$@"; do
        flags="${flags} -gencode arch=compute_${card},code=sm_${card}"
    done

    echo "${flags}" | xargs
}

configure_gpu_variant() {
    local default_cuda_arch=
    local default_cuda_arch_cards=

    # Keep legacy61 aligned with the current default while CUDA 12.x remains
    # the main toolchain. This lets us preserve the interface for the later
    # CUDA 13 split without changing today's build outputs.
    case "${DEEPDETECT_GPU_VARIANT}" in
    "default"|"legacy61")
        if [ "${DEEPDETECT_BUILD}" = "tensorrt" ]; then
            default_cuda_arch="6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9;12.0;12.1"
            default_cuda_arch_cards="61 62 70 72 75 80 86 89 120 121"
        else
            default_cuda_arch="6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9;12.0"
            default_cuda_arch_cards="61 62 70 72 75 80 86 89 120"
        fi
        ;;
    *)
        echo "Unknown DEEPDETECT_GPU_VARIANT value: ${DEEPDETECT_GPU_VARIANT}"
        echo "Supported GPU variants: ${deepdetect_gpu_variants[*]}"
        exit 1
        ;;
    esac

    DEEPDETECT_CUDA_ARCH=${DEEPDETECT_CUDA_ARCH:-${default_cuda_arch}}
    DEEPDETECT_CUDA_ARCH_FLAGS=${DEEPDETECT_CUDA_ARCH_FLAGS:-$(build_cuda_arch_flags ${default_cuda_arch_cards})}
    DEEPDETECT_OPENCV_CUDA_ARCH_BIN=${DEEPDETECT_OPENCV_CUDA_ARCH_BIN:-${DEEPDETECT_CUDA_ARCH//;/ }}
}

validate_gpu_variant() {
    local cuda_major=

    if [ "${DD_CUDA_VERSION}" ]; then
        cuda_major=${DD_CUDA_VERSION%%.*}
    fi

    case ";${DEEPDETECT_CUDA_ARCH};" in
    *";12.1;"*)
        if [ "${DEEPDETECT_BUILD}" != "tensorrt" ]; then
            echo "Compute capability 12.1 is only enabled for DEEPDETECT_BUILD=tensorrt"
            exit 1
        fi
        if [ "${cuda_major}" ] && [ "${cuda_major}" -lt 13 ] 2>/dev/null; then
            echo "Compute capability 12.1 requires DD_CUDA_VERSION >= 13, got ${DD_CUDA_VERSION}"
            exit 1
        fi
        ;;
    esac
}

# Help menu with arguments descriptions
help_menu() {
    IFS=,
    echo "Deepdetect build script"
    echo ""
    echo "Env variables usage: DEEPDETECT_ARCH=${deepdetect_arch[*]} DEEPDETECT_BUILD=${deepdetect_cpu_build_profiles[*]},[...] $0 "
    echo "GPU only: DEEPDETECT_GPU_VARIANT=${deepdetect_gpu_variants[*]}"
    echo ""
    echo "or"
    echo ""
    echo "Params usage: $0 [options...]" >&2
    echo
    echo "   -a, --deepdetect-arch          Choose Deepdetect architecture : ${deepdetect_arch[*]}"
    echo "   -b, --deepdetect-build         Choose Deepdetect build profile : CPU (${deepdetect_cpu_build_profiles[*]}) / GPU (${deepdetect_gpu_build_profiles[*]})"
    echo "   -c, --deepdetect-cuda-arch     Choose Deepdetect cuda arch (default: ${DEEPDETECT_CUDA_ARCH})"
    echo "   -g, --deepdetect-gpu-variant   Choose Deepdetect GPU variant : ${deepdetect_gpu_variants[*]}"
    echo "   -t, --build-tests              Build unit tests (ON/OFF)"
    echo
    exit 1
}

# Parse arguments
while (("$#")); do
    case "$1" in
    -a | --deepdetect-arch)
        DEEPDETECT_ARCH=$2
        shift 2
        ;;
    -b | --deepdetect-build)
        DEEPDETECT_BUILD=$2
        shift 2
        ;;
    -c | --deepdetect-cuda-arch)
        DEEPDETECT_CUDA_ARCH=$2
        shift 2
        ;;
    -g | --deepdetect-gpu-variant)
        DEEPDETECT_GPU_VARIANT=$2
        shift 2
        ;;
    -t | --build-tests)
        BUILD_TESTS=$2
        shift 2
        ;;
    -h | --help)
        help_menu
        break
        ;;
    esac
done

# Check arguments for incompatibility
if [[ -v DEEPDETECT_OPENCV4_BUILD_PATH ]] && [[ -v BUILD_OPENCV ]]; then
	echo "Please choose between setting DEEPDETECT_OPENCV4_BUILD_PATH or BUILD_OPENCV"
	echo "Use DEEPDETECT_OPENCV4_BUILD_PATH if you wish to use a custom opencv build"
	echo "Use BUILD_OPENCV if you wish for opencv to be built for you"
	exit
fi

configure_gpu_variant
validate_gpu_variant

# Deepdetect platform selector
select_platform() {
    clear
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo " Select Deepdetect build platform"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    for index in ${!deepdetect_arch[*]}; do
        echo "  $index. ${deepdetect_arch[$index]^^}"
    done
    echo ""

    local choice
    read -p "Enter choice : " choice
    DEEPDETECT_ARCH=${deepdetect_arch[$choice]}
}

# Deepdetect select CPU build profile
select_cpu_build_profile() {
    clear
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo " Select Deepdetect build profile"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    for index in ${!deepdetect_cpu_build_profiles[*]}; do
        echo "  $index. ${deepdetect_cpu_build_profiles[$index]}"
    done
    echo ""

    local choice
    read -p "Enter choice : " choice
    DEEPDETECT_BUILD=${deepdetect_cpu_build_profiles[$choice]}
}
# Deepdetect select GPU build profile
select_gpu_build_profile() {
    clear
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo " Select Deepdetect build profile"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    for index in ${!deepdetect_gpu_build_profiles[*]}; do
        echo "  $index. ${deepdetect_gpu_build_profiles[$index]}"
    done
    echo ""

    local choice
    read -p "Enter choice : " choice
    DEEPDETECT_BUILD=${deepdetect_gpu_build_profiles[$choice]}
}

# Use menu...
show_interactive_platform_selector() {
    # Main menu handler loop
    select_platform
    if [ ${DEEPDETECT_ARCH} == "cpu" ]; then
        select_cpu_build_profile
    fi
    if [ ${DEEPDETECT_ARCH} == "gpu" ] || ${DEEPDETECT_ARCH} == "jetson" ]; then
        select_gpu_build_profile
    fi
}

if [[ ${BUILD_OPENCV} == "ON" ]]; then
echo "Opencv will be built from source"
# Build OpenCV 4 with CUDA
DEEPDETECT_OPENCV4_BUILD_PATH="$(git rev-parse --show-toplevel)/build/opencv/opencv-4.10.0/build"
if [ ! -d opencv ]; then
echo "Downloading opencv"
mkdir opencv && cd opencv && wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.10.0.zip && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.10.0.zip && unzip opencv.zip && unzip opencv_contrib.zip
cd "opencv-4.10.0" && mkdir build && cd build
else
cd $DEEPDETECT_OPENCV4_BUILD_PATH
fi
cmake -D CMAKE_BUILD_TYPE=DEBUG \
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
-D CUDA_ARCH_BIN="${DEEPDETECT_OPENCV_CUDA_ARCH_BIN}" \
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
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.10.0/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF \
-D BUILD_opencv_cudev=ON \
-D BUILD_opencv_core=ON \
-D BUILD_opencv_imgproc=ON \
-D BUILD_opencv_imgcodecs=ON \
-D BUILD_opencv_videoio=ON \
-D BUILD_opencv_highgui=ON ..

make -j8
make install
cd ../../..
else
echo "Using custom opencv"
fi

# Build functions
cpu_build() {

    case ${DEEPDETECT_BUILD} in

    "tf")
        cmake .. -DUSE_TF=ON -DUSE_TF_CPU_ONLY=ON -DUSE_SIMSEARCH=ON -DUSE_TSNE=ON -DUSE_NCNN=OFF -DUSE_CPU_ONLY=ON -DRELEASE=${DEEPDETECT_RELEASE} -DBUILD_TESTS=${BUILD_TESTS} -DOpenCV_DIR=${DEEPDETECT_OPENCV4_BUILD_PATH}
        make -j6
        ;; 

    "armv7")
        cmake .. -DUSE_NCNN=ON -DRPI3=ON -DUSE_HDF5=OFF -DUSE_TORCH=OFF -DRELEASE=${DEEPDETECT_RELEASE} -DBUILD_TESTS=${BUILD_TESTS} -DOpenCV_DIR=${DEEPDETECT_OPENCV4_BUILD_PATH}
        make -j6
        ;; 

    *)
        cmake .. -DUSE_XGBOOST=OFF -DUSE_CAFFE=OFF -DUSE_CPU_ONLY=ON -DUSE_SIMSEARCH=OFF -DUSE_TSNE=OFF -DUSE_NCNN=OFF -DRELEASE=${DEEPDETECT_RELEASE} -DBUILD_TESTS=${BUILD_TESTS} -DOpenCV_DIR=${DEEPDETECT_OPENCV4_BUILD_PATH}
        make -j6
        ;; 
    esac

}

gpu_build() {
    local extra_flags=
    local default_flags="-DUSE_FAISS=OFF -DUSE_CUDNN=ON -DUSE_XGBOOST=OFF -DUSE_SIMSEARCH=OFF -DUSE_TSNE=OFF -DUSE_TORCH=ON -DUSE_OPENCV_VERSION=4 -DOpenCV_DIR=${DEEPDETECT_OPENCV4_BUILD_PATH}"

    case ${DEEPDETECT_BUILD} in
        "tf") extra_flags="$default_flags -DUSE_TF=ON" ;; 
        "caffe2") extra_flags="$default_flags -DUSE_CAFFE2=ON" ;; 
        "tensorrt") extra_flags="-DUSE_TENSORRT=ON -DUSE_TORCH=OFF -DUSE_CUDA_CV=ON -DUSE_OPENCV_VERSION=4 -DOpenCV_DIR=${DEEPDETECT_OPENCV4_BUILD_PATH}";;
        *) extra_flags="$default_flags";;
    esac
    echo $extra_flags

    cmake .. $extra_flags -DCUDA_ARCH_FLAGS="${DEEPDETECT_CUDA_ARCH_FLAGS}" -DCUDA_ARCH="${DEEPDETECT_CUDA_ARCH}" -DRELEASE="${DEEPDETECT_RELEASE}" -DBUILD_TESTS=${BUILD_TESTS} -DOpenCV_DIR="${DEEPDETECT_OPENCV4_BUILD_PATH}"
    make -j6
}

jetson_build() {
    local default_flags="-DUSE_FAISS=OFF -DUSE_XGBOOST=OFF -DUSE_SIMSEARCH=OFF -DUSE_TSNE=OFF -DUSE_TORCH=OFF -DJETSON=ON -DUSE_TENSORRT=ON -DUSE_CUDA_CV=ON -DUSE_OPENCV_VERSION=4 -DOpenCV_DIR=${DEEPDETECT_OPENCV4_BUILD_PATH} -DBUILD_TESTS=${BUILD_TESTS}"

    cmake .. $default_flags -DCUDA_ARCH="${DEEPDETECT_JETSON_ARCH}" -DRELEASE="${DEEPDETECT_RELEASE}"
    make -j6
}


# If no arguments provided, display usage information
if [ -z "$DEEPDETECT_ARCH" ] && [ $# -eq 0 ]; then
    show_interactive_platform_selector
fi

# Select build profile
if [[ ${DEEPDETECT_ARCH} == "cpu" ]]; then
    echo ""
    echo "Deepdetect build params :"
    echo "  DEEPDETECT_ARCH      : ${DEEPDETECT_ARCH}"
    echo "  DEEPDETECT_BUILD     : ${DEEPDETECT_BUILD}"
    echo ""
    cpu_build
elif [[ ${DEEPDETECT_ARCH} == "gpu" ]]; then
    echo ""
    echo "Deepdetect build params :"
    echo "  DEEPDETECT_ARCH      : ${DEEPDETECT_ARCH}"
    echo "  DEEPDETECT_BUILD     : ${DEEPDETECT_BUILD}"
    echo "  DEEPDETECT_GPU_VARIANT : ${DEEPDETECT_GPU_VARIANT}"
    echo "  DEEPDETECT_CUDA_ARCH : ${DEEPDETECT_CUDA_ARCH}"
    echo "  DEEPDETECT_CUDA_ARCH_FLAGS : ${DEEPDETECT_CUDA_ARCH_FLAGS}"
    echo "  DEEPDETECT_OPENCV4_BUILD_PATH : ${DEEPDETECT_OPENCV4_BUILD_PATH}"
    echo ""
    gpu_build
elif [[ ${DEEPDETECT_ARCH} == "jetson" ]]; then
    echo ""
    echo "Deepdetect build params :"
    echo "  DEEPDETECT_ARCH      : ${DEEPDETECT_ARCH}"
    echo "  DEEPDETECT_BUILD     : ${DEEPDETECT_BUILD}"
    echo "  DEEPDETECT_CUDA_ARCH : ${DEEPDETECT_JETSON_ARCH}"
    echo "  DEEPDETECT_OPENCV4_BUILD_PATH : ${DEEPDETECT_OPENCV4_BUILD_PATH}"
    echo ""
    jetson_build
else
    echo "Missing DEEPDETECT_ARCH variable to select build profile: cpu or gpu"
fi
