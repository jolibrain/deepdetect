#!/bin/bash

set -e

# Deepdetect architecture and build profiles
deepdetect_arch=(cpu gpu)
deepdetect_cpu_build_profiles=(default tf armv7)
deepdetect_gpu_build_profiles=(default tf caffe2 tensorrt)

# NOTE(sileht): list of all supported card by CUDA 10.2
# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
if [ ! "$DEEPDETECT_CUDA_ARCH" ]; then
    for card in 50 52 60 61 62 70 72; do
        DEEPDETECT_CUDA_ARCH="$DEEPDETECT_CUDA_ARCH -gencode arch=compute_${card},code=sm_${card}"
    done
    # trim spaces
    DEEPDETECT_CUDA_ARCH="$(echo ${DEEPDETECT_CUDA_ARCH} | xargs)"
fi


# Help menu with arguments descriptions
help_menu() {
    IFS=,
    echo "Deepdetect build script"
    echo ""
    echo "Env variables usage: DEEPDETECT_ARCH=${deepdetect_arch[*]} DEEPDETECT_BUILD=${deepdetect_cpu_build_profiles[*]},[...] $0 "
    echo ""
    echo "or"
    echo ""
    echo "Params usage: $0 [options...]" >&2
    echo
    echo "   -a, --deepdetect-arch          Choose Deepdetect architecture : ${deepdetect_arch[*]}"
    echo "   -b, --deepdetect-build         Choose Deepdetect build profile : CPU (${deepdetect_cpu_build_profiles[*]}) / GPU (${deepdetect_gpu_build_profiles[*]})"
    echo "   -c, --deepdetect-cuda-arch     Choose Deepdetect cuda arch (default: ${deepdetect_cuda_arch})"
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
    -h | --help)
        help_menu
        break
        ;;
    esac
done

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
    if [ ${DEEPDETECT_ARCH} == "gpu" ]; then
        select_gpu_build_profile
    fi
}

# Build functions
cpu_build() {

    case ${DEEPDETECT_BUILD} in

    "tf")
        cmake .. -DUSE_TF=ON -DUSE_TF_CPU_ONLY=ON -DUSE_SIMSEARCH=ON -DUSE_TSNE=ON -DUSE_NCNN=OFF -DUSE_CPU_ONLY=ON
        make
        ;;

    "armv7")
        cmake .. -DUSE_NCNN=ON -DRPI3=ON -DUSE_HDF5=OFF -DUSE_CAFFE=OFF
        make
        ;;

    *)
        #cmake .. -DUSE_XGBOOST=ON -DUSE_SIMSEARCH=ON -DUSE_TSNE=ON -DUSE_NCNN=ON -DUSE_CPU_ONLY=ON
        cmake .. -DUSE_XGBOOST=OFF -DUSE_SIMSEARCH=OFF -DUSE_TSNE=OFF -DUSE_NCNN=OFF -DUSE_CPU_ONLY=ON
        make
        ;;
    esac

}

gpu_build() {
    local extra_flags=
    case ${DEEPDETECT_BUILD} in
        "tf") extra_flags="-DUSE_TF=ON" ;;
        "caffe2") extra_flags="-DUSE_CAFFE2=ON" ;;
        "tensorrt") extra_flags="-DUSE_TENSORRT_OSS=ON" ;;
    esac
    cmake .. $extra_flags -DUSE_FAISS=ON -DUSE_CUDNN=ON -DUSE_XGBOOST=ON -DUSE_SIMSEARCH=ON -DUSE_TSNE=ON -DUSE_TORCH=ON -DCUDA_ARCH="${DEEPDETECT_CUDA_ARCH}"
    make
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
    echo "  DEEPDETECT_CUDA_ARCH : ${DEEPDETECT_CUDA_ARCH}"
    echo ""
    gpu_build
else
    echo "Missing DEEPDETECT_ARCH variable to select build profile: cpu or gpu"
fi
