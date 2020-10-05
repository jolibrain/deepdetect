#!/bin/bash

GPU_IMG=nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

sed \
    -e "s,FROM [^ ]* AS build,FROM ${GPU_IMG} AS build,g" \
    -e "s,ARG DEEPDETECT_ARCH=.*,ARG DEEPDETECT_ARCH=gpu,g" \
    cpu.Dockerfile > gpu.Dockerfile
