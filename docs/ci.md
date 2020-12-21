# Building DeepDetect like the Continuous Integration does:

## Preparing the ubuntu images with all deps

```
$ docker build -t dd-dev -f ci/devel.Dockerfile --progress=plain .
```

### What to try another cuda, tensorrt, cudnn or ubuntu ?
```
$ docker build -t dd-dev -f ci/devel.Dockerfile --progress=plain \
    --build-arg DD_UBUNTU_VERSION=20.04 \
    --build-arg DD_CUDA_VERSION=11.1 \
    --build-arg DD_CUDNN_VERSION=8 \
    --build-arg DD_TENSORRT_VERSION=7.2.1-1+cuda11.1 \
    .
```

## Building local DeepDetect within the built docker image

```
$ docker run -it -v $(pwd):/dd -w /dd dd-dev /bin/bash
root@4b4d72e9c8b4:/dd# mkdir build
root@4b4d72e9c8b4:/dd# cd build
root@4b4d72e9c8b4:/dd/build# cmake .. -DBUILD_TESTS=ON
root@4b4d72e9c8b4:/dd/build# make -j$(nproc)
```

## Running tests

```
docker run -it -v $(pwd):/dd -w /dd dd-dev /bin/bash
root@4b4d72e9c8b4:/dd# cd build && make tests

  OR

root@4b4d72e9c8b4:/dd# cd build/tests && ./ut_caffeapi
```
