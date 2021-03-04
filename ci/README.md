# Jenkins Jobs configuration

## deepdetect-prebuilt-cache


-> Jenkinsfile.prebuilt-cache

Compile all dependencies of DeepDetect and use Jenkins artefacts to keep them here:

```
/var/lib/jenkins/jobs/deepdetect-prebuilt-cache/branches/master/builds/<BUILD ID>/archive/build/
```

* Triggered manually

## deepdetect-prebuilt-cache-syncer

Job that just keep in sync `/var/lib/jenkins/jobs/deepdetect-prebuilt-cache/` between CIs servers

* Triggered manually


## deepdetect

Build and run all tests.

Everything is done inside a docker image: ci/devel.Dockerfile

The docker container mount the prebuilt directory as copy-on-write volume

> Jenkinsfile.unittests

trigger on pull request only

## deepdetect-jetson-nano

Build tensorrt backend and run tensorrt tests on a Jetson Nano

Everything is done inside a docker image: ci/devel-jetsone-nano.Dockerfile

> Jenkinsfile-jetson-nano.unittests

trigger on pull request with ci:embedded only

## deepdetect-docker-build

Build all docker images and push them on dockerhub.
Keep in sync the dockerhub README with the GitHub README

trigger every night on master branch
trigger manually on release tag

# Release process

On a clean master branch with all tags fetched:

```
$ git fetch --tags
$ ci/release.sh
```

If the result is OK, publish the release note on GitHub and push tags:

```
$ git push --tags
```

The script `ci/release.sh` updates CHANGELOG.md, commit it, create a tag, and
create the GitHub release.

# Building DeepDetect like the Jenkins  does:

## Preparing the ubuntu images with all deps

```
$ docker build -t dd-dev -f ci/devel.Dockerfile --progress=plain .
```

### Want to try another cuda, tensorrt, cudnn or ubuntu ?
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
