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

The docker container mounts the prebuilt directory as copy-on-write volume

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

## DeepDetect it self

On a clean master branch with all tags fetched:

```bash
$ git fetch --tags
$ git checkout master
$ git reset --hard origin/master
$ yarn
$ ci/release.sh
```

If the result is OK, publish the release note on GitHub and push tags:

```
$ git push --follow-tags origin master
```

The script `ci/release.sh` updates CHANGELOG.md, commits it, creates a tag, and
creates the GitHub release.

## Docker images

On Jenkins in `deepdetect-docker-build` job, `Tags` tab, ran the released version

## The platform

When the docker images have been released, `platform_ui` and `dd_platform_docker` can be released:

```bash
$ git fetch --tags
$ git checkout master
$ git reset --hard origin/master
$ yarn
$ ci/release.sh
$ git push --follow-tags origin master
```

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

# Adding a new slave node

On the new slave node as root:

```
apt install -y openjdk-11-jre
adduser jenkins --shell /bin/bash --disabled-password --home /var/lib/jenkins
usermod jenkins -a -G docker
mkdir /var/lib/jenkins/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHmrxyMYsQL8HSKjq4ASmxtWUXl4395XswKmGXDtQpvk jenkins@jenkins" > /var/lib/jenkins/.ssh/authorized_keys
chown -R jenkins:jenkins /var/lib/jenkins/.ssh
chmod 500 /var/lib/jenkins/.ssh/authorized_keys
```

On x86 GPU node, ensure cuda nvidia drivers and docker are installed too.

On jenkins Master nodes:

```
sudo -u jenkins -i
ssh-keyscan 10.10.77.72 >> /var/lib/jenkins/.ssh/known_hosts
```
On Jenkins UI:

* Click on `Manage Jenkins` -> `Manage Nodes and Clouds` -> `New Node`
* Set the `Node name`, select `Permanent Agent` and click on `Add`
* Set `Remote root directory` to `/var/lib/jenkins`
* In `Labels` add 
  * `gpu` for x86 nodes
  * `nano` for jetson nano nodes
* In `Usage`, select `Only build jobs with label expressions matching this node`
* In `Launch method`, select `Launch agents via SSH`
  * Set `Host` to the machine hostname or IP
  * Use `Jenkins` Credentials
* On x86, you can increse `# of executors` depending on RAM available.
* Click `Save`
* Click `Relaunch Agent`

When you see `Agent successfully connected and online` you're good.

For x86 GPU nodes only:

* Click on `Manage Jenkins` -> `Configure System`
* In `Lockable Resources Manager` section adds all GPUs the node have
  The naming is important, `Jenkins.unittests` job use it to reserve GPU
  For each GPU to must create a resources with:
  * name and description: `<UPPERCASE NODE NAME> GPU <GPU_INDEX>` (example: `NEPTUNE05T GPU 0`)
  * labels `<NODE NAME>-gpu` (example: `neptune05t-gpu`)
* Click `Save`

## How Jenkins jobs dispatch works

Job dispatch use Jenkins Labels. We have `master`, `gpu` and `nano`.

`master` is used mainly for sync prebuild cache and docker images
`gpu` to run unittests on pull requests
`nano` for jetson nano related jobs

In Jenkins job file, the node is select by the agent section, eg:

```
pipeline {
  agent { node { label 'gpu' } }
  ...
}
```
