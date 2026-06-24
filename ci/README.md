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

## deepdetect-python-wheels

Build CPU and GPU Python wheels, install each wheel in a clean virtualenv, and
run the Python binding unit tests plus a native import smoke test. The GPU
wheel stage also runs translated Torch integration coverage from
`tests/ut-torchapi.cc` for ResNet image training, YOLOX object detection
training, and SegFormer segmentation training.

Everything is done inside a docker image: ci/devel.Dockerfile

> Jenkinsfile.python-wheels

trigger on pull request only

On Jenkins:

* Create a new Multibranch Pipeline job named `deepdetect-python-wheels`
* Use the same repository, GitHub credentials, and PR discovery settings as the
  `deepdetect` job
* Set `Script Path` to `ci/Jenkinsfile.python-wheels`
* Ensure the job runs on nodes with the `gpu` label
* Ensure GPU lockable resources are configured with labels matching
  `<node-name>-gpu`, as described below
* Run `Scan Multibranch Pipeline Now`
* After the first successful PR run, add the resulting Jenkins check to GitHub
  branch protection if wheel validation should be required before merge

The job does not archive wheels. Its artifacts are validation-only and are
removed by workspace cleanup.

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

## DeepDetect itself

Run the integrated release process from a clean `master` branch:

```bash
$ git fetch --tags
$ git checkout master
$ git reset --hard origin/master
$ ci/release_all.sh patch --user <ssh-user>
```

The script creates a temporary staging clone, generates the release commit and
local tag there, builds and tests the release Docker images and CPU/GPU wheels,
then asks for confirmation before publishing anything. After confirmation it:

* pushes `deepdetect_cpu`, `deepdetect_gpu`, and `deepdetect_gpu_tensorrt`
  images to `docker.jolibrain.com` with `vX.Y.Z` and `latest` tags;
* uploads CPU/GPU wheels and regenerates the pip simple indexes under
  `https://www.deepdetect.com/download/wheels/`;
* pushes the release commit and tag to GitHub;
* creates a draft GitHub release with changelog notes plus Docker and wheel
  install links.

To preview the process without publishing Docker images, wheels, git refs, or a
GitHub release:

```bash
$ ci/release_all.sh --dry-run patch --user <ssh-user>
```

Use `--skip-docker` or `--skip-wheels` for targeted dry-run checks. Use `--yes`
only for trusted automation where the final confirmation prompt is not wanted.

The older `ci/release.sh` script is kept for source-only recovery work, but it
pushes the tag before Docker images and wheels are proven, so the integrated
release flow should be preferred for normal releases.

## Docker images

Docker images can still be built independently. For a local release-style build
and smoke test without pushing:

```bash
$ ci/build-docker-images.sh --tag v0.28.1 --release --no-push cpu gpu gpu_tensorrt
```

To push those already-built local images:

```bash
$ ci/build-docker-images.sh --tag v0.28.1 --push-only cpu gpu gpu_tensorrt
```

GPU image smoke tests run with `DOCKER_GPU_RUN_ARGS="--runtime nvidia"` by
default. Override it if the local Docker setup uses a different NVIDIA
interface, for example:

```bash
$ DOCKER_GPU_RUN_ARGS="--gpus all" ci/build-docker-images.sh --tag v0.28.1 --release --no-push gpu
```

The Jenkins `deepdetect-docker-build` job still uses the same helper script for
scheduled and tag-triggered builds.

## Python wheels

Large Python wheels are published outside PyPI under:

```
https://www.deepdetect.com/download/wheels/
```

Build, index, and upload CPU/GPU wheels with:

```bash
$ ci/release_wheels.sh --user <ssh-user>
```

The integrated release flow calls this script automatically. When used
directly, the script builds `deepdetect-cpu` and `deepdetect-gpu`, stages wheels under
`dist/python/wheelhouse`, generates pip simple indexes with `sha256` URL
fragments, and uploads them to the web server. The default public install
commands are:

```bash
$ python -m pip install --extra-index-url https://www.deepdetect.com/download/wheels/simple deepdetect-cpu
$ python -m pip install --extra-index-url https://www.deepdetect.com/download/wheels/simple deepdetect-gpu
```

On an exact git tag such as `v0.28.1`, the wheel version is `0.28.1`.
Otherwise the script uses a development version based on the DeepDetect
project version, git commit count, and commit hash, for example
`0.28.0.dev3361+g625ccc22`. Use `--version <version>` to override this during
testing.

Existing uploaded wheels are kept online. With the default `rsync` transport,
the script downloads the current remote wheelhouse before regenerating the
indexes, so old versions remain installable. Use `--dry-run --skip-build` to
test index generation locally without uploading.
The default upload target is `/var/www/deepdetect/public/download/wheels` on
`www.deepdetect.com`; override it with `--remote-dir` if needed.
The generated remote layout is:

```
public/download/wheels/simple/
public/download/wheels/files/cpu/
public/download/wheels/files/gpu/
```

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
    --build-arg DD_UBUNTU_VERSION=24.04 \
    --build-arg DD_CUDA_VERSION=13.0.2 \
    --build-arg DEEPDETECT_GPU_VARIANT=default \
    --build-arg DEEPDETECT_TENSORRT_VERSION= \
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
