<p align="center"><img src="https://www.deepdetect.com/img/icons/menu/sidebar/deepdetect.svg" alt="DeepDetect Logo" width="45%" /></p>

# Deep Learning Server and CLI

![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/jolibrain/deepdetect?color=success&sort=semver)
![GitHub Release Date](https://img.shields.io/github/release-date/jolibrain/deepdetect?display_date=created_at)
![GitHub commits since latest release](https://img.shields.io/github/commits-since/jolibrain/deepdetect/latest/master)

DeepDetect is a deep learning runtime, command-line tool, and REST server for
training and inference. The Python wheel embeds the DeepDetect runtime in the
current Python environment and provides the `deepdetect` CLI for repeatable
model workflows. The server remains available for long-running REST services,
containerized serving, and integrations that need a dedicated process.

DeepDetect focuses on practical model operations: create services, train
models, run predictions, monitor jobs, and keep model repositories organized on
the filesystem. It supports images, text, CSV/tabular data, time series,
sparse/SVM-style data, object-detection boxes, and segmentation masks through a
single API surface.

## Quickstart

Install one wheel variant in a Python environment. `deepdetect-cpu` and
`deepdetect-gpu` both provide `import deepdetect`, so they are mutually
exclusive in the same environment.

```bash
python -m pip install \
  --extra-index-url https://www.deepdetect.com/download/wheels/simple \
  deepdetect-cpu
```

For a CUDA-enabled environment, install the GPU package instead:

```bash
python -m pip install \
  --extra-index-url https://www.deepdetect.com/download/wheels/simple \
  deepdetect-gpu
```

Inspect the packaged CLI profiles and command options:

```bash
deepdetect inspect models
deepdetect train yolox --help
deepdetect infer segformer --help
```

The first CLI profiles are:

- `yolox`: object detection
- `segformer`: semantic segmentation
- `torchvision-detector`: managed PyTorch Faster R-CNN detection
- `external-pytorch-detector`: external PyTorch detection worker entrypoint

YAML config files make runs repeatable. The default examples are starting
points; replace dataset, weight, and repository paths before using them for a
real run.

```bash
deepdetect train yolox --config bindings/python/deepdetect/cli/yolox-default.yaml
deepdetect infer yolox image.jpg --config bindings/python/deepdetect/cli/yolox-default.yaml
deepdetect train external-pytorch-detector --config bindings/python/deepdetect/cli/external-pytorch-detector-default.yaml
```

A minimal object-detection run can use the same tiny fixtures as the wheel
tests. Start Visdom in a second terminal:

```bash
python -m pip install visdom
python -m visdom.server -port 8097
```

Prepare the quickstart dataset and model repository:

```bash
python bindings/python/scripts/prepare_cli_yolox_quickstart.py \
  --output /tmp/deepdetect-yolox-quickstart \
  --force
```

Train a very small YOLOX run:

```bash
deepdetect train yolox \
  --config /tmp/deepdetect-yolox-quickstart/yolox-quickstart.yaml \
  --terminal live
```

Then run inference on the `Sample image:` path printed by the preparation
script:

```bash
deepdetect infer yolox <sample-image> \
  --config /tmp/deepdetect-yolox-quickstart/yolox-quickstart.yaml \
  --visualize \
  --output /tmp/deepdetect-yolox-quickstart/detections.png
```

Use the `deepdetect-gpu` wheel and add `--gpu` to the train and infer commands
to run this example on CUDA.

See the [CLI specification](bindings/python/deepdetect/cli/CLI_SPEC.md) for
training, inference, monitoring, config precedence, and output formats.

## Core Capabilities

- Train and run inference from the `deepdetect` CLI or the REST API.
- Create model services backed by local model repositories.
- Run asynchronous training jobs and inspect their status.
- Work with image classification, object detection, semantic segmentation,
  language models, tabular data, time series, and sparse data.
- Use filesystem-based model storage without a database dependency.
- Emit structured CLI events for automation and human-readable terminal output.
- Use JSON request and response payloads for server integrations.
- Generate prediction outputs such as classes, scores, bounding boxes,
  segmentation masks, and model-specific metrics.

## Backends and Model Support

DeepDetect uses **Torch** as the primary backend for training and inference.
**TensorRT** is available for optimized inference with exported or compatible
models. Caffe-format protobufs and prototxt files may appear as compatibility
or model-format details, but Caffe is not an active runtime backend.

The CLI currently packages focused workflows for:

- **YOLOX** object detection.
- **SegFormer** semantic segmentation.

The broader Torch API supports additional model and template families,
including:

- Image classification with TorchVision-style classifiers such as ResNet, VGG,
  DenseNet, MobileNet, ShuffleNet, and SqueezeNet.
- Object detection with YOLOX, Faster R-CNN, and RetinaNet.
- Semantic segmentation with SegFormer and segmentation services.
- Language and traced models such as BERT and GPT-2.
- Time series with recurrent, N-BEATS, transformer, and time-transformer
  templates.
- Vision transformers such as ViT and Visformer.

See the [API reference](docs/api.md) for service parameters, connectors,
templates, and request/response details.

## Deployment

Use the Python wheel and CLI for local training, in-process inference, and
automation-friendly workflows.

Use the DeepDetect server when you need a long-running REST service, remote
clients, asynchronous jobs behind an HTTP API, or a dedicated serving process.
The REST API is documented in [docs/api.md](docs/api.md).

Use Docker for containerized serving and reproducible service environments.
See [docs/docker.md](docs/docker.md).

Build from source when you need a custom C++ build, server options, TensorRT
support, or local development changes. Start with [docs/source.md](docs/source.md).

## Python Client

The [Python REST client](clients/python) talks to a running DeepDetect server.
It is separate from the in-process Python wheel, which provides `import
deepdetect` and the `deepdetect` CLI. Wheel build and packaging details are in
[bindings/python/README.md](bindings/python/README.md).

## Documentation

- [REST API reference](docs/api.md)
- [Docker usage](docs/docker.md)
- [Source builds](docs/source.md)
- [Python wheel and embedded runtime](bindings/python/README.md)
- [Python CLI specification](bindings/python/deepdetect/cli/CLI_SPEC.md)

## Authors, License, and Reference

DeepDetect is designed, implemented, and supported by
[Jolibrain](https://jolibrain.com/) with help from contributors.

Authors are listed in [AUTHORS](AUTHORS). DeepDetect is distributed under the
GNU Lesser General Public License v3.0; see [COPYING](COPYING).

Project website: https://www.deepdetect.com/
