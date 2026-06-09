<p align="center"><img src="https://www.deepdetect.com/img/icons/menu/sidebar/deepdetect.svg" alt="DeepDetect Logo" width="45%" /></p>

<h1 align="center"> Open Source Deep Learning Server & API</h1>

[![Join the chat at https://gitter.im/beniz/deepdetect](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/beniz/deepdetect?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/jolibrain/deepdetect?color=success&sort=semver)
![GitHub Release Date](https://img.shields.io/github/release-date/jolibrain/deepdetect)
![GitHub commits since latest release (by date)](https://img.shields.io/github/commits-since/jolibrain/deepdetect/latest/master)


DeepDetect (https://www.deepdetect.com/) is a machine learning API and server written in C++11. It makes state of the art machine learning easy to work with and integrate into existing applications. It has support for both training and inference, with automatic conversion to embedded platforms with TensorRT (NVidia GPU) and NCNN (ARM CPU).

It implements support for supervised and unsupervised deep learning of images, text, time series and other data, with focus on simplicity and ease of use, test and connection into existing applications. It supports classification, object detection, segmentation, regression, autoencoders, ...

And it relies on external machine learning libraries through a very generic and flexible API. At the moment it has support for:

- deep learning with [Torch](https://pytorch.org/), [TensorRT](https://github.com/NVIDIA/TensorRT), [NCNN](https://github.com/Tencent/ncnn), and [Dlib](http://dlib.net/ml.html)
- distributed gradient boosting library [XGBoost](https://github.com/dmlc/xgboost)
- clustering with [T-SNE](https://github.com/DmitryUlyanov/Multicore-TSNE)
- similarity search with [Annoy](https://github.com/spotify/annoy/) and [FAISS](https://github.com/facebookresearch/faiss)

Please join the community on [Gitter](https://gitter.im/beniz/deepdetect), where we help users get through with installation, API, neural nets and connection to external applications.

---

| Build type | STABLE | DEVEL |
|------|--------|-------|
| SOURCE | <img src="https://img.shields.io/github/v/release/jolibrain/deepdetect?color=success&sort=semver"> | <img src="https://img.shields.io/github/commits-since/jolibrain/deepdetect/latest/master"> |

All DeepDetect Docker images available from https://docker.jolibrain.com/.

- To list all available images:
```
curl -X GET https://docker.jolibrain.com/v2/_catalog
```

- To list an image available tags, e.g. for the `deepdetect_cpu` image:
```
curl -X GET https://docker.jolibrain.com/v2/deepdetect_cpu/tags/list
```

---

* [Main features](#main-features)
* [Machine Learning functionalities per library](#machine-learning-functionalities-per-library)
* [Installation](https://www.deepdetect.com/quickstart-server/)
  * [From docker](https://github.com/jolibrain/deepdetect/tree/master/docs/docker.md)
  * [Python wheels](#python-wheels)
  * [From source](https://github.com/jolibrain/deepdetect/tree/master/docs/source.md)
  * From Amazon AMI: [GPU](https://aws.amazon.com/marketplace/pp/B01N4D483M) and [CPU](https://aws.amazon.com/marketplace/pp/B01N1RGWQZ)
  * [Mimic Continuous Integration testing](https://github.com/jolibrain/deepdetect/tree/master/docs/ci.md)

* Ecosystem
  * [Platform presentation](https://www.deepdetect.com/platform/)
  * [Platform installation with docker-compose](https://github.com/jolibrain/dd_platform_docker)
  * [Platform installation with helm (Kubernetes)](https://github.com/jolibrain/helm_chart)
  * [Tools and Clients](#tools-and-clients)
* Documentation:
  * [Introduction](https://www.deepdetect.com/overview/introduction/)
  * [API Quickstart](https://www.deepdetect.com/server/docs/imagenet-classifier/): setup an image classifier API service in a few minutes
  * [API Tutorials](https://www.deepdetect.com/server/docs/server_docs/): training from text, data and images, setup of prediction services, and export to external software (e.g. ElasticSearch)
  * [API Reference](https://www.deepdetect.com/api/)
  * [Examples](https://www.deepdetect.com/server/docs/examples/): MLP for data, text, multi-target regression to CNN and GoogleNet, finetuning, etc...)
  * [FAQ](https://www.deepdetect.com/overview/faq/)
* Demos:
  * [Image classification Web application](https://github.com/jolibrain/deepdetect/tree/master/demo/imgdetect) using HTML and javascript
  * [Image similarity search](https://github.com/jolibrain/deepdetect/tree/master/demo/imgsearch) using python client
  * [Image object detection](https://github.com/jolibrain/deepdetect/tree/master/demo/objdetect) using python client
  * [Image segmentation](https://github.com/jolibrain/deepdetect/tree/master/demo/segmentation) using python client
* [Performance tools and report](https://github.com/jolibrain/dd_performances) done on NVidia Desktop and embedded GPUs, along with Raspberry Pi 3.
* [References](#references)
* [Authors](#authors)

## Main features

- high-level API for machine learning and deep learning
- support for Torch, TensorRT, NCNN, Dlib, XGBoost, and T-SNE
- classification, regression, autoencoders, object detection, segmentation, time-series
- JSON communication format
- remote Python and Javacript clients
- dedicated server with support for asynchronous training calls
- high performances, benefit from multicore CPU and GPU
- built-in similarity search via neural embeddings
- connector to handle large collections of images with on-the-fly data augmentation (e.g. rotations, mirroring)
- connector to handle CSV files with preprocessing capabilities
- connector to handle text files, sentences, and character-based models
- connector to handle SVM file format for sparse data
- range of built-in model assessment measures (e.g. F1, multiclass log loss, ...)
- range of special losses (e.g Dice, contour, ...)
- no database dependency and sync, all information and model parameters organized and available from the filesystem
- flexible template output format to simplify connection to external applications
- templates for the most useful neural architectures (e.g. Googlenet, Alexnet, ResNet, convnet, character-based convnet, mlp, logistic regression, SSD, DeepLab, PSPNet, U-Net, CRNN, ShuffleNet, SqueezeNet, MobileNet, RefineDet, VOVNet, ...)
- support for sparse features and computations on both GPU and CPU
- built-in similarity indexing and search of predicted features, images, objects and probability distributions
- auto-generated documentation based on [Swagger](https://swagger.io/)


## Machine Learning functionalities per library

Torch is the primary training and serving backend. TensorRT and NCNN provide optimized inference, while Dlib, XGBoost, and T-SNE remain available for their specialized workloads.

Caffe-format protobufs and prototxt templates remain available only as a model interchange format used by Torch, TensorRT, and NCNN. They do not provide a Caffe runtime backend.

## Python Wheels

DeepDetect also provides experimental in-process Python wheels for Linux
x86-64. These wheels embed the DeepDetect runtime and use LibTorch from the
PyTorch Python wheels. They are separate from the REST client and expose
`import deepdetect`.

Install either the CPU or GPU package, not both in the same Python
environment:

```bash
python -m pip install \
  --extra-index-url https://www.deepdetect.com/download/wheels/simple \
  deepdetect-cpu
```

```bash
python -m pip install \
  --extra-index-url https://www.deepdetect.com/download/wheels/simple \
  deepdetect-gpu
```

If reinstalling a development wheel with the same version, disable the pip
cache:

```bash
python -m pip install --force-reinstall --no-cache-dir \
  --extra-index-url https://www.deepdetect.com/download/wheels/simple \
  deepdetect-gpu
```

Verify the native runtime:

```bash
python -c "import deepdetect; print(deepdetect.__version__); print(deepdetect.DeepDetect().build_info)"
```

Minimal usage:

```python
import deepdetect

dd = deepdetect.DeepDetect()
print(dd.info())

service = dd.create_service(
    "classifier",
    model={"repository": "/path/to/model"},
    mllib="torch",
    input_parameters={"connector": "image", "width": 224, "height": 224},
    mllib_parameters={"template": "resnet18", "nclasses": 2, "gpu": True},
)
print(service.predict(["image.jpg"], output_parameters={"best": 1}))
service.delete()
```

## Tools and Clients

* Python client:
  * REST client: https://github.com/jolibrain/deepdetect/tree/master/clients/python
  * 'a la scikit' bindings: https://github.com/ArdalanM/pyDD
* Javacript client: https://github.com/jolibrain/deepdetect-js
* Java client: https://github.com/kfadhel/deepdetect-api-java
* Early C# client: https://github.com/jolibrain/deepdetect/pull/98
* Log DeepDetect training metrics via Tensorboard: https://github.com/jolibrain/dd_board

## Backend migration

The Caffe, Caffe2, and TensorFlow backends have been removed. Existing services must be converted to Torch or exported to ONNX for TensorRT before upgrading. Requests using `mllib` values `caffe`, `caffe2`, `tf`, or `tensorflow` return HTTP 400 with DeepDetect error code 1006.

## References

- DeepDetect (https://www.deepdetect.com/)
- XGBoost (https://github.com/dmlc/xgboost)
- T-SNE (https://github.com/DmitryUlyanov/Multicore-TSNE)

## Authors
DeepDetect is designed, implemented and supported by [Jolibrain](https://jolibrain.com/) with the help of other contributors.
