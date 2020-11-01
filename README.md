<p align="center"><img src="https://www.deepdetect.com/img/icons/menu/sidebar/deepdetect.svg" alt="DeepDetect Logo" width="45%" /></p>

<h1 align="center"> Open Source Deep Learning Server & API</h1>

[![Join the chat at https://gitter.im/beniz/deepdetect](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/beniz/deepdetect?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

DeepDetect (https://www.deepdetect.com/) is a machine learning API and server written in C++11. It makes state of the art machine learning easy to work with and integrate into existing applications. It has support for both training and inference, with automatic conversion to embedded platforms with TensorRT (NVidia GPU) and NCNN (ARM CPU).

It implements support for supervised and unsupervised deep learning of images, text, time series and other data, with focus on simplicity and ease of use, test and connection into existing applications. It supports classification, object detection, segmentation, regression, autoencoders, ...

And it relies on external machine learning libraries through a very generic and flexible API. At the moment it has support for:

- the deep learning libraries [Caffe](https://github.com/BVLC/caffe), [Tensorflow](https://tensorflow.org), [Caffe2](https://caffe2.ai/), [Torch](https://pytorch.org/), [NCNN](https://github.com/Tencent/ncnn) [Tensorrt](https://github.com/NVIDIA/TensorRT) and [Dlib](http://dlib.net/ml.html)
- distributed gradient boosting library [XGBoost](https://github.com/dmlc/xgboost)
- clustering with [T-SNE](https://github.com/DmitryUlyanov/Multicore-TSNE)
- similarity search with [Annoy](https://github.com/spotify/annoy/) and [FAISS](https://github.com/facebookresearch/faiss)

Please join either the community on [Gitter](https://gitter.im/beniz/deepdetect), where we help users get through with installation, API, neural nets and connection to external applications.

---

* [Main features](#main-features)
* [Machine Learning functionalities per library](#machine-learning-functionalities-per-library)
* [Installation](https://www.deepdetect.com/quickstart-server/)
  * [From docker](https://github.com/jolibrain/deepdetect/tree/master/docs/docker.md)
  * [From source](https://github.com/jolibrain/deepdetect/tree/master/docs/source.md)
  * From Amazon AMI: [GPU](https://aws.amazon.com/marketplace/pp/B01N4D483M) and [CPU](https://aws.amazon.com/marketplace/pp/B01N1RGWQZ)

* [Models ready to use](#models)
* Ecosystem
  * [Platform](https://www.deepdetect.com/platform/)
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

## Main features:

- high-level API for machine learning and deep learning
- support for Caffe, Tensorflow, XGBoost, T-SNE, Caffe2, NCNN, TensorRT, Pytorch
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


## Machine Learning functionalities per library

|                   | Caffe | Caffe2 | XGBoost | TensorRT | NCNN | Libtorch | Tensorflow | T\-SNE | Dlib |
|------------------:|:-----:|:------:|:-------:|:--------:|:----:|:--------:|:----------:|:------:|:----:|
| **Serving**       |       |        |         |          |      |          |            |        |      |
| Training \(CPU\)  | Y     | Y      | Y       | N/A      | N/A  | Y        | N          | Y      | N    |
| Training \(GPU\)  | Y     | Y      | Y       | N/A      | N/A  | Y        | N          | Y      | N    |
| Inference \(CPU\) | Y     | Y      | Y       | N        | Y    | Y        | Y          | N/A    | Y    |
| Inference \(GPU\) | Y     | Y      | Y       | Y        | N    | Y        | Y          | N/A    | Y    |
|                   |       |        |         |          |      |          |            |        |      |
| **Models**        |       |        |         |          |      |          |            |        |      |
| Classification    | Y     | Y      | Y       | Y        | Y    | Y        | Y          | N/A    | Y    |
| Object Detection  | Y     | Y      | N       | Y        | Y    | N        | N          | N/A    | Y    |
| Segmentation      | Y     | N      | N       | N        | N    | N        | N          | N/A    | N    |
| Regression        | Y     | N      | Y       | N        | N    | Y        | N          | N/A    | N    |
| Autoencoder       | Y     | N      | N/A     | N        | N    | N        | N          | N/A    | N    |
| NLP               | Y     | N      | Y       | N        | N    | Y        | N          | Y      | N    |
| OCR / Seq2Seq     | Y     | N      | N       | N        | Y    | N        | N          | N      | N    |
| Time\-Series      | Y     | N      | N       | N        | Y    | Y        | N          | N      | N    |
|                   |       |        |         |          |      |          |            |        |      |
| **Data**          |       |        |         |          |      |          |            |        |      |
| CSV               | Y     | N      | Y       | N        |  N   | N        | N          | Y      | N    |
| SVM               | Y     | N      | Y       | N        |  N   | N        | N          | N      | N    |
| Text words        | Y     | N      | Y       | N        |  N   | N        | N          | N      | N    |
| Text characters   | Y     | N      | N       | N        |  N   | N        | N          | Y      | N    |
| Images            | Y     | Y      | N       | Y        |  Y   | Y        | Y          | Y      | Y    |
| Time\-Series      | Y     | N      | N       | N        |  Y   | N        | N          | N      | N    |

## Tools and Clients

* Python client:
  * REST client: https://github.com/jolibrain/deepdetect/tree/master/clients/python
  * 'a la scikit' bindings: https://github.com/ArdalanM/pyDD
* Javacript client: https://github.com/jolibrain/deepdetect-js
* Java client: https://github.com/kfadhel/deepdetect-api-java
* Early C# client: https://github.com/jolibrain/deepdetect/pull/98
* Log DeepDetect training metrics via Tensorboard: https://github.com/jolibrain/dd_board

## Models

|                          | Caffe | Tensorflow | Source        | Top-1 Accuracy (ImageNet) |
|--------------------------|-------|------------|---------------|---------------------------|
| AlexNet                  | Y     | N          | BVLC          |          57.1%                 |
| SqueezeNet               | [Y](https://deepdetect.com/models/squeezenet/squeezenet_v1.1.caffemodel)     | N          | DeepScale              |       59.5%                    |
| Inception v1 / GoogleNet | [Y](https://deepdetect.com/models/ggnet/bvlc_googlenet.caffemodel)     | [Y](https://deepdetect.com/models/tf/inception_v1.pb)          | BVLC / Google |             67.9%              |
| Inception v2             | N     | [Y](https://deepdetect.com/models/tf/inception_v2.pb)          | Google        |     72.2%                      |
| Inception v3             | N     | [Y](https://deepdetect.com/models/tf/inception_v3.pb)          | Google        |         76.9%                  |
| Inception v4             | N     | [Y](https://deepdetect.com/models/tf/inception_v4.pb)          | Google        |         80.2%                  |
| ResNet 50                | [Y](https://deepdetect.com/models/resnet/ResNet-50-model.caffemodel)     | [Y](https://deepdetect.com/models/tf/resnet_v1_50/resnet_v1_50.pb)          | MSR           |      75.3%                     |
| ResNet 101               | [Y](https://deepdetect.com/models/resnet/ResNet-101-model.caffemodel)     | [Y](https://deepdetect.com/models/tf/resnet_v1_101/resnet_v1_101.pb)          | MSR           |        76.4%                   |
| ResNet 152               | [Y](https://deepdetect.com/models/resnet/ResNet-152-model.caffemodel)     | [Y](https://deepdetect.com/models/tf/resnet_v1_152/resnet_v1_152.pb)         | MSR           |               77%            |
| Inception-ResNet-v2      | N     | [Y](https://deepdetect.com/models/tf/inception_resnet_v2.pb)          | Google        |       79.79%                    |
| VGG-16                   | [Y](https://deepdetect.com/models/vgg_16/VGG_ILSVRC_16_layers.caffemodel)     | [Y](https://deepdetect.com/models/tf/vgg_16/vgg_16.pb)          | Oxford        |               70.5%            |
| VGG-19                   | [Y](https://deepdetect.com/models/vgg_19/VGG_ILSVRC_19_layers.caffemodel)     | [Y](https://deepdetect.com/models/tf/vgg_19/vgg_19.pb)          | Oxford        |               71.3%            |
| ResNext 50                | [Y](https://deepdetect.com/models/resnext/resnext_50)     | N          | https://github.com/terrychenism/ResNeXt           |      76.9%                     |
| ResNext 101                | [Y](https://deepdetect.com/models/resnext/resnext_101)     | N          | https://github.com/terrychenism/ResNeXt           |      77.9%                     |
| ResNext 152               | [Y](https://deepdetect.com/models/resnext/resnext_152)     | N          | https://github.com/terrychenism/ResNeXt           |      78.7%                     |
| DenseNet-121                   | [Y](https://deepdetect.com/models/densenet/densenet_121_32/)     | N          | https://github.com/shicai/DenseNet-Caffe        |               74.9%            |
| DenseNet-161                   | [Y](https://deepdetect.com/models/densenet/densenet_161_48/)     | N          | https://github.com/shicai/DenseNet-Caffe        |               77.6%            |
| DenseNet-169                   | [Y](https://deepdetect.com/models/densenet/densenet_169_32/)     | N          | https://github.com/shicai/DenseNet-Caffe        |               76.1%            |
| DenseNet-201                   | [Y](https://deepdetect.com/models/densenet/densenet_201_32/)     | N          | https://github.com/shicai/DenseNet-Caffe        |               77.3%            |
| SE-BN-Inception                   | [Y](https://deepdetect.com/models/senets/se_bn_inception/)     | N          | https://github.com/hujie-frank/SENet        |               76.38%            |
| SE-ResNet-50                   | [Y](https://deepdetect.com/models/senets/se_resnet_50/)     | N          | https://github.com/hujie-frank/SENet        |               77.63%            |
| SE-ResNet-101                   | [Y](https://deepdetect.com/models/senets/se_resnet_101/)     | N          | https://github.com/hujie-frank/SENet        |               78.25%            |
| SE-ResNet-152                   | [Y](https://deepdetect.com/models/senets/se_resnet_152/)     | N          | https://github.com/hujie-frank/SENet        |               78.66%            |
| SE-ResNext-50                   | [Y](https://deepdetect.com/models/senets/se_resnext_50/)     | N          | https://github.com/hujie-frank/SENet        |               79.03%            |
| SE-ResNext-101                   | [Y](https://deepdetect.com/models/senets/se_resnext_101/)     | N          | https://github.com/hujie-frank/SENet        |               80.19%            |
| SENet                   | [Y](https://deepdetect.com/models/senets/se_net/)     | N          | https://github.com/hujie-frank/SENet        |               81.32%            |
| VOC0712 (object detection) | [Y](https://deepdetect.com/models/voc0712_dd.tar.gz) | N | https://github.com/weiliu89/caffe/tree/ssd | 71.2 mAP |
| InceptionBN-21k | [Y](https://deepdetect.com/models/inception/inception_bn_21k) | N | https://github.com/pertusa/InceptionBN-21K-for-Caffe | 41.9% |
| Inception v3 5K | N | [Y](https://deepdetect.com/models/tf/openimages_inception_v3) | https://github.com/openimages/dataset |  |
| [5-point Face Landmarking Model (face detection)](http://dlib.net/files/mmod_human_face_detector.dat.bz2) | N | N | http://blog.dlib.net/2017/09/fast-multiclass-object-detection-in.html |  |
| [Front/Rear vehicle detection (object detection)](http://dlib.net/files/mmod_front_and_rear_end_vehicle_detector.dat.bz2) | N | N | http://blog.dlib.net/2017/09/fast-multiclass-object-detection-in.html |  |

More models:

- List of free, even for commercial use, deep neural nets for image classification, and character-based convolutional nets for text classification: https://www.deepdetect.com/applications/list_models/

<!---
#FIXME(sileht): it's a feature detail, should be moved somewhere in deepdetect.com/server/docs/
## Templates

DeepDetect comes with a built-in system of neural network templates (Caffe backend only at the moment). This allows the creation of custom networks based on recognized architectures, for images, text and data, and with much simplicity.

Usage:
- specify `template` to use, from `mlp`, `convnet` and `resnet`
- specify the architecture with the `layers` parameter:
  - for `mlp`, e.g. `[300,100,10]`
  - for `convnet`, e.g. `["1CR64","1CR128","2CR256","1024","512"], where the main pattern is `xCRy` where `y` is the number of outputs (feature maps), `CR` stands for Convolution + Activation (with `relu` as default), and `x` specifies the number of chained `CR` blocks without pooling. Pooling is applied between all `xCRy`
- for `resnets`:
   - with images, e.g. `["Res50"]` where the main pattern is `ResX` with X the depth of the Resnet
   - with character-based models (text), use the `xCRy` pattern of convnets instead, with the main difference that `x` now specifies the number of chained `CR` blocks within a resnet block
   - for Resnets applied to CSV or SVM (sparse data), use the `mlp` pattern. In this latter case, at the moment, the `resnet` is built with blocks made of two layers for each specified layer after the first one. Here is an example: `[300,100,10]` means that a first hidden layer of size `300` is applied followed by a `resnet` block made of two `100` fully connected layer, and another block of two `10` fully connected layers. This is subjected to future changes and more control.
-->

## References

- DeepDetect (https://www.deepdetect.com/)
- Caffe (https://github.com/jolibrain/caffe)
- XGBoost (https://github.com/dmlc/xgboost)
- T-SNE (https://github.com/DmitryUlyanov/Multicore-TSNE)

## Authors
DeepDetect is designed, implemented and supported by [Jolibrain](https://jolibrain.com/) with the help of other contributors.
