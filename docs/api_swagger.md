# Introduction

Welcome to the DeepDetect API!

DeepDetect is a Machine Learning server. At this stage, it provides a flexible API to train deep neural networks and gradient boosted trees, and use them where they are needed, in both development and production.

## Principles

The Open Source software provides a server, an API, and the underlying Machine Learning procedures for training statistical models. The REST API defines a set of resources and options in order to access and command the server over a network.

### Architecture

The software defines a very simple flow, from data to the statistical model and the final application. The main elements and vocabulary are in that order:

* `data` or `dataset`: images, numerical data, or text
* `input connector`: entry point for data into DeepDetect. Specialized versions handle different data types (e.g. images, text, CSV, ...)
* `model`: repository that holds all the files necessary for building and usage of a statistical model such as a neural net
* `service`: the central holder of models and connectors, living in memory and servicing the machine learning capabilities through the API. While the `model` can be held permanently on disk, a `service` is spawn around it and destroyed at will
* `mllib`: the machine learning library used for operations, two are supported at the moment, Caffe, Caffe2, XGBoost, Dlib, NCNN and Tensorflow, more are on the way
* `training`: the computational phase that uses a dataset to build a statistical model with predictive abilities on statistically relevant data
* `prediction`: the computational phase that uses a trained statistical model in order to make a guess about one or more samples of data
* `output connector`: the DeepDetect output, that supports templates so that the output can be easily customized by the user in order to fit in the final application

### API Principles

The main idea behind the API is that it allows users to spawn Machine Learning `services`, each serving its own purpose, and to interact with them.

The REST API builds around four resources:

* `/info`: yields the general information about the server and the services currently being active on it
* `/services`: yields access to creation and destruction of Machine Learning services.
* `/train`: controls the resources for the potentially long computational phase of building the statistical model from a `dataset`
* `/predict`: takes data in, and uses a trained statistical model to make predictions over some properties of the data

Each of the resources are detailed below, along with their options and examples to be tested on the command line.

# Train

Trains a statistical model from a dataset, the model can be further used for prediction

The DeepDetect server supports both blocking and asynchronous training calls. Training is often a very computational operation that can last for days in some cases.

Blocking calls block the communication with the server, and returns results once completed. They are not well suited to most machine learning tasks.

Asynchronous calls run the training in the background as a separate thread (`PUT /train`). Status of the training job can be consulted live with by calling on the server (`GET /train`). The final report on an asynchronous training job is consumed by the first `GET /train` call after completion of the job. After that, the job is definitely destroyed.

<div class="alert alert-primary mx-2" style="width: 58%">
⚠️ Asynchronous training calls are the default, use of blocking calls is useful for testing and debugging
</div>

<div class="alert alert-danger mx-2" style="width: 58%">
⚠️ The current integration of the Caffe back-end for deep learning does not allow making predictions while training. However, two different services can train and predict at the same time.
</div>

# Predict

Makes predictions from data out of an existing statistical model. If `measure` is specified, the prediction expects a supervised dataset and produces accuracy measures as output, otherwise it is prediction for every of the input samples.

# Connectors

The DeepDetect API supports the control of input and output connectors.

* `input connectors` are parametrized with the `input` JSON object

> input connector:

```json
"parameters":{
   "input":{

  }
}
```

* `output connectors` are parametrized with the `output` JSON object

> output connector:

```json
"parameters":{
   "output":{

  }
}
```

<div class="alert alert-primary mx-2" style="width: 58%">
⚠️ Connectors are defined at service creation but their options can be modified in `train` and `predict` calls as needed.
</div>

## Input connectors
The `connector` field defines its type:

* `image` instantiates the image input connector
* `csv` instantiates the input connector for CSV files
* `txt` instantiates the input connector for text files

Input connectors work almost the same during both the training and prediction phases. But the training phase usually deals with large masses of data, and therefore the connectors above are optimized to automate some tasks, typically building and preprocessing the dataset at training time.

## Output connectors

The output connector controls the output formats for supervised and unsupervised models.

Its two main features are the control of the number of predictions per URI, and the output templating, which allows for custom output and seamless integration in external applications. Other options modulates the output format.

The variables that are usable in the output template format are those from the standard JSON output. See the [output template](#output-templates) dedicated section for more details and examples.

# Output Templates

> example of a custom output template:

```
status={{#status}}{{code}}{{/status}}\n{{#body}}{{#predictions}}*{{uri}}:\n{{#classes}}{{cat}}->{{prob}}\n{{/classes}}{{/predictions}}{{/body}}
```

> turn the standard JSON output of a predict call into a custom string output

```shell
curl -X POST "http://localhost:8080/predict" -d "{\"service\":\"imageserv\",\"parameters\":{\"mllib\":{\"gpu\":true},\"input\":{\"width\":224,\"height\":224},\"output\":{\"best\":3,\"template\":\"status={{#status}}{{code}}{{/status}}\n{{#body}}{{#predictions}}*{{uri}}:\n{{#classes}}{{cat}}->{{prob}}\n{{/classes}}{{/predictions}}{{/body}}\"}},\"data\":[\"ambulance.jpg\"]}"
```

> yields:

```
status=200
*ambulance.jpg:
n02701002 ambulance->0.993358
n03977966 police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria->0.00642457
n03769881 minibus->9.11523e-05
```
> instead of:

```json
{"status":{"code":200,"msg":"OK"},"head":{"method":"/predict","time":28.0,"service":"imageserv"},"body":{"predictions":{"uri":"ambulance.jpg","classes":[{"prob":0.993358314037323,"cat":"n02701002 ambulance"},{"prob":0.006424566265195608,"cat":"n03977966 police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria"},{"prob":0.00009115227294387296,"cat":"n03769881 minibus"}]}}}
```

The DeepDetect server and API allow you to ease the connection to your applications through output templates. Output templates are an easy way to customize the output of the `/predict` calls. Take variables from the standard JSON output and reuse their values in the format of your choice.

Instead of decoding the standard JSON output of the DeepDetect server, the API allows to transmit output templates in the [Mustache](https://mustache.github.io/) [format](https://mustache.github.io/mustache.5.html). No more glue code, the server does the job for you! See examples below.

Parameter    | Type   | Optional | Default                        | Description
---------    | ----   | -------- | -------                        | -----------
template     | string | yes      | empty                          | Output template in Mustache format
network      | object | yes      | empty                          | Output network parameters for pushing the output into another listening software

- Network object

Parameter    | Type   | Optional | Default                        | Description
---------    | ----   | -------- | -------                        | -----------
url          | string | no       | N/A                            | URL of the remote service to connect to (e.g http://localhost:9200)
http_method  | string | yes      | POST                           | HTTP connecting method, from "POST", "PUT", etc...
content_type | string | yes      | Content-Type: application/json | Content type HTTP header string

Using Mustache, you can turn the JSON into anything, from XML to specialized formats, with application to indexing into search engines, post-processing, UI rendering, etc...

# Model Templates

> Example of a 3-layer MLP with 512 hidden units in each layer and PReLU activations:

```json
{"parameters":{"mllib":{"template":"mlp","nclasses":9,"layers":[512,512,512],"activation":"PReLU","nclasses":9}}}
```

> Example of GoogleNet for 1000 classes of images:

```json
{"parameters":{"input":{"connector":"image","width":224,"height":224},"mllib":{"template":"googlenet","nclasses":1000}}}
```

The DeepDetect server and API come with a set of Machine Learning model templates.

At the moment templates are available for [Caffe](https://caffe.berkeleyvision.org/) and [Pytorch](https://pytorch.org/) backends. They include some of the most powerful deep neural net architectures for image classification, and other customizable classic and useful architectures.

## Neural network templates

All models below are used by passing their id to the `mllib/template` parameter in `PUT /services` calls:

### Pytorch

#### Native models

- LSTM-like models (including autoencoder): `recurrent`
- NBEATS model: `nbeats`
- Vision transformer: `vit` and `visformer`
- Transformer-based timeseries models: `ttransformer`
- [TorchVision image classification models](https://pytorch.org/vision/0.8/models.html):
	- `resnet18`
	- `resnet34`
	- `resnet50`
	- `resnet101`
	- `resnet152`
	- `resnext50_32x4d`
	- `resnext101_32x8d`
	- `wideresnet50_2`
	- `wideresnet101_2`
	- `alexnet`
	- `vgg11`
	- `vgg13`
	- `vgg16`
	- `vgg19`
	- `vgg11bn`
	- `vgg13bn`
	- `vgg16bn`
	- `vgg19bn`
	- `mobilenetv2`
	- `densenet121`
	- `densenet169`
	- `densenet201`
	- `densenet161`
	- `mnasnet0_5`
	- `mnasnet0_75`
	- `mnasnet1_0`
	- `mnasnet1_3`
	- `shufflenetv2_x0_5`
	- `shufflenetv2_x1_0`
	- `shufflenetv2_x1_5`
	- `shufflenetv2_x2_0`
	- `squeezenet1_0`
	- `squeezenet1_1`

#### Traced models

These templates require an external traced model to work:

- Language models:
	- `bert`
	- `gpt2`
- Detection models:
	- `fasterrcnn`
	- `retinanet`
    - `yolox`
- Segmentation models:
    - `segformer`

## Parameters

Model instantiation parameters for recurrent template (applies to all backends supporting templates):

Parameter       | Template  | Type            | Default                      | Description
---------       | --------- | ------          | ---------------------------- | -----------
layers          | recurrent | array of string | []                           | ["L50","L50"] means 2 layers of LSTMs with hidden size of 50. ["L100","L100", "T", "L300"] means an lstm autoencoder with encoder composed of 2 LSTM layers of hidden size 100 and decoder is one LSTM layer of hidden size 300

### Pytorch

Template parameters for native templates (nbeats/ttransformer):

Parameter       | Template  | Type            | Default                      | Description
---------       | --------- | ------          | ---------------------------- | -----------
template_params.stackdef | nbeats    | array of string | ["t2","s","g3","b3","h10" ] | default means: trend stack with theta = 2, seasonal stack with theta maxed , generic stack with theta = 3, 3 blocks per stacks, hidden unit size of 10 everywhere
template_params.vit_flavor | vit | string | vit_base_patch16 | Vision transformer architecture, from smaller to larger: vit_tiny_patch16, vit_small_patch16, vit_base_patch32, vit_base_patch16, vit_large_patch16, vit_large_patch32, vit_huge_patch16, vit_huge_patch32
template_params.visformer_flavor | visformer | visformer_tiny | Visformer architecture, from visformer_tiny or visformer_small
template_params.realformer | vit | bool | false | Whether to use the 'realformer' residual among attention heads
template_params.positional_encoding.type | ttransformer | string | "sincos" | Positional encoding "sincos for original frequential encoding, "naive" for simple enumeration
template_params.positional_encoding.learn | ttransformer | bool | false | learn or not positional encoding (starting from above value)
template_params.positional_encoding.dropout | ttransformer | float | 0.1 | value of dropout in positional encodin
template_params.embed.layers | ttransformer | int | 3 | Number of layers of MLP value embedder
template_params.embed.activation | ttransformer | string | relu | "relu", "gelu" or "siren" : activation type of MLP embedder
template_params.embed.dim | ttransformer | int | 32 | size of embedding for MLP embedder (per timestep) (embed.dim must be divisible by encoder.heads)
template_params.embed.type | ttransformer | string | step | "step" embeds every step separately, "serie" embeds every serie separately, "all" embeds all timesteps of all serie at once (needs a lot of memory")
template_params.embed.dropout | ttransformer | float | 0.1 | value of dropout in MLP embedder
template_params.encoder.heads | ttransformer | int | 8 | number of heads for transformer encoder (embed.dim must be divisible by encoder.heads)
template_params.encoder.layers | ttransformer | int | 1 | number of layers in transformer encoder
template_params.encoder.hidden_dim | ttransformer | int | input_dim * embed.dim | internal dim of feedfoward net in encoder layer
template_params.encoder.activation | ttransformer | string | relu | "relu" or "gelu"
template_params.encoder.dropout | ttransformer | float | 0.1 | dropout value for encoder stack
template_params.decoder.type | ttransformer | string | simple | simple is a MLP, "transformer" is attention based decoder
template_params.decoder.heads | ttransformer | int | 8 | number of heads for transformer decoder (if any)
template_params.decoder.layers | ttransformer | int | 1 | number of layers of decoder
template_params.decoder.dropout | ttransformer | float | 0.1 | dropout value for decoder stack
template_params.autoreg | ttransformer | bool | false | false for nbeats style decoding, ie gives a window of prediction at one, true for autoregressive, ie predicts value one after the others then use previsouly predicted values as a context

# Errors

The DeepDetect API uses the following error HTTP and associated custom error codes when applicable:


HTTP Status Code | Meaning
---------------- | -------
400              | Bad Request -- Malformed syntax in request or JSON body
403              | Forbidden -- The requested resource or method cannot be accessed
404              | Not Found -- The requested resource, service or model does not exist
409              | Conflict -- The requested method cannot be processed due to a conflict
500              | Internal Server Error -- Other errors, including internal Machine Learning libraries errors

DeepDetect Error Code | Meaning
--------------------- | -------
1000                  | Unknown Library -- The requested Machine Learning library is unknown
1001                  | No Data -- Empty data provided
1002                  | Service Not Found -- Machine Learning service not found
1003                  | Job Not Found -- Training job not found
1004                  | Input Connector Not Found -- Unknown or incompatible input connector and service
1005                  | Service Input Bad Request -- Any service error from input connector
1006                  | Service Bad Request -- Any bad parameter request error from Machine Learning library
1007                  | Internal ML Library Error -- Internal Machine Learning library error
1008                  | Train Predict Conflict -- Algorithm does not support prediction while training
1009                  | Output Connector Network Error -- Output connector has failed to connect to external software via network

# Examples

See the <a href="https://deepdetect.com/overview/examples/">Examples</a> page, and the <a href="https://deepdetect.com/overview/faq/">FAQ</a> for tips and tricks.

Examples include:

* Text: neural networks, from words or low-level characters
* Images: neural networks, visual search
* Generic Data: neural networks, gradient boosted trees, sparse data, ...

Demos include:

* Image classification User Interface, https://github.com/beniz/deepdetect/tree/master/demo/imgdetect
* Visual search, https://github.com/beniz/deepdetect/tree/master/demo/imgsearch
* Integration with Elasticsearch, https://deepdetect.com/tutorials/es-image-classifier
