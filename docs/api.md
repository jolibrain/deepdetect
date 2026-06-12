---
title: API Reference
layout: api
aliases: [/api/]
---

# Introduction

Welcome to the DeepDetect API!

DeepDetect is a deep learning server and runtime. It provides a REST API for
creating services, training models, running predictions, and monitoring jobs in
development and production. The same runtime is also available through the
Python wheel, which provides `import deepdetect` and the `deepdetect` CLI.

## Principles

The Open Source software provides a server, an API, and the underlying Machine Learning procedures for training statistical models. The REST API defines a set of resources and options in order to access and command the server over a network.

### Architecture

The software defines a very simple flow, from data to the statistical model and the final application. The main elements and vocabulary are in that order:

* `data` or `dataset`: images, numerical data, text, time series, sparse data, detection boxes, or segmentation masks
* `input connector`: entry point for data into DeepDetect. Specialized versions handle different data types (e.g. images, text, CSV, ...)
* `model`: repository that holds all the files necessary for building and usage of a statistical model such as a neural net
* `service`: the central holder of models and connectors, living in memory and servicing the machine learning capabilities through the API. While the `model` can be held permanently on disk, a `service` is spawn around it and destroyed at will
* `mllib`: the machine learning library used for operations. Use `torch` for training and inference, or `tensorrt` for optimized inference with exported or compatible models. Optional legacy backends such as `ncnn`, `dlib`, `xgboost`, and `tsne` are build-dependent and may not be available in current packages.

Caffe, Caffe2, TensorFlow, and the `tf` alias are retired. Convert existing services to Torch or ONNX/TensorRT before upgrading; retired names return HTTP 400 with DeepDetect error code 1006.
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

## Python Interfaces

There are two Python entry points:

* The REST client in [clients/python](../clients/python) talks to a running DeepDetect server with `from dd_client import DD`.
* The in-process package in [bindings/python](../bindings/python) embeds the runtime, exposes `import deepdetect`, and powers the `deepdetect` CLI. See the [CLI specification](../bindings/python/deepdetect/cli/CLI_SPEC.md) for packaged `yolox` and `segformer` workflows.

Example in-process usage:

```python
import deepdetect

dd = deepdetect.DeepDetect()
service = dd.create_service(
    "imageserv",
    model={"repository": "/home/me/models/images"},
    mllib="torch",
    input_parameters={"connector": "image", "width": 224, "height": 224, "rgb": True},
    mllib_parameters={"template": "resnet18", "nclasses": 2},
)
result = service.predict(["image.jpg"], output_parameters={"best": 1})
service.delete()
```

# Info

## Get Server Information

```shell
curl -X GET "http://localhost:8080/info"


> The above command returns JSON of the form:

{
	"status":{
		"code":200,
		"msg":"OK"
		},
	"head":
		{
		"method":"/info",
		"version":"0.1",
		"branch":"master",
		"commit":"e8592d5de7f274a82d574025b5a2b647973fccb3",
		"services":[]
		}
}
```

```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

dd.info()

> Result is a dict:

{u'status': {u'msg': u'OK', u'code': 200}, u'head': {u'services': [], u'commit': u'34b9db3dad8c91b165dbcd22d6116fdfe4d78761', u'version': u'0.1', u'method': u'/info', u'branch': u'master'}}

```

Returns general information about the deepdetect server, including the list of existing services.

### HTTP Request

`GET /info`

### Query Parameters

Parameter | Type | Optional | Default | Description
--------- | ---- | -------- | ------- | -----------
status | bool | yes | false  | returns detailed information on every existing services (including training and current statistics)

# Services

Create, get information and delete machine learning services

## Create a service

> Create a Torch image classification service using a ResNet template.

``` shell
curl -X PUT "http://localhost:8080/services/imageserv" -d "{\"mllib\":\"torch\",\"description\":\"example image classification service\",\"type\":\"supervised\",\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":224,\"height\":224,\"rgb\":true,\"scale\":0.0039},\"mllib\":{\"template\":\"resnet18\",\"nclasses\":2}},\"model\":{\"repository\":\"/home/me/models/images\"}}"

# If "/home/me/models/images" correctly exists, the output is

{"status":{"code":201,"msg":"Created"}}
```

> Create a TensorRT inference service from an exported model repository.

``` shell
curl -X PUT "http://localhost:8080/services/trtserv" -d "{\"mllib\":\"tensorrt\",\"description\":\"example optimized inference service\",\"type\":\"supervised\",\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":640,\"height\":640,\"rgb\":true},\"mllib\":{\"template\":\"yolox\",\"nclasses\":80,\"datatype\":\"fp16\",\"maxBatchSize\":8}},\"model\":{\"repository\":\"/home/me/models/yolox-trt\"}}"
```

```python

from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

description = 'example image classification service'

mllib = 'torch'
model = {'repository':'/home/me/models/images'}
parameters_input = {'connector':'image','width':224,'height':224,'rgb':True,'scale':0.0039}
parameters_mllib = {'template':'resnet18','nclasses':2}
parameters_output = {}
dd.put_service('imageserv',model,description,mllib,
               parameters_input,parameters_mllib,parameters_output)

> returns:

{u'status': {u'msg': u'Created', u'code': 201}}

```

Creates a new machine learning service on the server.

### HTTP Request

`PUT /services/<service_name>`

### Query Parameters

#### General

Parameter   | Type   | Optional | Default      | Description
---------   | ----   | -------- | -------      | -----------
mllib       | string | No       | N/A          | Name of the machine learning library. Recommended values are `torch` and `tensorrt`; optional legacy/build-dependent values include `ncnn`, `dlib`, `xgboost`, and `tsne`. Removed values `caffe`, `caffe2`, `tf`, and `tensorflow` return error 1006.
type        | string | No       | `supervised` | Machine Learning service type: `supervised` yields a series of metrics related to a supervised objective, or `unsupervised`, typically for state-space compression or accessing neural network's inner layers.
description | string | yes      | empty        | Service description
model       | object | No       | N/A          | Information for the statistical model to be built and/or used by the service
input       | object | No       | N/A          | Input information for connecting to data
output      | object | yes      | empty        | Output information

- Model Object

Parameter         | Type   | Optional | Default   | Description
---------         | ----   | -------- | -------   | -----------
repository        | string | No       | N/A       | Repository for the statistical model files
weights           | string | yes      | empty     | Weights filename of a pre-trained network (e.g. for finetuning or resuming a net)
create_repository | bool   | yes      | false     | Whether to create the model repository directory if it does not exist already
index_preload     | bool   | yes      | false     | Whether to preload a similarity search index

#### Connectors

- Input Object

Parameter | Type   | Optional | Default | Description
--------- | ----   | -------- | ------- | -----------
connector | string | No       | N/A     | Input data format, e.g. `image`, `txt`, `csv`, `csvts`, `svm`, or `video` depending on the selected backend
timeout   | int    | yes      | 6000    | timeout on all predict calls for data retrieval

Image (`image`)

Parameter      | Type         | Optional | Default | Description
---------      | ----         | -------- | ------- | -----------
width          | int          | yes      | 227     | Resize images to width (`image` only)
height         | int          | yes      | 227     | Resize images to height (`image` only)
bw             | bool         | yes      | false   | Treat images as black & white
rgb	       | bool	      | yes 	 | false   | Use RGB images
histogram_equalization | bool | yes      | false   | Whether to equalize the image histogram
mean           | float        | yes      | 128     | mean pixel value to be subtracted to input image
mean           | array of int | yes      | N/A     | mean pixel value per channel to be subtracted to input image
std            | float        | yes      | 128     | standard pixel value deviation to be applied to input image
scale	       | float	      | yes	 | 1.0	   | Multiply value of each pixel by a factor. By default pixel values are between 0 and 255.
scale_min      | float        | yes	 | 600     | min scaling dim size
scale_max      | float        | yes      | 1000    | max scaling dim size
segmentation   | bool         | yes      | false   | whether to setup an image connector for a segmentation task
multi_label    | bool         | yes      | false   | whether to setup a multi label image task
root_folder    | string       | yes      | false   | root folder for image data layer
ctc            | bool         | yes      | false   | whether using a sequence target, required for OCR tasks
unchanged_data | bool         | yes      | false   | do not allow data modification (e.g. interpolation upon resizing, ...). Useful for audio spectrogram as input images.
bbox           | bool         | yes      | false   | whether to setup an object detection model

CSV (`csv`)

Parameter    | Type            | Optional | Default | Description
---------    | ----            | -------- | ------- | -----------
label        | string          | no       | N/A     | Label column name
ignore       | array of string | yes      | empty   | Array of column names to ignore
label_offset | int             | yes      | 0       | Negative offset (e.g. -1) so that labels range from 0 onward
separator    | string          | yes      | ','     | Column separator character
quote	     | string	       | yes	  | '"'	    | Quote character in CSV file
id           | string          | yes      | empty   | Column name of the training examples identifier field, if any
scale        | bool            | yes      | false   | Whether to scale all values internally into uniform range
scale_type   | string          | yes      | "minmax" | scaling type in "minmax", "znorm"
categoricals | array           | yes      | empty   | List of categorical variables
db           | bool            | yes      | false   | whether to gather data into a database, useful for very large datasets, allows treatment in constant-size memory

CSV Time-series (`csvts`)

Parameter | Type            | Optional | Default | Description
--------- | ----            | -------- | ------- | -----------
label     | string          | no       | N/A     | Label column name
ignore    | array of string | yes      | empty   | Array of column names to ignore
separator | string          | yes      | ','     | Column separator character
quote	  | string	       | yes	  | '"'	    | Quote character in CSV file
id        | string          | yes      | empty   | Column name of the training examples identifier field, if any
scale     | bool            | yes      | false   | Whether to scale all values 
scale_type | string          | yes      | "minmax" | scaling type in "minmax" (scales into [-0.5,0.5]), "znorm"
db        | bool            | yes      | false   | whether to gather data into a database, useful for very large datasets, allows treatment in constant-size memory


Text (`txt`)

Parameter          | Type   | Optional | Default                                            | Description
---------          | ----   | -------- | -------                                            | -----------
sentences          | bool   | yes      | false                                              | whether to turn every line into a document (requires dataset as file with one sentence per line in every class repository)
characters         | bool   | yes      | false                                              | character-level text processing, as opposed to word-based text processing
sequence           | int    | yes      | N/A                                                | for character-level text processing, the fixed length of each sample of text
read_forward       | bool   | yes      | false                                              | for character-level text processing, whether to read content from left to right
alphabet           | string | yes      | abcdefghijklmnopqrstuvwxyz 0123456789 ,;.!?:'"/\\\ \|_@#$%^&*~\`+-=<>()[]{} | for character-level text processing, the alphabet of recognized symbols
sparse             | bool   | yes      | false                                              | whether to use sparse text features
ordered_words      | bool   | yes      | false                                              | enable word-based processing with positional information, mandatory for bert/gpt2 like models
wordpiece_tokens   | bool   | yes      | false                                              | set to true if vocabulary contains partial words, ie like in bert/gpt2 models
punctuation_tokens | bool   | yes      | false                                              | if true, treat each punctuation sign as a token; if false, punctuation is stripped from input
word_start         | string | yes      | ""                                                 | in most gpt2 vocabularies, start of word has generally to be set to "Ġ".
suffix_start       | string | yes      | "##"                                               | in bert-like vocabularies, suffixes are prefixed by `##`

SVM (`svm`)

No parameters

See the section on [Connectors](#connectors) for more details.

#### Machine learning libraries

- Torch

Torch is the primary backend for training and inference.

Parameter          | Type         | Optional | Default | Description
---------          | ----         | -------- | ------- | -----------
nclasses           | int          | no, classification/detection/segmentation | N/A | Number of output classes
ntargets           | int          | no, regression only | N/A | Number of regression targets
template           | string       | yes      | empty   | Model template, for example `resnet18`, `yolox`, `fasterrcnn`, `retinanet`, `bert`, `gpt2`, `recurrent`, `nbeats`, `vit`, `visformer`, or `ttransformer`
template_params    | object       | yes      | template-dependent | Template-specific parameters, listed in [Model Templates](#model-templates)
segmentation       | bool         | yes      | false   | Whether the model is a segmentation model
ctc                | bool         | yes      | false   | Whether the model uses CTC sequence output
gpu                | bool         | yes      | false   | Whether to use GPU
gpuid              | int or array | yes      | 0       | GPU id, `-1` for all GPUs, or an array such as `[1,3]`
datatype           | string       | yes      | fp32    | Prediction datatype: `fp32`, `fp16`, or `fp64`
multi_label        | bool         | yes      | false   | Model outputs an independent score for each class
concurrent_predict | bool         | yes      | true    | Enable concurrent prediction calls
from_repository    | string       | yes      | empty   | Initialize model repository from another repository
resume_from        | string       | yes      | empty   | At service creation, load checkpoint state from `latest` or `best`
finetuning         | bool         | yes      | false   | Prepare a neural net template for finetuning with a different head
freeze_traced      | bool         | yes      | false   | Freeze the traced part of a model during finetuning
loss               | string       | yes      | empty   | Special loss, for example for time series, detection, or segmentation templates
timesteps          | int          | yes      | -1      | Number of output timesteps, `-1` for automatic mode
embedding_size     | int          | yes      | 768     | Embedding size for NLP models
self_supervised    | string       | yes      | empty   | Self-supervised mode, for example `mask`
jit_compiler_params | object      | yes      | empty   | Experimental Torch JIT compiler settings

- XGBoost (optional legacy backend)

XGBoost remains available only in builds compiled with XGBoost support. Prefer
Torch for new deep learning services.

Parameter  | Type | Optional                 | Default | Description
---------  | ---- | --------                 | ------- | -----------
nclasses   | int  | no (classification only) | N/A     | Number of output classes (`supervised` service type)
ntargets   | int  | no (regression only)     | N/A     | Number of regression targets (only 1 supported by XGBoost)
regression | bool | yes                      | false   | Whether to train a regressor

- NCNN (optional legacy backend)

NCNN is build-dependent and retained for compatibility with existing deployments.

Parameter  | Type   | Optional | Default                                                                 | Description
---------  | ----   | -------- | -------                                                                 | -----------
inputblob  | string | yes      | data                                                                    | network input blob name
outputblob | string | yes      | depends on network type (ie prob or rnn_pred or probs or detection_out) | network output blob name

- TensorRT

TensorRT is an optimized inference backend for exported or compatible models.

Parameter          | Type   | Optional | Default     | Description
---------          | ----   | -------- | -------     | -----------
tensorRTEngineFile | string | yes      | "TRTengine" | prefix of filename of TRT compiled model (complete name defaults to   "TRTengine_bs48")
readEngine         | bool   | yes      | true        | if a compiled model file with corresponding prefix, whatever batch size exists in repo, use it instead of recompiling a TRT model
writeEngine        | bool   | yes      | true        | if a new TRT model was compiled, write it to disk
maxWorkspaceSize   | int    | yes      | 1024        | max memory (in MB) usable by TRT during model compilation. Useful mainly on constrained devices: 1024 is 1GB and may cause the process to be killed if not enough memory is available. 256 limits memory consumption and creates lower-batch-size networks.
maxBatchSize       | int    | yes      | 48          | maximum batch size processable by TRT compiled model.  If a precompiled engine starting with tensorRTEngineFile name is present and readEngine is set to true , this previous engine is used, overriding this option.
dla                | int    | true     | -1          | id of DLA to use, if available on your hardware
datatype           | string | true     | "fp32"      | datatype inside compiled TRT model: `fp32`, `fp16`, `int32`, or `int8`. INT8 requires calibration data.
gpuid              | int    | yes      | 0           | GPU id for TensorRT inference
calibration        | bool   | yes      | false       | Whether a prediction request should be used to calibrate an INT8 model

- Output Object

Parameter    | Type | Optional | Default | Description
---------    | ---- | -------- | ------- | -----------
measure      | array of string | yes | depending on problem type | measure to use at test time


Problem type | Default | Possible values | Description
------------ | ------- | --------------- | -----------
timeserie    |   L1    | L1, L2, mase, mape, smape, mase, owa, mae, mse; L1_all, L2_all, mase_all, mape_all, smape_all, mase_all, owa_all, mae_all, mse_all | L1: mean error, L2: mean squared error, mase: mean absolute scaled error, mape: mean absolute percentage error, smape: symmetric mean absolute percentage error, owa: overall weighted average, mae: mean absolute error, mse: mean squared error; versions with `_all` also show metrics per dimension/series, not only averages.



## Get information on a service

```shell
curl -X GET "http://localhost:8080/services/imageserv"

> Assuming the service `imageserv` was previously created, yields

{
  "status":{
	     "code":200,
	     "msg":"OK"
	  },
  "body":{
	     "mllib":"torch",
	     "description":"example image classification service",
	     "name":"imageserv",
	     "jobs":
		{
		  "job":1,
		  "status":"running"
		}
	 }
}
```
```python

from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

dd.get_service('imageserv')

> returns:

{u'status': {u'msg': u'OK', u'code': 200}, u'body': {u'jobs': {}, u'mllib': u'torch', u'name': u'imageserv', u'description': u'example image classification service'}}
```

Returns information on an existing service

### HTTP Request

`GET /services/imageserv`

### Query Parameters

None

## Delete a service

```shell
curl -X DELETE "http://localhost:8080/services/imageserv?clear=full"

> Yields

{"status":{"code":200,"msg":"OK"}}

```

```python

from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

dd.delete_service('imageserv',clear='full')

```

### HTTP Request

`DELETE /services/imageserv`

### Query Parameters

Parameter | Type   | Optional | Default | Description
--------- | ----   | -------- | ------- | -----------
clear     | string | yes      | mem     | `full`, `lib`, `mem`, `dir` or `index`. `full` clears the model and service repository, `lib` removes model files only according to the behavior specified by the service's ML library, `mem` removes the service from memory without affecting the files, `dir` removes the whole directory, `index` removes the index when using similarity search.

# Train

Trains a statistical model from a dataset, the model can be further used for prediction

The DeepDetect server supports both blocking and asynchronous training calls. Training is often a very computational operation that can last for days in some cases.

Blocking calls block the communication with the server, and returns results once completed. They are not well suited to most machine learning tasks.

Asynchronous calls run the training in the background as a separate thread (`PUT /train`). Status of the training job can be consulted live with by calling on the server (`GET /train`). The final report on an asynchronous training job is consumed by the first `GET /train` call after completion of the job. After that, the job is definitely destroyed.

<div class="alert alert-primary mx-2" style="width: 58%">
⚠️ Asynchronous training calls are the default, use of blocking calls is useful for testing and debugging
</div>

<div class="alert alert-danger mx-2" style="width: 58%">
</div>

## Launch a training job

> Blocking train call from an image dataset

```shell
curl -X POST "http://127.0.0.1:8080/train" -d "{\"service\":\"imageserv\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":300,\"test_interval\":100,\"base_lr\":0.0001,\"solver_type\":\"ADAM\"},\"net\":{\"batch_size\":4},\"resume\":false},\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true,\"test_split\":0.15},\"output\":{\"measure\":[\"acc\",\"f1\"]}},\"data\":[\"/home/me/datasets/images/train\"]}"

{"status":{"code":201,"msg":"Created"},"body":{"measure":{"iteration":300.0,"train_loss":0.6463099718093872,"f1":0.75,"acc":0.7675070028011205}},"head":{"method":"/train","time":403.0}}

```

```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

parameters_input = {'seed':12345,'db':True,'shuffle':True,'test_split':0.15}
parameters_mllib = {
    'gpu':True,
    'solver':{'iterations':300,'test_interval':100,'base_lr':0.0001,'solver_type':'ADAM'},
    'net':{'batch_size':4},
    'resume':False,
}
parameters_output = {'measure':['acc','f1']}
train_data = ['/home/me/datasets/images/train']

dd.post_train('imageserv',train_data,parameters_input,parameters_mllib,parameters_output,jasync=False)
```

> Asynchronous train call from an image dataset


```shell
curl -X POST "http://127.0.0.1:8080/train" -d "{\"service\":\"imageserv\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":100000,\"test_interval\":1000,\"base_lr\":0.0001,\"solver_type\":\"ADAM\"},\"net\":{\"batch_size\":4}},\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true,\"test_split\":0.15},\"output\":{\"measure\":[\"acc\",\"f1\"]}},\"data\":[\"/home/me/datasets/images/train\"]}"

{"status":{"code":201,"msg":"Created"},"head":{"method":"/train","job":1,"status":"running"}}
```

```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

parameters_input = {'seed':12345,'db':True,'shuffle':True,'test_split':0.15}
parameters_mllib = {
    'gpu':True,
    'solver':{'iterations':100000,'test_interval':1000,'base_lr':0.0001,'solver_type':'ADAM'},
    'net':{'batch_size':4},
}
parameters_output = {'measure':['acc','f1']}
train_data = ['/home/me/datasets/images/train']

dd.post_train('imageserv',train_data,parameters_input,parameters_mllib,parameters_output,jasync=True)

```

> Requesting the status of an asynchronous training job:

```shell
curl -X GET "http://localhost:8080/train?service=imageserv&job=1"

{"status":{"code":200,"msg":"OK"},"head":{"method":"/train","job":1,"status":"running","time":74.0},"body":{"measure":{"iteration":445.0,"train_loss":0.7159726023674011,"f1":0.72,"acc":0.76}}}
```
```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

dd.get_train('imageserv',job=1)
```

Launches a blocking or asynchronous training job from a service

### HTTP Request

`PUT or POST /train`

### Query Parameters

#### General

Parameter | Type   | Optional | Default | Description
--------- | ----   | -------- | ------- | -----------
service   | string | No       | N/A     | service resource identifier
async     | bool   | No       | true    | whether to start a non-blocking training call
data      | object | yes      | empty   | input dataset for training, in some cases can be handled by the input connectors, in general non optional though

#### Input Connectors

- Image (`image`)

Parameter    | Type | Optional | Default | Description
---------    | ---- | -------- | ------- | -----------
width        | int  | yes      | 227     | Resize images to width (`image` only)
height       | int  | yes      | 227     | Resize images to height (`image` only)
bw           | bool | yes      | false   | Treat images as black & white
rgb	     | bool	       | yes 	 | false   | Use RGB images
histogram_equalization | bool | yes      | false   | Whether to equalize the image histogram
mean           | float        | yes      | 128     | mean pixel value to be subtracted to input image
mean           | array of int | yes      | N/A     | mean pixel value per channel to be subtracted to input image
std            | float        | yes      | 128     | standard pixel value deviation to be applied to input image
scale	       | float	      | yes	 | 1.0	   | Multiply value of each pixel by a factor. By default pixel values are between 0 and 255.
scale_min      | float        | yes	 | 600     | min scaling dim size
scale_max      | float        | yes      | 1000    | max scaling dim size
test_split   | real | yes      | 0       | Test split part of the dataset
shuffle      | bool | yes      | false   | Whether to shuffle the training set (prior to splitting)
seed         | int  | yes      | -1      | Shuffling seed for reproducible results (-1 for random seeding)
segmentation | bool | yes      | false   | whether to setup an image connector for a segmentation training job
bbox         | bool | yes      | false   | whether to setup an image connector for an object detection training job
db_width     | int  | yes      | 0       | in database image width (object detection only)
db_height    | int  | yes      | 0       | in database image height (object detection only)
align        | bool | yes      | false   | for ocr tasks only, align width on highest dimension
scale_min    | int  | yes      | N/A     | image auto min scaling
scale_max    | int  | yes      | N/A     | image auto max scaling

- CSV (`csv`)

Parameter            | Type            | Optional | Default | Description
---------            | ----            | -------- | ------- | -----------
label                | string          | no       | N/A     | Label column name
ignore               | array of string | yes      | empty   | Array of column names to ignore
label_offset         | int             | yes      | 0       | Negative offset (e.g. -1) so that labels range from 0 onward
separator            | string          | yes      | ','     | Column separator character
quote	             | string	       | yes	  | '"'	    | Quote character in CSV file
id                   | string          | yes      | empty   | Column name of the training examples identifier field, if any
scale                | bool            | yes      | false   | Whether to scale all values into [0,1]
min_vals,max_vals    | array           | yes      | empty   | Instead of `scale`, provide the scaling parameters, as returned from a training call
categoricals         | array           | yes      | empty   | List of categorical variables
categoricals_mapping | object          | yes      | empty   | Categorical mappings, as returned from a training call
db                   | bool            | yes      | false   | whether to gather data into a database, useful for very large datasets, allows training in constant-size memory
test_split           | real            | yes      | 0       | Test split part of the dataset
shuffle              | bool            | yes      | false   | Whether to shuffle the training set (prior to splitting)
seed                 | int             | yes      | -1      | Shuffling seed for reproducible results (-1 for random seeding)

- CSV Time-series (`csvts`)

Parameter         | Type            | Optional | Default | Description
---------         | ----            | -------- | ------- | -----------
label             | string          | no       | N/A     | Label column name
ignore            | array of string | yes      | empty   | Array of column names to ignore
label_offset      | int             | yes      | 0       | Negative offset (e.g. -1) so that labels range from 0 onward
separator         | string          | yes      | ','     | Column separator character
quote	          | string	       | yes	  | '"'	    | Quote character in CSV file
id                | string          | yes      | empty   | Column name of the training examples identifier field, if any
scale             | bool            | yes      | false   | Whether to scale all values into [0,1]
min_vals,max_vals | array           | yes      | empty   | Instead of `scale`, provide the scaling parameters, as returned from a training call
db                | bool            | yes      | false   | whether to gather data into a database, useful for very large datasets, allows training in constant-size memory
test_split        | real            | yes      | 0       | Test split part of the dataset
shuffle           | bool            | yes      | false   | Whether to shuffle the training set (prior to splitting)
seed              | int             | yes      | -1      | Shuffling seed for reproducible results (-1 for random seeding)


- Text (`txt`)

Parameter          | Type   | Optional | Default                                            | Description
---------          | ----   | -------- | -------                                            | -----------
count              | int    | yes      | true                                               | whether to count words and report counters
min_count          | int    | yes      | 5                                                  | min word count occurrences for a word to be taken into account
min_word_length    | int    | yes      | 5                                                  | min word length for a word to be taken into account
tfidf              | bool   | yes      | false                                              | whether to compute TF/IDF for every word
sentences          | bool   | yes      | false                                              | whether to turn every line into a document (requires dataset as file with one sentence per line in every class repository)
characters         | bool   | yes      | false                                              | character-level text processing, as opposed to word-based text processing
sequence           | int    | yes      | N/A                                                | for character-level text processing, the fixed length of each sample of text
read_forward       | bool   | yes      | false                                              | for character-level text processing, whether to read content from left to right
alphabet           | string | yes      | abcdefghijklmnopqrstuvwxyz 0123456789 ,;.!?:'"/\\\ \|_@#$%^&*~\`+-=<>()[]{} | for character-level text processing, the alphabet of recognized symbols
test_split         | real   | yes      | 0                                                  | Test split part of the dataset
shuffle            | bool   | yes      | false                                              | Whether to shuffle the training set (prior to splitting)
seed               | int    | yes      | -1                                                 | Shuffling seed for reproducible results (-1 for random seeding)
db                 | bool   | yes      | false                                              | whether to gather data into a database, useful for very large datasets, allows training in constant-size memory
sparse             | bool   | yes      | false                                              | whether to use sparse text features
embedding          | bool   | yes      | false                                              | whether to use an embedding as input to the network (replaces one-hot vectors with straight indices)
ordered_words      | bool   | yes      | false                                              | enable word-based processing with positional information, mandatory for bert/gpt2 like models
wordpiece_tokens   | bool   | yes      | false                                              | set to true if vocabulary contains partial words, ie like in bert/gpt2 models
punctuation_tokens | bool   | yes      | false                                              | if true, treat each punctuation sign as a token; if false, punctuation is stripped from input
word_start         | string | yes      | ""                                                 | in most gpt2 vocabularies, start of word has generally to be set to "Ġ".
suffix_start       | string | yes      | "##"                                               | in most bert-like vocabularies, suffixes are prefixed by `##`


- SVM (`svm`)

No parameters

#### Output connector

Parameter         | Type   | Optional | Default | Description
---------         | ----   | -------- | ------- | -----------
best              | int    | yes      | 1       | Number of top predictions returned by data URI (supervised)
measure           | array  | yes      | empty   | Output measures requested, from `acc`: accuracy, `acc-k`: top-k accuracy, replace k with number (e.g. `acc-5`), `f1`: f1, precision and recall, `mcll`: multi-class log loss, `auc`: area under the curve, `cmdiag`: diagonal of confusion matrix (requires `f1`), `cmfull`: full confusion matrix (requires `f1`), `mcc`: Matthews correlation coefficient, `eucll`: euclidean distance (e.g. for regression tasks),`l1`: l1 distance (e.g. for regression tasks), `percent`: mean relative error in percent,  `kl`: KL_divergence, `js`: JS divergence, `was`: Wasserstein, `ks`: Kolmogorov Smirnov, `dc`: distance correlation, `r2`: R2, `deltas`: delta scores, 'raw': ouput raw results, in case of predict call, this requires a special deploy.prototxt that is a test network (to have ground truth)
target_repository | string | yes      | empty   | target directory to which to copy the best model files once training has completed

#### Machine learning libraries

- Torch

Torch training accepts the general Torch service options plus training-specific
solver, network, resume, and augmentation parameters.

Parameter          | Type   | Optional | Default | Description
---------          | ----   | -------- | ------- | -----------
solver.iterations  | int    | no       | N/A     | Number of training iterations
solver.test_interval | int | yes      | backend default | Number of iterations between test passes
solver.base_lr     | float  | yes      | backend default | Base learning rate
solver.solver_type | string | yes      | backend default | Optimizer, for example `ADAM`
solver.iter_size   | int    | yes      | 1       | Gradient accumulation size
net.batch_size     | int    | yes      | backend default | Training batch size
net.test_batch_size | int   | yes      | net.batch_size | Test batch size
resume             | bool   | yes      | false   | Resume training from the service repository
resume_from        | string | yes      | empty   | Resume checkpoint selector, usually `latest` or `best`
mirror             | bool   | yes      | false   | Enable horizontal mirroring augmentation
rotate             | bool   | yes      | false   | Enable rotation augmentation
crop_size          | int    | yes      | 0       | Training crop size
test_crop_samples  | int    | yes      | 1       | Number of crop samples at test time
cutout             | float  | yes      | 0       | Cutout augmentation probability/strength
geometry           | object | yes      | empty   | Perspective, zoom, and padding augmentation options
noise              | object | yes      | empty   | Noise augmentation options
distort            | object | yes      | empty   | Distortion augmentation options
dataloader_threads | int    | yes      | backend default | Number of dataloader threads

- XGBoost (optional legacy backend)

XGBoost remains available only in builds compiled with XGBoost support.

General:

Parameter     | Type   | Optional | Default                | Description
---------     | ----   | -------- | -------                | -----------
objective     | string | yes      | multi:softprob         | objective function, among multi:softprob, binary:logistic, reg:linear, reg:logistic
booster       | string | yes      | gbtree                 | which booster to use, gbtree or gblinear
num_feature   | int    | yes      | set by xgbbost         | maximum dimension of the feature
eval_metric   | string | yes      | according to objective | evaluation metric internal to XGBoost
base_score    | double | yes      | 0.5                    | initial prediction score, global bias
seed          | int    | yes      | 0                      | random number seed
iterations    | int    | no       | N/A                    | number of boosting iterations
test_interval | int    | yes      | 1                      | number of iterations between each testing pass
save_period   | int    | yes      | 0                      | number of iterations between model saving to disk

Booster_params:

Parameter        | Type   | Optional | Default | Description
---------        | ----   | -------- | ------- | -----------
eta              | double | yes      | 0.3     | step size shrinkage
gamma            | double | yes      | 0       | minimum loss reduction
max_depth        | int    | yes      | 6       | maximum depth of a tree
min_child_weight | int    | yes      | 1       | minimum sum of instance weight
max_delta_step   | int    | yes      | 0       | maximum delta step
subsample        | double | yes      | 1.0     | subsample ratio of training instances
colsample        | double | yes      | 1.0     | subsample ratio of columns when constructing each tree
lambda           | double | yes      | 1.0     | L2 regularization term on weights
alpha            | double | yes      | 0.0     | L1 regularization term on weights
lambda_bias      | double | yes      | 0.0     | L2 regularization for linear booster
tree_method      | string | yes      | auto    | tree construction algorithm, from auto, exact, approx
scale_pos_weight | double | yes      | 1.0     | control the balance of positive and negative weights

For more details on all XGBoost parameters see the XGBoost documentation.

- TSNE (optional legacy backend)

TSNE remains available only in builds compiled with TSNE support.

Parameter  | Type | Optional | Default | Description
---------  | ---- | -------- | ------- | -----------
perplexity | int  | yes      | 30      | perplexity is related to the number of nearest neighbors used to learn the manifold
iterations | int  | yes      | 5000    | number of optimization iterations


## Get information on a training job

> Requesting the status of an asynchronous training job:

```shell
curl -X GET "http://localhost:8080/train?service=imageserv&job=1"

{"status":{"code":200,"msg":"OK"},"head":{"method":"/train","job":1,"status":"running","time":74.0},"body":{"measure":{"iteration":445.0,"train_loss":0.7159726023674011,"f1":0.72,"acc":0.76}}}
```
```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

dd.get_train('imageserv',job=1)
```

Returns information on a training job running asynchronously

### HTTP Request

`GET /train`

### Query Parameters

Parameter                         | Type   | Optional | Default | Description
---------                         | ----   | -------- | ------- | -----------
service                           | string | no       | N/A     | name of the service the training job is running on
job                               | int    | no       | N/A     | job identifier
timeout                           | int    | yes      | 0       | timeout before the status is obtained
parameters.output.measure_hist    | bool   | yes      | false   | whether to return the full measure history until current point, useful for plotting
parameters.output.max_hist_points | int    | yes      | 10000   | max number of measure history points (subsampled from history)

## Delete a training job

```shell
curl -X DELETE "http://localhost:8080/train?service=imageserv&job=1"

{"status":{"code":200,"msg":"OK"},"head":{"time":196.0,"status":"terminated","method":"/train","job":1}}
```
```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

dd.delete_train('imageserv',job=1)
```

Kills a training job running asynchronously

### HTTP Request

`DELETE /train`

### Query Parameters

Parameter | Type | Optional | Default | Description
--------- | ---- | -------- | ------- | -----------
service | string | no | N/A | name of the service the training job is running on
job | int | no | N/A | job identifier


# Predict

Makes predictions from data out of an existing statistical model. If `measure` is specified, the prediction expects a supervised dataset and produces accuracy measures as output, otherwise it is prediction for every of the input samples.

## Prediction from service

> Prediction from image URL:

```shell
curl -X POST "http://localhost:8080/predict" -d "{\"service\":\"imageserv\",\"parameters\":{\"input\":{\"width\":224,\"height\":224},\"output\":{\"best\":3}},\"data\":[\"http://i.ytimg.com/vi/0vxOhd4qlnA/maxresdefault.jpg\"]}"

{"status":{"code":200,"msg":"OK"},"head":{"method":"/predict","time":1591.0,"service":"imageserv"},"body":{"predictions":{"uri":"http://i.ytimg.com/vi/0vxOhd4qlnA/maxresdefault.jpg","loss":0.0,"classes":[{"prob":0.24278657138347627,"cat":"n03868863 oxygen mask"},{"prob":0.20703653991222382,"cat":"n03127747 crash helmet"},{"prob":0.07931024581193924,"cat":"n03379051 football helmet"}]}}}
```
```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

data = ['http://i.ytimg.com/vi/0vxOhd4qlnA/maxresdefault.jpg']
parameters_input = {'width':224,'height':224}
parameters_mllib = {'gpu':False}
parameters_output = {'best':3}

predict_output = dd.post_predict('imageserv',data,parameters_input,parameters_mllib,parameters_output)
```

> Prediction from CSV file:

```shell
curl -X POST "http://localhost:8080/predict" -d "{\"service\":\"covert\",\"parameters\":{\"input\":{\"id\":\"Id\",\"separator\":\",\",\"scale\":true}},\"data\":[\"models/covert/test10.csv\"]}"

{"status":{"code":200,"msg":"OK"},"head":{"method":"/predict","time":16.0,"service":"covert"},"body":{"predictions":[{"uri":"15121","loss":0.0,"classes":{"prob":0.9999997615814209,"cat":"6"}},{"uri":"15122","loss":0.0,"classes":{"prob":0.9962882995605469,"cat":"5"}},{"uri":"15130","loss":0.0,"classes":{"prob":0.9999340772628784,"cat":"1"}},{"uri":"15123","loss":0.0,"classes":{"prob":1.0,"cat":"3"}},{"uri":"15124","loss":0.0,"classes":{"prob":1.0,"cat":"3"}},{"uri":"15128","loss":0.0,"classes":{"prob":1.0,"cat":"1"}},{"uri":"15125","loss":0.0,"classes":{"prob":0.9999998807907105,"cat":"3"}},{"uri":"15126","loss":0.0,"classes":{"prob":0.7535045146942139,"cat":"3"}},{"uri":"15129","loss":0.0,"classes":{"prob":0.9999986886978149,"cat":"1"}},{"uri":"15127","loss":0.0,"classes":{"prob":1.0,"cat":"1"}}]}}
```

> Prediction over test set, with output metrics

```shell
curl -X POST 'http://localhost:8080/predict' -d '{"service":"n20","parameters":{"mllib":{"gpu":true},"output":{"measure":["f1"]}},"data":["/path/to/news20/"]}'

{"status":{"code":200,"msg":"OK"},"head":{"method":"/predict","time":18271.0,"service":"n20"},"body":{"measure":{"f1":0.8152690151793434,"recall":0.8219119954158582,"precision":0.8087325557838578,"accp":0.815365025466893}}}
```

```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

data = ['models/covert/test10.csv']
parameters_input = {'id':'id','separator':',','scale':True}
parameters_mllib = {'gpu':True}
parameters_output = {}

predict_output = dd.post_predict('covert',data,parameters_input,parameters_mllib,parameters_output)
```

Make predictions from data

### HTTP Request

`POST /predict`

### Query Parameters

#### General

Parameter | Type             | Optional | Default | Description
--------- | ----             | -------- | ------- | -----------
service   | string           | no       | N/A     | name of the service to make predictions from
data      | array of strings | no       | N/A     | array of data URI over which to make predictions, supports base64 for images

#### Input Connectors

Note: it is good practice to configure the `input` connector at service creation, and then leave its parameters empty at `predict` time.

Parameter | Type | Optional | Default | Description
--------- | ---- | -------- | ------- | -----------
timeout   | int  | yes      | 6000    | timeout on predict call for data retrieval

- Image (`image`)

Parameter    | Type         | Optional | Default | Description
---------    | ----         | -------- | ------- | -----------
width        | int          | yes      | 227     | Resize images to width (`image` only)
height       | int          | yes      | 227     | Resize images to height (`image` only)
crop_width   | int          | yes      | 0       | Center crop images to width (`image` only)
crop_height  | int          | yes      | 0       | Center crop images to height (`image` only)
bw           | bool         | yes      | false   | Treat images as black & white
mean         | float        | yes      | 128     | mean pixel value to be subtracted to input image
mean         | array of int | yes      | N/A     | mean pixel value per channel to be subtracted to input image
std          | float        | yes      | 128     | standard pixel value deviation to be applied to input image
segmentation | yes          | yes      | false   | whether a segmentation service
interp       | string       | yes      | cubic   | Image interpolation method (cubic, linear, nearest, lanczos4, area)
cuda         | bool         | yes      | false   | Whether to use CUDA to resize images (use USE_CUDA_CV=ON build flag)

- CSV (`csv`)

Parameter            | Type            | Optional | Default | Description
---------            | ----            | -------- | ------- | -----------
ignore               | array of string | yes      | empty   | Array of column names to ignore
separator            | string          | yes      | ','     | Column separator character
quote	             | string	       | yes	  | '"'	    | Quote character in CSV file
id                   | string          | yes      | empty   | Column name of the training examples identifier field, if any
scale                | bool            | yes      | false   | Whether to scale all values into [0,1]
min_vals,max_vals    | array           | yes      | empty   | Instead of `scale`, provide the scaling parameters, as returned from a training call
categoricals_mapping | object          | yes      | empty   | Categorical mappings, as returned from a training call

- Text (`txt`)

Parameter       | Type   | Optional | Default                                            | Description
---------       | ----   | -------- | -------                                            | -----------
count           | int    | yes      | true                                               | whether to count words and report counters
min_count       | int    | yes      | 5                                                  | min word count occurrences for a word to be taken into account
min_word_length | int    | yes      | 5                                                  | min word length for a word to be taken into account
tfidf           | bool   | yes      | false                                              | whether to compute TF/IDF for every word
sentences       | bool   | yes      | false                                              | whether to turn every line into a document (requires dataset as file with one sentence per line in every class repository)
characters      | bool   | yes      | false                                              | character-level text processing, as opposed to word-based text processing
sequence        | int    | yes      | N/A                                                | for character-level text processing, the fixed length of each sample of text
read_forward    | bool   | yes      | false                                              | for character-level text processing, whether to read content from left to right
alphabet        | string | yes      | abcdefghijklmnopqrstuvwxyz 0123456789 ,;.!?:'"/\\\ \|\_@#$%^&\*~\`+-=<>()[]{} | for character-level text processing, the alphabet of recognized symbols
sparse          | bool   | yes      | false                                              | whether to use sparse text features

- SVM (`svm`)

No parameters

#### Output

Parameter            | Type   | Optional | Default                 | Description
---------            | ----   | -------- | -------                 | -----------
best                 | int    | yes      | 1                       | Number of top predictions returned by data URI (supervised)
template             | string | yes      | empty                   | Output template in Mustache format
network              | object | yes      | empty                   | Output network parameters for pushing the output into another listening software
measure              | array  | yes      | empty                   | Output measures requested, from `acc`: accuracy, `acc-k`: top-k accuracy, replace k with number (e.g. `acc-5`), `f1`: f1, precision and recall, `mcll`: multi-class log loss, `auc`: area under the curve, `cmdiag`: diagonal of confusion matrix (requires `f1`), `cmfull`: full confusion matrix (requires `f1`), `mcc`: Matthews correlation coefficient
confidence_threshold | double | yes      | 0.0                     | only returns classifications or detections with probability strictly above threshold
bbox                 | bool   | yes      | false                   | returns bounding boxes around object when using an object detection model, such that (xmin,ymax) yields the top left corner and (xmax,ymin) the lower right corner of a box.
best_bbox            | int    | yes      | -1                      | if > 0, returns only the `best_bbox` with highest confidence
nms_threshold        | float  | yes      | 0.45                    | non-maximum suppression threshold for detection outputs
top_k                | int    | yes      | -1                      | limit internal top-k detection candidates when supported by the backend
regression           | bool   | yes      | false   | whether the output of a model is a regression target (i.e. vector of one or more floats)
rois                 | string | yes      | empty                   | set the ROI layer from which to extract features from bounding boxes. Both boxes and features are returned when using an object detection model with an ROI pooling layer
index                | bool   | yes      | false                   | whether to index the output from prediction, for similarity search
build_index          | bool   | yes      | false                   | whether to build similarity index after prediction, no more indexing can be done afterward
search               | bool   | yes      | false                   | whether to use the predicted output for similarity search and return pre-indexed nearest neighbors
search_nn            | int    | yes      | 10                      | number of similarity search results
multibox_rois        | bool   | yes      | false                   | aggregates bounding boxes ROIs features (requires `rois`) for image similarity search
index_type           | string | yes      | Flat                    | for faiss index indexing backend only : a FAISS index factory string , see https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
index_gpu            | bool   | yes      | false                   | for faiss indexing backend only: if available, build index on GPU
index_gpuid          | int    | yes      | all                     | for faiss indexing backend only : which gpu to use if index_gpu is true
train_samples        | int    | yes      | 100000                  | for faiss indexing backend only :  number of samples to use for training index. Larger values lead to better indexes (more evenly distributed) but cause much larger index training time. Many indexes need a minimal value depending on the number of clusters built,  see https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index.
ondisk               | bool   | yes      | true                    | for faiss indexing backend only :  try to directly build indexes on mmaped files (IVF index_types only can do so)
nprobe               | int    | yes      | max(ninvertedlist/50,2) | for faiss indexing backend only : number of cluster searched for closest images: for highly compressing indexes, setting nprobe to larger values may allow better precision
ctc                  | bool   | yes      | false                   | whether the output is a sequence (using CTC encoding)
confidences          | array  | yes      | empty                   | Segmentation only: output confidence maps for "best" class, "all" classes, or classes being specified by number, e.g. "1","3".
logits_blob          | string | yes      | ""                      | in classification services, this adds raw logits to output. Useful for calibration purposes
logits               | bool   | yes      | false                   | in detection services, this adds logits to output. Useful for calibration purposes

- Network object

Parameter    | Type   | Optional | Default                        | Description
---------    | ----   | -------- | -------                        | -----------
url          | string | no       | N/A                            | URL of the remote service to connect to (e.g http://localhost:9200)
http_method  | string | yes      | POST                           | HTTP connecting method, from "POST", "PUT", etc...
content_type | string | yes      | Content-Type: application/json | Content type HTTP header string

The variables that are usable in the output template format are those from the standard JSON output. See the [output template](#templates) dedicated section for more details and examples.

#### Machine learning libraries

- Torch

Parameter     | Type   | Optional | Default | Description
---------     | ----   | -------- |---------| -----------
gpu           | bool   | yes      | false   | Whether to use GPU
gpuid         | int or array | yes | 0       | GPU id, use single int for single GPU, `-1` for using all GPUs, and array e.g. `[1,3]` for selecting among multiple GPUs
extract_layer | string | yes      | ""      | Returns tensor values from intermediate layers. In bert models "hidden_state" allows to extract raw hidden_states values to return as output. If set to 'last', simply returns the tensor values from last layer.
forward_method | string | yes | ""      | Executes a custom function from within a traced/JIT model, instead of the standard forward()
multi_label | bool | yes | false   | Model outputs an independent score for each class
concurrent_predict | bool | yes | true    | Enable/disable concurrent predict for the model
datatype | string | yes | fp32 | Prediction datatype: `fp32`, `fp16`, or `fp64`

- TensorRT

Parameter     | Type   | Optional | Default | Description
---------     | ----   | -------- |---------| -----------
gpuid         | int    | yes      | 0       | GPU id. TensorRT accepts a single GPU id for inference.
calibration   | bool   | yes      | false   | Use this prediction request as INT8 calibration data.

- XGBoost (optional legacy backend)

No parameter required.

- NCNN (optional legacy backend)

Parameter  | Type   | Optional | Default                                                                 | Description
---------  | ----   | -------- | -------                                                                 | -----------
inputblob  | string | yes      | data                                                                    | network input blob name
outputblob | string | yes      | depends on network type (ie prob or rnn_pred or probs or detection_out) | network output blob name

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
* `csvts` instantiates the input connector for CSV time series
* `txt` instantiates the input connector for text files
* `svm` instantiates the sparse/SVM-style input connector
* `video` instantiates the video input connector for Torch video services

Input connectors work almost the same during both the training and prediction phases. But the training phase usually deals with large masses of data, and therefore the connectors above are optimized to automate some tasks, typically building and preprocessing the dataset at training time.

Below is a summary of input connectors options, though they are all already defined in each API resource and call documentation.

- Image (`image`)

Parameter  | Type | Optional | Default | Description
---------  | ---- | -------- | ------- | -----------
width      | int  | yes      | 227     | Resize images to width ("image" only)
height     | int  | yes      | 227     | Resize images to height ("image" only)
bw         | bool | yes      | false   | Treat images as black & white
rgb        | bool | yes      | false   | Use RGB images
scale      | float | yes     | 1.0     | Multiply each pixel by a factor
mean       | array | yes     | empty   | Mean value per channel to subtract
std        | array | yes     | empty   | Standard deviation per channel
segmentation | bool | yes    | false   | Configure an image connector for segmentation
bbox       | bool | yes      | false   | Configure an image connector for object detection
unchanged_data | bool | yes  | false   | Do not modify input data during resize/preprocessing where supported
interp     | string | yes    | cubic   | Image interpolation method
aspect_ratio_pad | bool | yes | false  | Pad image to a square when resizing with aspect ratio
test_split | real | yes      | 0       | Test split part of the dataset
shuffle    | bool | yes      | false   | Whether to shuffle the training set (prior to splitting)
seed       | int  | yes      | -1      | Shuffling seed for reproducible results (-1 for random seeding)

- CSV (`csv`)

Parameter            | Type            | Optional | Default | Description
---------            | ----            | -------- | ------- | -----------
label                | string          | no       | N/A     | Label column name
ignore               | array of string | yes      | empty   | Array of column names to ignore
label_offset         | int             | yes      | 0       | Negative offset (e.g. -1) so that labels range from 0 onward
separator            | string          | yes      | ','     | Column separator character
quote	             | string	       | yes	  | '"'	    | Quote character in CSV file
id                   | string          | yes      | empty   | Column name of the training examples identifier field, if any
scale                | bool            | yes      | false   | Whether to scale all values into [0,1]
min_vals,max_vals    | array           | yes      | empty   | Instead of `scale`, provide the scaling parameters, as returned from a training call
categoricals         | array           | yes      | empty   | List of categorical variables
categoricals_mapping | object          | yes      | empty   | Categorical mappings, as returned from a training call
test_split           | real            | yes      | 0       | Test split part of the dataset
shuffle              | bool            | yes      | false   | Whether to shuffle the training set (prior to splitting)
seed                 | int             | yes      | -1      | Shuffling seed for reproducible results (-1 for random seeding)
db                   | bool            | yes      | false   | whether to gather data into a database, useful for very large datasets, allows training in constant-size memory

- Text (`txt`)

Parameter       | Type   | Optional | Default                                            | Description
---------       | ----   | -------- | -------                                            | -----------
count           | int    | yes      | true                                               | whether to count words and report counters
min_count       | int    | yes      | 5                                                  | min word count occurrences for a word to be taken into account
min_word_length | int    | yes      | 5                                                  | min word length for a word to be taken into account
tfidf           | bool   | yes      | false                                              | whether to compute TF/IDF for every word
sentences       | bool   | yes      | false                                              | whether to turn every line into a document (requires dataset as file with one sentence per line in every class repository)
characters      | bool   | yes      | false                                              | character-level text processing, as opposed to word-based text processing
sequence        | int    | yes      | N/A                                                | for character-level text processing, the fixed length of each sample of text
read_forward    | bool   | yes      | false                                              | for character-level text processing, whether to read content from left to right
alphabet        | string | yes      | abcdefghijklmnopqrstuvwxyz 0123456789 ,;.!?:'"/\\\ \|_@#$%^&*~\`+-=<>()[]{} | for character-level text processing, the alphabet of recognized symbols
test_split      | real   | yes      | 0                                                  | Test split part of the dataset
shuffle         | bool   | yes      | false                                              | Whether to shuffle the training set (prior to splitting)
seed            | int    | yes      | -1                                                 | Shuffling seed for reproducible results (-1 for random seeding)
db              | bool   | yes      | false                                              | whether to gather data into a database, useful for very large datasets, allows training in constant-size memory
sparse          | bool   | yes      | false                                              | whether to use sparse text features

- SVM (`svm`)

No parameters

## Output connectors

The output connector controls the output formats for supervised and unsupervised models.

Its two main features are the control of the number of predictions per URI, and the output templating, which allows for custom output and seamless integration in external applications. Other options modulates the output format.

Parameter            | Type   | Optional | Default | Description
---------            | ----   | -------- | ------- | -----------
best                 | int    | yes      | 1       | Number of top predictions returned by data URI (supervised)
measure              | array  | yes      | empty   | Output measures requested, from `acc`: accuracy, `acc-k`: top-k accuracy, replace k with number (e.g. `acc-5`), `f1`: f1, precision and recall, `mcll`: multi-class log loss, `auc`: area under the curve, `cmdiag`: diagonal of confusion matrix (requires `f1`), `cmfull`: full confusion matrix (requires `f1`), `mcc`: Matthews correlation coefficient
template             | string | yes      | empty   | Output template in Mustache format
confidence_threshold | double | yes      | 0.0     | only returns classifications or detections with probability strictly above threshold
bbox                 | bool   | yes      | false   | returns bounding boxes around object when using an object detection model
best_bbox            | int    | yes      | -1      | if > 0, returns only the `best_bbox` detections with highest confidence
nms_threshold        | float  | yes      | 0.45    | non-maximum suppression threshold for detection outputs
top_k                | int    | yes      | -1      | limit internal top-k detection candidates when supported by the backend
regression           | bool   | yes      | false   | whether the output of a model is a regression target (i.e. vector of one or more floats)
rois                 | string | yes      | empty                   | set the ROI layer from which to extract features from bounding boxes. Both boxes and features are returned when using an object detection model with an ROI pooling layer
index                | bool   | yes      | false                   | whether to index the output from prediction, for similarity search
build_index          | bool   | yes      | false                   | whether to build similarity index after prediction, no more indexing can be done afterward
search               | bool   | yes      | false                   | whether to use the predicted output for similarity search and return pre-indexed nearest neighbors
search_nn            | int    | yes      | 10                      | number of similarity search results
multibox_rois        | bool   | yes      | false                   | aggregates bounding boxes ROIs features (requires `rois`) for image similarity search
index_type           | string | yes      | Flat                    | for faiss index indexing backend only : a FAISS index factory string , see https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
index_gpu            | bool   | yes      | false                   | for faiss indexing backend only: if available, build index on GPU
index_gpuid          | int    | yes      | all                     | for faiss indexing backend only : which gpu to use if index_gpu is true
train_samples        | int    | yes      | 100000                  | for faiss indexing backend only :  number of samples to use for training index. Larger values lead to better indexes (more evenly distributed) but cause much larger index training time. Many indexes need a minimal value depending on the number of clusters built,  see https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index.
ondisk               | bool   | yes      | true                    | for faiss indexing backend only :  try to directly build indexes on mmaped files (IVF index_types only can do so)
nprobe               | int    | yes      | max(ninvertedlist/50,2) | for faiss indexing backend only : number of cluster searched for closest images: for highly compressing indexes, setting nprobe to larger values may allow better precision
ctc                  | bool   | yes      | false                   | whether the output is a sequence (using CTC encoding)
confidences          | array  | yes      | empty                   | Segmentation only: output confidence maps for "best" class, "all" classes, or classes being specified by number, e.g. "1","3".
logits_blob          | string | yes      | ""                      | in classification services, this adds raw logits to output. Useful for calibration purposes
logits               | bool   | yes      | false                   | in detection services, this adds logits to output. Useful for calibration purposes


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

> Example of a ResNet-18 classifier with two classes:

```json
{"parameters":{"input":{"connector":"image","width":224,"height":224,"rgb":true,"scale":0.0039},"mllib":{"template":"resnet18","nclasses":2}}}
```

> Example of a YOLOX object detection service:

```json
{"parameters":{"input":{"connector":"image","width":640,"height":640,"rgb":true,"bbox":true},"mllib":{"template":"yolox","nclasses":80}}}
```

The DeepDetect server and API come with a set of Machine Learning model templates.

Templates are available for the Torch backend. Some templates instantiate
native modules directly, while others expect a traced/JIT model in the model
repository.

## Neural network templates

All models below are used by passing their id to the `mllib/template` parameter in `PUT /services` calls:

### Torch

#### Native models

- LSTM-like models (including autoencoder): `recurrent`
- NBEATS model: `nbeats`
- Vision transformer: `vit` and `visformer`
- Transformer-based timeseries models: `ttransformer`
- CRNN OCR model: `crnn`
- TorchVision-style image classification models compiled into the native factory:
	- `resnet18`
	- `resnet34`
	- `resnet50`
	- `resnet101`
	- `resnet152`
	- `resnext50_32x4d`
	- `resnext101_32x8d`
	- `wideresnet50_2`
	- `wideresnet101_2`

#### Traced models

These templates require an external traced model to work:

- Language models:
	- `bert`
	- `gpt2`
- Detection models:
	- `fasterrcnn`
	- `retinanet`
	- `detr`
	- `yolox`
- Segmentation models:
	- DeepLabV3 and SegFormer-style services use `segmentation:true` with a compatible repository.

## Parameters

Model instantiation parameters for recurrent template (applies to all backends supporting templates):

Parameter       | Template  | Type            | Default                      | Description
---------       | --------- | ------          | ---------------------------- | -----------
layers          | recurrent | array of string | []                           | ["L50","L50"] means 2 layers of LSTMs with hidden size of 50. ["L100","L100", "T", "L300"] means an lstm autoencoder with encoder composed of 2 LSTM layers of hidden size 100 and decoder is one LSTM layer of hidden size 300

### Torch

Template parameters for native  templates (nbeats/ttransformer):

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

Repository examples include:

* Python wheel and CLI examples in [examples/python](../examples/python)
* Image classification demo in [demo/imgdetect](../demo/imgdetect)
* Object detection demo in [demo/objdetect](../demo/objdetect)
* Image similarity search demo in [demo/imgsearch](../demo/imgsearch)
* Segmentation demo in [demo/segmentation](../demo/segmentation)
