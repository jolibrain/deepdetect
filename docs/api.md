---
title: API Reference
layout: api
aliases: [/api/]
---

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

> Create a service from a multilayer Neural Network template, taking input from a CSV for prediction over 9 classes with 3 layers.

``` shell
curl -X PUT "http://localhost:8080/services/myserv" -d "{\"mllib\":\"caffe\",\"description\":\"example classification service\",\"type\":\"supervised\",\"parameters\":{\"input\":{\"connector\":\"csv\"},\"mllib\":{\"template\":\"mlp\",\"nclasses\":9,\"layers\":[512,512,512],\"activation\":\"prelu\"}},\"model\":{\"repository\":\"/home/me/models/example\"}}"

# If "/home/me/models/example" correctly exists, the output is

{"status":{"code":201,"msg":"Created"}}
```

``` shell
curl -X PUT "http://localhost:8080/services/myserv" -d "{\"mllib\":\"xgboost\",\"description\":\"example classification service\",\"type\":\"supervised\",\"parameters\":{\"input\":{\"connector\":\"csv\"},\"mllib\":{\"nclasses\":9}},\"model\":{\"repository\":\"/home/me/models/example\"}}"
```

```python

from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

description = 'example classification service'

layers = [512,512,512]
mllib = 'caffe'
model = {'templates':'../templates/caffe/','repository':'home/me/models/example'}
parameters_input = {'connector':'csv'}
parameters_mllib = {'template':'mlp','nclasses':9,'layers':layers,'activation':'prelu'}
parameters_output = {}
dd.put_service('myserv',model,description,mllib,
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
mllib       | string | No       | N/A          | Name of the Machine Learning library, from `caffe`, `caffe2`, `xgboost`, `tsne` and `tensorflow`
type        | string | No       | `supervised` | Machine Learning service type: `supervised` yields a series of metrics related to a supervised objective, or `unsupervised`, typically for state-space compression or accessing neural network's inner layers.
description | string | yes      | empty        | Service description
model       | object | No       | N/A          | Information for the statistical model to be built and/or used by the service
input       | object | No       | N/A          | Input information for connecting to data
output      | object | yes      | empty        | Output information

- Model Object

Parameter         | Type   | Optional | Default   | Description
---------         | ----   | -------- | -------   | -----------
repository        | string | No       | N/A       | Repository for the statistical model files
templates         | string | yes      | templates | Repository for model templates
weights           | string | yes      | empty     | Weights filename of a pre-trained network (e.g. for finetuning or resuming a net)
create_repository | bool   | yes      | false     | Whether to create the model repository directory if it does not exist already
index_preload     | bool   | yes      | true      | Whether to preload a similarity search index, set to false for fast init

#### Connectors

- Input Object

Parameter | Type   | Optional | Default | Description
--------- | ----   | -------- | ------- | -----------
connector | string | No       | N/A     | Either "image" or "csv", defines the input data format
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
segmentation   | bool         | yes      | false   | whether to setup an image connector for a segmentation task (`caffe` only)
multi_label    | bool         | yes      | false   | whether to setup a multi label image task (`caffe` only)
root_folder    | string       | yes      | false   | root folder for image data layer (i.e multi label image service for training with `caffe`)
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
id           | string          | yes      | empty   | Column name of the training examples identifier field, if any
scale        | bool            | yes      | false   | Whether to scale all values into [0,1]
categoricals | array           | yes      | empty   | List of categorical variables
db           | bool            | yes      | false   | whether to gather data into a database, useful for very large datasets, allows treatment in constant-size memory

CSV Time-series (`csvts`)

Parameter | Type            | Optional | Default | Description
--------- | ----            | -------- | ------- | -----------
label     | string          | no       | N/A     | Label column name
ignore    | array of string | yes      | empty   | Array of column names to ignore
separator | string          | yes      | ','     | Column separator character
id        | string          | yes      | empty   | Column name of the training examples identifier field, if any
scale     | bool            | yes      | false   | Whether to scale all values into [0,1]
db        | bool            | yes      | false   | whether to gather data into a database, useful for very large datasets, allows treatment in constant-size memory


Text (`txt`)

Parameter          | Type   | Optional | Default                                            | Description
---------          | ----   | -------- | -------                                            | -----------
sentences          | bool   | yes      | false                                              | whether to turn every line into a document (requires dataset as file with one sentence per line in every class repository)
characters         | bool   | yes      | false                                              | character-level text processing, as opposed to word-based text processing
sequence           | int    | yes      | N/A                                                | for character-level text processing, the fixed length of each sample of text
read_forward       | bool   | yes      | false                                              | for character-level text processing, whether to read content from left to right
alphabet           | string | yes      | abcdefghijklmnopqrstuvwxyz 0123456789 ,;.!?:'"/\\\ | _@#$%^&*~`+-=<>()[]{} | for character-level text processing, the alphabet of recognized symbols
sparse             | bool   | yes      | false                                              | whether to use sparse features (and sparce computations with Caffe for huge memory savings, for xgboost use `svm` connector instead)
ordered_words      | bool   | yes      | false                                              | enable word-based processing with positionnal information, mandatory for bert/gpt2 like models
wordpiece_tokens   | bool   | yes      | false                                              | set to true if vocabulary contains partial words, ie like in bert/gpt2 models
punctuation_tokens | bool   | yes      | false                                              | if true, treat each punctuation sign as a token; if false, punctuation is stripped from input
word_start         | string | yes      | ""                                                 | in most gpt2 vocabularies, start of word has generally to be set to "Ġ".
suffix_start       | string | yes      | "##"                                               | in bert-like vocabularies, suffixes are prefixed by `##`

SVM (`svm`)

No parameters

See the section on [Connectors](#connectors) for more details.

#### Machine learning libraries

- Caffe

Parameter            | Type            | Optional                 | Default   | Description
---------            | ----            | --------                 | -------   | -----------
nclasses             | int             | no (classification only) | N/A       | Number of output classes (`supervised` service type)
ntargets             | int             | no (regression only)     | N/A       | Number of regression targets
gpu                  | bool            | yes                      | false     | Whether to use GPU
gpuid                | int or array    | yes                      | 0         | GPU id, use single int for single GPU, `-1` for using all GPUs, and array e.g. `[1,3]` for selecting among multiple GPUs
template             | string          | yes                      | empty     | Neural network template, from `lregression`, `mlp`, `convnet`, `alexnet`, `googlenet`, `nin`, `resnet_18`, `resnet_32`, `resnet_50`, `resnet_101`, `resnet_152`
engine               | string          | yes                      | `DEFAULT` | CUDNN engine type : orginal CUDNN implementation may leak a lot of memory when loading unloading multiple models, especially when using group convolutions. `CUDNN` is caffe orginal implementation, it tries to parralize with many handles/stream, buts these handles are never fully released internally in cudnn, even after an explicit "release" call, hence the memory leak. `CUDNN_SINGLE_HANDLE` use only one handle/stream, which is much less leak prone. `CUDNN_MIN_MEMORY` explictly forces cudnn algorithm that do not require any external buffer allocation. `CAFFE` forces CPU implementation, and `DEFAULT` fallbacsk to CAFFE if CUDNN support was not enabled at compile time, or CUDNN_SINGLE_HANDLE if CUDNN support was compiled in.
layers               | array of int    | yes                      | [50]      | Number of neurons per layer (`mlp` only)
layers               | array of string | yes                      | [1000]    | Type of layer and number of neurons peer layer: XCRY for X successive convolutional layers of Y filters and activation layers followed by a max pooling layer, an int as a string for specifying the final fully connected layers size, e.g. \["2CR32","2CR64","1000"\] (`convnet` only)
activation           | string          | yes                      | relu      | Unit activation (`mlp` and `convnet` only), from `sigmoid`,`tanh`,`relu`,`prelu`,`elu`
dropout              | real or array   | yes                      | 0.5       | Dropout rate between layers (templates, `mlp` and `convnet` only)
regression           | bool            | yes                      | false     | Whether the network is a regressor (templates, `mlp` and `convnet` only)
autoencoder          | bool            | yes                      | false     | Whether the network is an autoencoder (template `mlp` only)
crop_size            | int             | yes                      | N/A       | Size of random image crops as input to the net (templates and `convnet` only)
rotate               | bool            | yes                      | false     | Whether to apply random rotations to input images (templates and `convnet` only)
mirror               | bool            | yes                      | false     | Whether to apply random mirroring of input images (templates and `convnet` only)
finetuning           | bool            | yes                      | false     | Whether to prepare neural net template for finetuning (requires `weights`)
db                   | bool            | yes                      | false     | whether to set a database as input of neural net, useful for handling large datasets and training in constant-memory (requires `mlp` or `convnet`)
scaling_temperature  | real            | yes                      | 1.0       | sets the softmax temperature of an existing network (e.g. useful for model calibration)
loss                 | string          | yes                      | N/A       | Special network losses, from `dice`, `dice_multiclass`, `dice_weighted`, `dice_weighted_batch` or `dice_weighted_all`, useful for image segmentation, and `L1` or `L2`, useful for time-series via `csvts` connector
ssd_expand_prob      | float           | yes                      | between 0 and 1, probability of expanding the image (to improve detection of small/very small objects)
ssd_max_expand_ratio | float           | yes                      | bbox zoom out ratio, e.g. 4.0
ssd_mining_type      | str             | yes                      | N/A       | "HARD_EXAMPLE" or "MAX_NEGATIVE"
ssd_neg_pos_ratio    | float           | yes                      | N/A       | ratio of negative sampled examples wrt positive examples (bbox), e.g. 3.0
ssd_neg_overlap      | float           | yes                      | N/A       | max overlap of negative samples with positive samples (bbox), between 0 and 1, e.g. 0.5
ssd_keep_top_k       | float           | yes                      | N/A       | keep k examples after nms has finished

See the [Model Templates](#model_templates) section for more details.

Noise (images only):

Parameter      | Type   | Optional | Default | Description
---------      | ----   | -------- | ------- | -----------
prob           | double | yes      | 0.0     | Probability of each effect occurence
all_effects    | bool   | yes      | false   | Apply all effects below, randomly
decolorize     | bool   | yes      | N/A     | Whether to decolorize image
hist_eq        | bool   | yes      | N/A     | Whether to equalize histogram
inverse        | bool   | yes      | N/A     | Whether to inverse image
gauss_blur     | bool   | yes      | N/A     | Whether to apply Gaussian blur
posterize      | bool   | yes      | N/A     | Whether to posterize image
erode          | bool   | yes      | N/A     | Whether to erode image
saltpepper     | bool   | yes      | N/A     | Whether to apply salt & pepper effect to image
clahe          | bool   | yes      | N/A     | Whether to apply CLAHE
convert_to_hsv | bool   | yes      | N/A     | Whether to convert to HSV
convert_to_lab | bool   | yes      | N/A     | Whether to convert to LAB

Distort (images only):

Parameter       | Type   | Optional | Default | Description
---------       | ----   | -------- | ------- | -----------
prob            | double | yes      | 0.0     | Probability of each effect occurence
all_effects     | bool   | yes      | false   | Apply all effects below, randomly
brightness      | bool   | yes      | N/A     | Whether to distort image brightness
contrast        | bool   | yes      | N/A     | Whether to distort image contrast
saturation      | bool   | yes      | N/A     | Whether to distort image saturation
HUE             | bool   | yes      | N/A     | Whether to distort image HUE
random ordering | bool   | yes      | N/A     | Whether to randomly reorder the image channels

Geometry (images only):

Parameter        | Type   | Optional | Default  | Description
---------        | ----   | -------- | -------  | -----------
prob             | double | yes      | 0.0      | Probability of each effect occurence
all_effects      | bool   | yes      | false    | Apply all effects below, randomly
persp_horizontal | bool   | yes      | true     | Whether to distort the perspective horizontally
persp_vertical   | bool   | yes      | true     | Whether to distort the perspective vertically
zoom_out         | bool   | yes      | true     | distance change, look further away
zoom_in          | bool   | yes      | true     | distance changee, look from closer by
zoom_factor      | float  | yes      | 0.25     | 0.25 means that image can be *1.25 or /1.25
persp_factor     | float  | yes      | 0.25     | 0.25 means that new image corners  be in *1.25 or 0.75
pad_mode         | string | yes      | mirrored | filling around image, from `mirrored` / `constant` (black) / `repeat_nearest`

- Caffe2

Parameter  | Type         | Optional                 | Default | Description
---------  | ----         | --------                 | ------- | -----------
nclasses   | int          | no (classification only) | N/A     | Number of output classes (`supervised` service type)
gpu        | bool         | yes                      | false   | Whether to use GPU
gpuid      | int or array | yes                      | 0       | GPU id, use single int for single GPU, `-1` for using all GPUs, and array e.g. `[1,3]` for selecting among multiple GPUs
template   | string       | yes                      | empty   | Neural network template (from `lenet`, `alexnet`, `resnet_50`)
mirror     | bool         | yes                      | false   | Whether to apply random mirroring of input images (templates and `convnet` only)
finetuning | bool         | yes                      | false   | Whether to prepare neural net template for finetuning (requires `weights`)

- XGBoost

Parameter  | Type | Optional                 | Default | Description
---------  | ---- | --------                 | ------- | -----------
nclasses   | int  | no (classification only) | N/A     | Number of output classes (`supervised` service type)
ntargets   | int  | no (regression only)     | N/A     | Number of regression targets (only 1 supported by XGBoost)
regression | bool | yes                      | false   | Whether to train a regressor

- Tensorflow

Parameter   | Type   | Optional                 | Default | Description
---------   | ----   | --------                 | ------- | -----------
nclasses    | int    | no (classification only) | N/A     | Number of output classes (`supervised` service type)
inputlayer  | string | yes                      | auto    | network input layer name
outputlayer | string | yes                      | auto    | network output layer name

- NCNN

Parameter  | Type   | Optional | Default                                                                 | Description
---------  | ----   | -------- | -------                                                                 | -----------
inputblob  | string | yes      | data                                                                    | network input blob name
outputblob | string | yes      | depends on network type (ie prob or rnn_pred or probs or detection_out) | network output blob name

- TensorRT

Parameter          | Type   | Optional | Default     | Description
---------          | ----   | -------- | -------     | -----------
tensorRTEngineFile | string | yes      | "TRTengine" | prefix of filename of TRT compiled model (complete name defaults to   "TRTengine_bs48")
readEngine         | bool   | yes      | true        | if a compiled model file with corresponding prefix, whatever batch size exists in repo, use it instead of recompiling a TRT model
writeEngine        | bool   | yes      | true        | if a new TRT model was compiled, write it to disk
maxWorkspaceSize   | int    | yes      | 1024        | max memory (in MB) usable by TRT during model compilation. Usefull mainly on nano : 1024 is 1GB and may cause dede to be sigkilled if not enough memory. 256 allows to limit memory consumption and create low batchsize nets.
maxBatchSize       | int    | yes      | 48          | maximum batch size processable by TRT compiled model.  If a precompiled engine starting with tensorRTEngineFile name is present and readEngine is set to true , this previous engine is used, overriding this option.
dla                | int    | true     | -1          | id of DLA to use, if available on your hardware
datatype           | string | true     | "fp32"      | datatype inside compiled TRT model (available : "fp32", "fp16" (also known as half), "int8". "int8" is strongly discouraged at the moment as it has not been tested and needs a special procedure to calibrate quantization based on precise final task and representative data.

- Output Object

Parameter    | Type | Optional | Default | Description
---------    | ---- | -------- | ------- | -----------
store_config | bool | yes      | false   | stores the creation call in a `config.json` file in the model directory
measure      | array of string | yes | depending on problem type | measure to use at test time


Problem type | Default | Possible values | Description
------------ | ------- | --------------- |
timeserie    |   L1    | L1, L2, mase, mape, smape, mase, owa | L1: mean error, L2: mean squared error, mase : mean absolute scaled error, mape: mean absolute percentage error, smape: symetric mean absolute percentage error, owa: overall weighted average.



## Get information on a service

```shell
curl -X GET "http://localhost:8080/services/myserv"

> Assuming the service 'myserv' was previously created, yields

{
  "status":{
	     "code":200,
	     "msg":"OK"
	  },
  "body":{
	     "mllib":"caffe",
	     "description":"example classification service",
	     "name":"myserv",
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

dd.get_service('myserv')

> returns:

{u'status': {u'msg': u'OK', u'code': 200}, u'body': {u'jobs': {}, u'mllib': u'caffe', u'name': u'myserv', u'description': u'example classification service'}}
```

Returns information on an existing service

### HTTP Request

`GET /services/myserv`

### Query Parameters

None

## Delete a service

```shell
curl -X DELETE "http://localhost:8080/services/myserv?clear=full"

> Yields

{"status":{"code":200,"msg":"OK"}}

```

```python

from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

dd.delete_service('myserv',clear='full')

```

### HTTP Request

`DELETE /services/myserv`

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
⚠️ The current integration of the Caffe back-end for deep learning does not allow making predictions while training. However, two different services can train and predict at the same time.
</div>

## Launch a training job

> Blocking train call from CSV dataset

```shell
curl -X POST "http://127.0.0.1:8080/train" -d "{\"service\":\"myserv\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":300,\"test_interval\":100},\"net\":{\"batch_size\":5000}},\"input\":{\"label\":\"target\",\"id\":\"id\",\"separator\":\",\",\"shuffle\":true,\"test_split\":0.15,\"scale\":true},\"output\":{\"measure\":[\"acc\",\"mcll\"]}},\"data\":[\"/home/me/example/train.csv\"]}"

{"status":{"code":201,"msg":"Created"},"body":{"measure":{"iteration":299.0,"train_loss":0.6463099718093872,"mcll":0.5919793284503224,"acc":0.7675070028011205}},"head":{"method":"/train","time":403.0}}

```

```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

parameters_input = {'label':'target','id':'id','separator':',','shuffle':True,'test_split':0.5,'scale':True}
parameters_mllib = {'gpu':True,'solver':{'iterations':300,'test_iterval':100},'net':{'batch_size':5000}}
parameters_output = {'measure':['acc','mcll']}
train_data = ['/home/me/example/train.csv/']

dd.post_train('myserv',train_data,parameters_input,parameters_mllib,parameters_output,async=False)
```

> Asynchronous train call from CSV dataset


```shell
curl -X POST "http://127.0.0.1:8080/train" -d "{\"service\":\"myserv\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":100000,\"test_interval\":1000},\"net\":{\"batch_size\":512}},\"input\":{\"label\":\"target\",\"id\":\"id\",\"separator\":\",\",\"shuffle\":true,\"test_split\":0.15,\"scale\":true},\"output\":{\"measure\":[\"acc\",\"mcll\"]}},\"data\":[\"/home/me/models/example/train.csv\"]}"

{"status":{"code":201,"msg":"Created"},"head":{"method":"/train","job":1,"status":"running"}}
```

```shell
curl -X POST "http://127.0.0.1:8080/train" -d "{\"service\":\"myserv\",\"async\":true,\"parameters\":{\"mllib\":{\"objective\":\"multi:softprob\",\"booster_params\":{\"max_depth\":10}},\"input\":{\"label\":\"target\",\"id\":\"id\",\"separator\":\",\",\"shuffle\":true,\"test_split\":0.15,\"scale\":true},\"output\":{\"measure\":[\"acc\",\"mcll\"]}},\"data\":[\"/home/me/models/example/train.csv\"]}"

{"status":{"code":201,"msg":"Created"},"head":{"method":"/train","job":1,"status":"running"}}
```

```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

parameters_input = {'label':'target','id':'id','separator':',','shuffle':True,'test_split':0.5,'scale':True}
parameters_mllib = {'gpu':True,'solver':{'iterations':300,'test_iterval':100},'net':{'batch_size':5000}}
parameters_output = {'measure':['acc','mcll']}
train_data = ['/home/me/example/train.csv/']

dd.post_train('myserv',train_data,parameters_input,parameters_mllib,parameters_output,async=True)

```

> Requesting the status of an asynchronous training job:

```shell
curl -X GET "http://localhost:8080/train?service=myserv&job=1"

{"status":{"code":200,"msg":"OK"},"head":{"method":"/train","job":1,"status":"running","time":74.0},"body":{"measure":{"iteration":445.0,"train_loss":0.7159726023674011,"mcll":2.1306082640485237,"acc":0.16127989657401424}}}
```
```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

dd.get_train('myserv',job=1)
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
bw           | bool | yes      | false   | Treat images as black & white (Caffe only)
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
label_offset         | int             | yes      | 0       | Negative offset (e.g. -1) s othat labels range from 0 onward
separator            | string          | yes      | ','     | Column separator character
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
label_offset      | int             | yes      | 0       | Negative offset (e.g. -1) s othat labels range from 0 onward
separator         | string          | yes      | ','     | Column separator character
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
min_count          | int    | yes      | 5                                                  | min word count occurences for a word to be taken into account
min_word_length    | int    | yes      | 5                                                  | min word length for a word to be taken into account
tfidf              | bool   | yes      | false                                              | whether to compute TF/IDF for every word
sentences          | bool   | yes      | false                                              | whether to turn every line into a document (requires dataset as file with one sentence per line in every class repository)
characters         | bool   | yes      | false                                              | character-level text processing, as opposed to word-based text processing
sequence           | int    | yes      | N/A                                                | for character-level text processing, the fixed length of each sample of text
read_forward       | bool   | yes      | false                                              | for character-level text processing, whether to read content from left to right
alphabet           | string | yes      | abcdefghijklmnopqrstuvwxyz 0123456789 ,;.!?:'"/\\\ | _@#$%^&*~`+-=<>()[]{} | for character-level text processing, the alphabet of recognized symbols
test_split         | real   | yes      | 0                                                  | Test split part of the dataset
shuffle            | bool   | yes      | false                                              | Whether to shuffle the training set (prior to splitting)
seed               | int    | yes      | -1                                                 | Shuffling seed for reproducible results (-1 for random seeding)
db                 | bool   | yes      | false                                              | whether to gather data into a database, useful for very large datasets, allows training in constant-size memory
sparse             | bool   | yes      | false                                              | whether to use sparse features (and sparce computations with Caffe for huge memory savings, for xgboost use `svm` connector instead)
embedding          | bool   | yes      | false                                              | whether to use an embedding as input to the network (replaces one-hot vectors with straight indices)
ordered_words      | bool   | yes      | false                                              | enable word-based processing with positionnal information, mandatory for bert/gpt2 like models
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
measure           | array  | yes      | empty   | Output measures requested, from `acc`: accuracy, `acc-k`: top-k accuracy, replace k with number (e.g. `acc-5`), `f1`: f1, precision and recall, `mcll`: multi-class log loss, `auc`: area under the curve, `cmdiag`: diagonal of confusion matrix (requires `f1`), `cmfull`: full confusion matrix (requires `f1`), `mcc`: Matthews correlation coefficient, `eucll`: euclidean distance (e.g. for regression tasks), `kl`: KL_divergence, `js`: JS divergence, `was`: Wasserstein, `ks`: Kolmogorov Smirnov, `dc`: distance correlation, `r2`: R2, `deltas`: delta scores, 'raw': ouput raw results, in case of predict call, this requires a special deploy.prototxt that is a test network (to have ground truth)
target_repository | string | yes      | empty   | target directory to which to copy the best model files once training has completed

#### Machine learning libraries

- Caffe

General:

Parameter     | Type           | Optional | Default        | Description
---------     | ----           | -------- | -------        | -----------
gpu           | bool           | yes      | false          | Whether to use GPU
gpuid         | int or array   | yes      | 0              | GPU id, use single int for single GPU, `-1` for using all GPUs, and array e.g. `[1,3]` for selecting among multiple GPUs
resume        | bool           | yes      | false          | Whether to resume training from .solverstate and .caffemodel files
class_weights | array of float | yes      | 1.0 everywhere | Whether to weight some classes more / less than others, e.g. [1.0,0.1,1.0]
ignore_label  | int            | yes      | N/A            | A single label to be ignored by the loss (i.e. no gradients)
timesteps     | int            | yes      | N/A            | Number of timesteps for recurrence ('csvts', `ctc` OCR) models (in case of csvts, used only at train time)
offset        | int            | yes      | N/A            | Offset beween start point of sequences with connector `cvsts`, defining the overlap of input series. For [0, n] steps a timestep of t and an offset of k, series [0..t-1], [k..t+k-1], [2k, 2k+t-1] ... will be cosntructed. If some elements at the end could not be taken using this, it will add a final [n-t+1..n] sequence (used only at train time).

Solver:

Parameter            | Type         | Optional | Default | Description
---------            | ----         | -------- | ------- | -----------
iterations           | int          | yes      | N/A     | Max number of solver's iterations
snapshot             | int          | yes      | N/A     | Iterations between model snapshots
snapshot_prefix      | string       | yes      | empty   | Prefix to snapshot file, supports repository
solver_type          | string       | yes      | SGD     | from "SGD", "ADAGRAD", "NESTEROV", "RMSPROP", "ADADELTA", "ADAM",  "AMSGRAD", "RANGER", "RANGER_PLUS", "ADAMW", "SGDW", "AMSGRADW" (*W version for decoupled weight decay, RANGER_PLUS is ranger + adabelief + centralized_gradient)
clip                 | bool         | yes      | false (true if RANGER* selected) | gradients with absolute value greater than clip_value will be clipped to below values
clip_value           | real         | yes      | 5.0     | gradients with absolute value greater than clip_value will be clipped to this value
clip_norm            | real         | yes      | 100.0   | gradients with euclidean norm greater than clip_norm will be clipped to this value
rectified            | bool         | yes      | false   | rectified momentum variance ie https://arxiv.org/abs/1908.03265 valid for ADAM[W] and AMSGRAD[W]
adabelief            | bool         | yes      | false   | adabelief mod for ADAM https://arxiv.org/abs/2010.07468
gradient_centralization | bool         | yes      | false   | centralized gradient mod for ADAM ie https://arxiv.org/abs/2004.01461v2
test_interval        | int          | yes      | N/A     | Number of iterations between testing phases
test_initialization  | bool         | true     | N/A     | Whether to start training by testing the network
lr_policy            | string       | yes      | N/A     | learning rate policy ("step", "inv", "fixed", "sgdr", ...)
base_lr              | real         | yes      | N/A     | Initial learning rate
warmup_lr            | real         | yes      | N/A     | warmup starting learning rate (linearly goes to base_lr)
warmup_iter          | int          | yes      | 0       | number of warmup iterations
gamma                | real         | yes      | N/A     | Learning rate drop factor
stepsize             | int          | yes      | N/A     | Number of iterations between the dropping of the learning rate
stepvalue            | array of int | yes      | N/A     | Iterations at which a learning rate change takes place, with `multistep` `lr_policy'
momentum             | real         | yes      | N/A     | Learning momentum
period               | int          | yes      | -1      | N/A | Period in number of iterations with SGDR, best to use ncycles instead
ncycles              | int          | yes      | 1       | Number of restart cycles with SGDR
weight_decay         | real         | yes      | N/A     | Weight decay
power                | real         | yes      | N/A     | Power applicable to some learning rate policies
iter_size            | int          | yes      | 1       | Number of passes (iter_size * batch_size) at every iteration
rand_skip            | int          | yes      | 0       | Max number of images to skip when resuming training (only with segmentation or multilabel and Caffe backend)
lookahead            | bool         | yes      | false   | weither to use lookahead strategy from  https://arxiv.org/abs/1907.08610v1
lookahead_steps      | int          | yes      | 6       | number of lookahead steps for lookahead strategy
lookahead_alpha      | real         | yes      | 0.5     | size of step towards full lookahead
decoupled_wd_periods | int          | yes      | 4       | number of search periods for SGDW ADAMW AMSGRADW (periods end with a restart)
decoupled_wd_mult    | real         | yes      | 2.0     | muliplier of period for SGDW ADAMW AMSGRADW
lr_dropout           | real         | yes      | 1.0     | learning rate dropout, as in https://arxiv.org/abs/1912.00144 1.0 means no dropout, 0.0 means no learning at all (this value is the probability of keeping computed value and not putting zero)

Note: most of the default values for the parameters above are to be found in the Caffe files describing a given neural network architecture, or within Caffe library, therefore regarded as N/A at DeepDetect level.

Net:

Parameter       | Type | Optional | Default | Description
---------       | ---- | -------- | ------- | -----------
batch_size      | int  | yes      | N/A     | Training batch size
test_batch_size | int  | yes      | N/A     | Testing batch size

- Torch

General:

Parameter       | Type   | Optional | Default | Description
---------       | ----   | -------- | ------- | -----------
gpu             | bool   | yes      | false   | whether to use gpu
gpuid           | int or array | yes | 0      | GPU id, use single int for single GPU, `-1` for using all GPUs, and array e.g. `[1,3]` for selecting among multiple GPUs
nclasses        | int    | yes      | none    | if set to some int, add a classifier (linear/fullyConnected) with corresponding number of classes after torch traced model
ntargets   | int             | no (regression only)     | N/A     | Number of regression targets
self_supervised | string | yes      | ""      | self-supervised mode: "mask" for masked language model
embedding_size  | int    | yes      | 768     | embedding size for NLP models
freeze_traced   | bool   | yes      | false   | Freeze the traced part of the net during finetuning (e.g. for classification)
retain_graph	| bool	 | yes	    | false   | Whether to use `retain_graph` with torch autograd
template        | string | yes      | ""      | for language models, either "bert" or "gpt2", "recurrent" for LSTM-like models (including autoencoder), "nbeats" for nbeats model, "vit" for vision transformer
regression | bool            | yes                      | false   | Whether the model is a regressor
timesteps     | int            | yes      | N/A            | Number of timesteps for time models (LSTM/NBEATS...) : this sets the length of sequences that will be given for learning, every timestep contains inputs and outputs as defined by the csv/csvts connector
offset        | int            | yes      | N/A            | Offset beween start point of sequences with connector `cvsts`, defining the overlap of input series
forecast_timesteps      | int            | yes      | N/A       | for nbeats model, this gives the length of the forecast
backcast_timesteps      | int            | yes      | N/A       | for nbeats model, this gives the length of the backcast


Model instantiation parameters:

Parameter       | Template  | Type            | Default                      | Description
---------       | --------- | ------          | ---------------------------- | -----------
template_params.stackdef | nbeats    | array of string | ["t2","s8","g3","b3","h10" ] | default means: trend stack with theta = 2, seasonal stack with theta = 8 , generic stack with theta = 3, 3 blocks per stacks, hidden unit size of 10 everywhere
template_params.vit_flavor | vit | string | vit_base_patch16 | Vision transformer architecture, from smaller to larger: vit_tiny_patch16, vit_small_patch16, vit_base_patch32, vit_base_patch16, vit_large_patch16, vit_large_patch32, vit_huge_patch16, vit_huge_patch32
template_params.realformer | vit | bool | false | Whether to use the 'realformer' residual among attention heads
layers          | recurrent | array of string | []                           | ["L50","L50"] means 2 layers of LSTMs with hidden size of 50. ["L100","L100", "T", "L300"] means an lstm autoencoder with encoder composed of 2 LSTM layers of hidden size 100 and decoder is one LSTM layer of hidden size 300


Solver:

Parameter     | Type   | Optional | Default | Description
---------     | ----   | -------- | ------- | -----------
iterations    | int    | yes      | N/A     | Max number of solver's iterations
snapshot      | int    | yes      | N/A     | Iterations between model snapshots
solver_type   | string | yes      | SGD     | from "SGD", "ADAGRAD",  "RMSPROP", "ADAM", "RANGER", "RANGER_PLUS"
beta1         | real   | yes      | 0.9     | for RANGER* : beta1 param
beta2         | real   | yes      | 0.999   | for RANGER* : beta2 param
weight_decay  | real   | yes      | 0.0     | for RANGER* : weight decay
rectified     | bool   | yes      | true    | for RANGER* : enable/disable rectified ADAM
lookahead     | bool   | yes      | true    | for RANGER* : enable/disable lookahead
lookahead_steps | int  | yes      | 6       | for RANGER* : if lookahead enabled, number of steps
lookahead_alpha | real | yes      | 0.5     | for RANGER* : if lookahead enables, alpha param
adabelief     | bool   | yes      | false for RANGER, true for RANGER_PLUS   | for RANGER* : enable/disable adabelief
gradient_centralization | bool | yes | false for RANGER, true for RANGER_PLUS| for RANGER* : enable/disable gradient centralization
test_interval | int    | yes      | N/A     | Number of iterations between testing phases
base_lr       | real   | yes      | N/A     | Initial learning rate
iter_size     | int    | yes      | 1       | Number of passes (iter_size * batch_size) at every iteration
resume        | bool   | yes      | false   | Whether to resume training from solver state

Net:

Parameter       | Type | Optional | Default | Description
---------       | ---- | -------- | ------- | -----------
batch_size      | int  | yes      | N/A     | Training batch size
test_batch_size | int  | yes      | N/A     | Testing batch size


- Caffe2

General:

Parameter | Type         | Optional | Default | Description
--------- | ----         | -------- | ------- | -----------
gpu       | bool         | yes      | false   | Whether to use GPU
gpuid     | int or array | yes      | 0       | GPU id, use single int for single GPU, `-1` for using all GPUs, and array e.g. `[1,3]` for selecting among multiple GPUs
resume    | bool         | yes      | false   | Whether to resume training from .solverstate and .caffemodel files

Solver:

Parameter     | Type   | Optional | Default | Description
---------     | ----   | -------- | ------- | -----------
iterations    | int    | yes      | N/A     | Max number of solver's iterations
snapshot      | int    | yes      | N/A     | Iterations between model snapshots
solver_type   | string | yes      | SGD     | from "SGD", "ADAGRAD", "NESTEROV", "RMSPROP", "ADADELTA", "ADAM" and "AMSGRAD"
test_interval | int    | yes      | N/A     | Number of iterations between testing phases
lr_policy     | string | yes      | N/A     | learning rate policy ("step", "inv", "fixed", "sgdr", ...)
base_lr       | real   | yes      | N/A     | Initial learning rate
gamma         | real   | yes      | N/A     | Learning rate drop factor
stepsize      | int    | yes      | N/A     | Number of iterations between the dropping of the learning rate
momentum      | real   | yes      | N/A     | Learning momentum
power         | real   | yes      | N/A     | Power applicable to some learning rate policies

Net:

Parameter       | Type | Optional | Default | Description
---------       | ---- | -------- | ------- | -----------
batch_size      | int  | yes      | N/A     | Training batch size
test_batch_size | int  | yes      | N/A     | Testing batch size

- XGBoost

General:

Parameter     | Type   | Optional | Default                | Description
---------     | ----   | -------- | -------                | -----------
objective     | string | yes      | multi:softprob         | objective function, among multi:softprob, binary:logistic, reg:linear, reg:logistic
booster       | string | yes      | gbtree                 | which booster to use, gbtree or gblinear
num_feature   | int    | yes      | set by xgbbost         | maximum dimension of the feature
eval_metric   | string | yes      | according to objective | evaluation metric internal to xgboost
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
subsample        | double | yes      | 1.0     | subsample ratio of traning instance
colsample        | double | yes      | 1.0     | subsample ratio of columns when contructing each tree
lambda           | double | yes      | 1.0     | L2 regularization term on weights
alpha            | double | yes      | 0.0     | L1 regularization term on weights
lambda_bias      | double | yes      | 0.0     | L2 regularization for linear booster
tree_method      | string | yes      | auto    | tree construction algorithm, from auto, exact, approx
scale_pos_weight | double | yes      | 1.0     | control the balance of positive and negative weights

For more details on all XGBoost parameters see the dedicated page at https://github.com/dmlc/xgboost/blob/master/doc/parameter.md

- Tensorflow

Not implemented, see Predict

- TSNE

Parameter  | Type | Optional | Default | Description
---------  | ---- | -------- | ------- | -----------
perplexity | int  | yes      | 30      | perplexity is related to the number of nearest neighbors used to learn the manifold
iterations | int  | yes      | 5000    | number of optimization iterations


## Get information on a training job

> Requesting the status of an asynchronous training job:

```shell
curl -X GET "http://localhost:8080/train?service=myserv&job=1"

{"status":{"code":200,"msg":"OK"},"head":{"method":"/train","job":1,"status":"running","time":74.0},"body":{"measure":{"iteration":445.0,"train_loss":0.7159726023674011,"mcll":2.1306082640485237,"acc":0.16127989657401424}}}
```
```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

dd.get_train('myserv',job=1)
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
curl -X DELETE "http://localhost:8080/train?service=myserv&job=1"

{"status":{"code":200,"msg":"OK"},"head":{"time":196.0,"status":"terminated","method":"/train","job":1}}
```
```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

dd.delete_train('myserv',job=1)
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

predict_output = dd.post_predict('myserv',data,parameters_input,parameters_mllib,parameters_output)
```

> Prediction from CSV file:

```shell
curl -X POST "http://localhost:8080/predict" -d "{\"service\":\"covert\",\"parameters\":{\"input\":{\"id\":\"Id\",\"separator\":\",\",\"scale\":true}},\"data\":[\"models/covert/test10.csv\"]}"

{"status":{"code":200,"msg":"OK"},"head":{"method":"/predict","time":16.0,"service":"covert"},"body":{"predictions":[{"uri":"15121","loss":0.0,"classes":{"prob":0.9999997615814209,"cat":"6"}},{"uri":"15122","loss":0.0,"classes":{"prob":0.9962882995605469,"cat":"5"}},{"uri":"15130","loss":0.0,"classes":{"prob":0.9999340772628784,"cat":"1"}},{"uri":"15123","loss":0.0,"classes":{"prob":1.0,"cat":"3"}},{"uri":"15124","loss":0.0,"classes":{"prob":1.0,"cat":"3"}},{"uri":"15128","loss":0.0,"classes":{"prob":1.0,"cat":"1"}},{"uri":"15125","loss":0.0,"classes":{"prob":0.9999998807907105,"cat":"3"}},{"uri":"15126","loss":0.0,"classes":{"prob":0.7535045146942139,"cat":"3"}},{"uri":"15129","loss":0.0,"classes":{"prob":0.9999986886978149,"cat":"1"}},{"uri":"15127","loss":0.0,"classes":{"prob":1.0,"cat":"1"}}]}}
```

> Prediction over test set, with output metrics

```shell
curl -X POST 'http://localhost:8080/predict' -d '{"service":"n20","parameters":{"mllib":{"gpu":true},"output":{"measure":["f1"]}},"data":["/path/to/news20/"]}'

{"status":{"code":200,"msg":"OK"},"head":{"method":"/predict","time":18271.0,"service":"n20"},"body":{"measure":{"f1":0.8152690151793434,"recall":0.8219119954158582,precision":0.8087325557838578,"accp":0.815365025466893}}}
```

```python
from dd_client import DD

dd = DD('localhost')
dd.set_return_format(dd.RETURN_PYTHON)

data = ['models/covert/test10.csv']
parameters_input = {'id':'id','separator':',',scale:True}
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

Note: it is good practice to configure the `input` connector at service creation, and then leave it's parameters empty at `predict` time.

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
bw           | bool         | yes      | false   | Treat images as black & white (Caffe only)
mean         | float        | yes      | 128     | mean pixel value to be subtracted to input image (`tensorflow` only)
mean         | array of int | yes      | N/A     | mean pixel value per channel to be subtracted to input image (`caffe` only)
std          | float        | yes      | 128     | standard pixel value deviation to be applied to input image (`tensorflow` only)
segmentation | yes          | yes      | false   | whether a segmentation service
interp       | string       | yes      | cubic   | Image interpolation method (cubic, linear, nearest, lanczos4, area)
cuda         | bool         | yes      | false   | Whether to use CUDA to resize images (use USE_CUDA_CV=ON build flag)

- CSV (`csv`)

Parameter            | Type            | Optional | Default | Description
---------            | ----            | -------- | ------- | -----------
ignore               | array of string | yes      | empty   | Array of column names to ignore
separator            | string          | yes      | ','     | Column separator character
id                   | string          | yes      | empty   | Column name of the training examples identifier field, if any
scale                | bool            | yes      | false   | Whether to scale all values into [0,1]
min_vals,max_vals    | array           | yes      | empty   | Instead of `scale`, provide the scaling parameters, as returned from a training call
categoricals_mapping | object          | yes      | empty   | Categorical mappings, as returned from a training call

- Text (`txt`)

Parameter       | Type   | Optional | Default                                            | Description
---------       | ----   | -------- | -------                                            | -----------
count           | int    | yes      | true                                               | whether to count words and report counters
min_count       | int    | yes      | 5                                                  | min word count occurences for a word to be taken into account
min_word_length | int    | yes      | 5                                                  | min word length for a word to be taken into account
tfidf           | bool   | yes      | false                                              | whether to compute TF/IDF for every word
sentences       | bool   | yes      | false                                              | whether to turn every line into a document (requires dataset as file with one sentence per line in every class repository)
characters      | bool   | yes      | false                                              | character-level text processing, as opposed to word-based text processing
sequence        | int    | yes      | N/A                                                | for character-level text processing, the fixed length of each sample of text
read_forward    | bool   | yes      | false                                              | for character-level text processing, whether to read content from left to right
alphabet        | string | yes      | abcdefghijklmnopqrstuvwxyz 0123456789 ,;.!?:'"/\\\ | _@#$%^&*~`+-=<>()[]{} | for character-level text processing, the alphabet of recognized symbols
sparse          | bool   | yes      | false                                              | whether to use sparse features (and sparce computations with Caffe for huge memory savings, for xgboost use `svm` connector instead)

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
regression           | bool   | yes      | false   | whether the output of a model is a regression target (i.e. vector of one or more floats)
rois                 | string | yes      | empty                   | set the ROI layer from which to extract the features from bounding boxes. Both the boxes and features ar returned when using an object detection model with ROI pooling layer
index                | bool   | yes      | false                   | whether to index the output from prediction, for similarity search
build_index          | bool   | yes      | false                   | whether to build similarity index after prediction, no more indexing can be done afterward
search               | bool   | yes      | false                   | whether to use the predicted output for similarity search and return pre-indexed nearest neighbors
search_nn            | int    | yes      | 10                      | number of similarity search results
multibox_rois        | bool   | yes      | false                   | aggregates bounding boxes ROIs features (requires `rois`) for image similarity search
index_type           | string | yes      | Flat                    | for faiss index indexing backend only : a FAISS index factory string , see https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
index_gpu            | bool   | yes      | false                   | for faiss indexing backend only : if available, build idnex on GPU
index_gpuid          | int    | yes      | all                     | for faiss indexing backend only : which gpu to use if index_gpu is true
train_samples        | int    | yes      | 100000                  | for faiss indexing backend only :  number of samples to use for training index. Larger values lead to better indexes (more evenly distributed) but cause much larger index training time. Many indexes need a minimal value depending on the number of clusters built,  see https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index.
ondisk               | bool   | yes      | true                    | for faiss indexing backend only :  try to directly build indexes on mmaped files (IVF index_types only can do so)
nprobe               | int    | yes      | max(ninvertedlist/50,2) | for faiss indexing backend only : number of cluster searched for closest images: for highly compressing indexes, setting nprobe to larger values may allow better precision
ctc                  | bool   | yes      | false                   | whether the output is a sequence (using CTC encoding)
confidences          | array  | yes      | empty                   | Segmentation only: output confidence maps for "best" class, "all" classes, or classes being specified by number, e.g. "1","3".
logits_blob          | string | yes      | ""                      | in classification services, this add raw logits to output. Usefull for calibration purposes
logits               | bool   | yes      | False                   | in detection services, this add logits to output. Usefull for calibration purposes.

- Network object

Parameter    | Type   | Optional | Default                        | Description
---------    | ----   | -------- | -------                        | -----------
url          | string | no       | N/A                            | URL of the remote service to connect to (e.g http://localhost:9200)
http_method  | string | yes      | POST                           | HTTP connecting method, from "POST", "PUT", etc...
content_type | string | yes      | Content-Type: application/json | Content type HTTP header string

The variables that are usable in the output template format are those from the standard JSON output. See the [output template](#templates) dedicated section for more details and examples.

#### Machine learning libraries

- Caffe / Caffe2

Parameter     | Type         | Optional | Default | Description
---------     | ----         | -------- | ------- | -----------
gpu           | bool         | yes      | false   | Whether to use GPU
gpuid         | int or array | yes      | 0       | GPU id, use single int for single GPU, `-1` for using all GPUs, and array e.g. `[1,3]` for selecting among multiple GPUs
extract_layer | string       | yes      | false   | name of the neural net's inner layer to return as output. Requires the service to be declared as 'unsupervised'

Net:

Parameter       | Type | Optional | Default | Description
---------       | ---- | -------- | ------- | -----------
test_batch_size | int  | yes      | N/A     | Prediction batch size (the server iterates as many batches as necessary to predict over all posted data)

- Torch

Parameter     | Type   | Optional | Default | Description
---------     | ----   | -------- | ------- | -----------
gpu           | bool   | yes      | false   | Whether to use GPU
gpuid         | int or array | yes | 0      | GPU id, use single int for single GPU, `-1` for using all GPUs, and array e.g. `[1,3]` for selecting among multiple GPUs
extract_layer | string | yes      | ""      | in bert models "hidden_state" allows to extract raw hidden_states values to return as output. Requires the service to be declared as 'unsupervised'


- XGBoost

No parameter required.

- Tensorflow

Parameter       | Type   | Optional | Default | Description
---------       | ----   | -------- | ------- | -----------
test_batch_size | int    | yes      | N/A     | Prediction batch size (the server iterates as many batches as necessary to predict over all posted data)
inputlayer      | string | yes      | auto    | network input layer name
outputlayer     | string | yes      | auto    | network output layer name
extract_layer   | string | yes      | false   | name of the neural net's inner layer to return as output. Requires the service to be declared as 'unsupervised' (subsumes `outputlayer` in an `unsupervised` service)

- NCNN

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
* `txt` instantiates the input connector for text files

Input connectors work almost the same during both the training and prediction phases. But the training phase usually deals with large masses of data, and therefore the connectors above are optimized to automate some tasks, typically building and preprocessing the dataset at training time.

Below is a summary of input connectors options, though they are all already defined in each API resource and call documentation.

- Image (`image`)

Parameter  | Type | Optional | Default | Description
---------  | ---- | -------- | ------- | -----------
width      | int  | yes      | 227     | Resize images to width ("image" only)
height     | int  | yes      | 227     | Resize images to height ("image" only)
bw         | bool | yes      | false   | Treat images as black & white
test_split | real | yes      | 0       | Test split part of the dataset
shuffle    | bool | yes      | false   | Whether to shuffle the training set (prior to splitting)
seed       | int  | yes      | -1      | Shuffling seed for reproducible results (-1 for random seeding)

- CSV (`csv`)

Parameter            | Type            | Optional | Default | Description
---------            | ----            | -------- | ------- | -----------
label                | string          | no       | N/A     | Label column name
ignore               | array of string | yes      | empty   | Array of column names to ignore
label_offset         | int             | yes      | 0       | Negative offset (e.g. -1) s othat labels range from 0 onward
separator            | string          | yes      | ','     | Column separator character
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
min_count       | int    | yes      | 5                                                  | min word count occurences for a word to be taken into account
min_word_length | int    | yes      | 5                                                  | min word length for a word to be taken into account
tfidf           | bool   | yes      | false                                              | whether to compute TF/IDF for every word
sentences       | bool   | yes      | false                                              | whether to turn every line into a document (requires dataset as file with one sentence per line in every class repository)
characters      | bool   | yes      | false                                              | character-level text processing, as opposed to word-based text processing
sequence        | int    | yes      | N/A                                                | for character-level text processing, the fixed length of each sample of text
read_forward    | bool   | yes      | false                                              | for character-level text processing, whether to read content from left to right
alphabet        | string | yes      | abcdefghijklmnopqrstuvwxyz 0123456789 ,;.!?:'"/\\\ | _@#$%^&*~`+-=<>()[]{} | for character-level text processing, the alphabet of recognized symbols
test_split      | real   | yes      | 0                                                  | Test split part of the dataset
shuffle         | bool   | yes      | false                                              | Whether to shuffle the training set (prior to splitting)
seed            | int    | yes      | -1                                                 | Shuffling seed for reproducible results (-1 for random seeding)
db              | bool   | yes      | false                                              | whether to gather data into a database, useful for very large datasets, allows training in constant-size memory
sparse          | bool   | yes      | false                                              | whether to use sparse features (and sparce computations with Caffe for huge memory savings, for xgboost use `svm` connector instead)

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
regression           | bool   | yes      | false   | whether the output of a model is a regression target (i.e. vector of one or more floats)
rois                 | string | yes      | empty                   | set the ROI layer from which to extract the features from bounding boxes. Both the boxes and features ar returned when using an object detection model with ROI pooling layer
index                | bool   | yes      | false                   | whether to index the output from prediction, for similarity search
build_index          | bool   | yes      | false                   | whether to build similarity index after prediction, no more indexing can be done afterward
search               | bool   | yes      | false                   | whether to use the predicted output for similarity search and return pre-indexed nearest neighbors
search_nn            | int    | yes      | 10                      | number of similarity search results
multibox_rois        | bool   | yes      | false                   | aggregates bounding boxes ROIs features (requires `rois`) for image similarity search
index_type           | string | yes      | Flat                    | for faiss index indexing backend only : a FAISS index factory string , see https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
index_gpu            | bool   | yes      | false                   | for faiss indexing backend only : if available, build idnex on GPU
index_gpuid          | int    | yes      | all                     | for faiss indexing backend only : which gpu to use if index_gpu is true
train_samples        | int    | yes      | 100000                  | for faiss indexing backend only :  number of samples to use for training index. Larger values lead to better indexes (more evenly distributed) but cause much larger index training time. Many indexes need a minimal value depending on the number of clusters built,  see https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index.
ondisk               | bool   | yes      | true                    | for faiss indexing backend only :  try to directly build indexes on mmaped files (IVF index_types only can do so)
nprobe               | int    | yes      | max(ninvertedlist/50,2) | for faiss indexing backend only : number of cluster searched for closest images: for highly compressing indexes, setting nprobe to larger values may allow better precision
ctc                  | bool   | yes      | false                   | whether the output is a sequence (using CTC encoding)
confidences          | array  | yes      | empty                   | Segmentation only: output confidence maps for "best" class, "all" classes, or classes being specified by number, e.g. "1","3".
logits_blob          | string | yes      | ""                      | in classification services, this add raw logits to output. Usefull for calibration purposes
logits               | bool   | yes      | False                   | in detection services, this add logits to output. Usefull for calibration purposes.


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
{"parameters":{"mllib":{"template":"mlp","nclasses":9,"layers":[512,512,512],"activation":"PReLU","nclasses":9}}
```

> Example of GoogleNet for 1000 classes of images:

```json
{"parameters":{"input":{"connector":"image","width":224,"height":224},"mllib":{"template":"googlenet","nclasses":1000}}
```

The DeepDetect server and API come with a set of Machine Learning model templates.

At the moment these templates are available for the [Caffe](https://caffe.berkeleyvision.org/) Deep Learning library. They include some of the most powerful deep neural net architectures for image classification, and other customizable classic and useful architectures.

## Neural network templates

All models below are used by passing their id to the `mllib/template` parameter in `PUT /services` calls:

Model ID    | Type                     | Input          | Description
--------    | ----                     | -----          | -----------
lregression | linear                   | CSV / Txt            | logistic regression
mlp         | neural net               | CSV / Txt            | multilayer perceptron, fully configurable from API, see parameters below
recurrent   | recurrent neural net     | CSV / CSVTS    | LSTM-based networks, fully configurable from API
convnet     | convolutional neural net | Images         | convolutional neural net, with layers configurable from API, see parameters below
alexnet     | deep neural net          | Images 227x227 | 'AlexNet', convolutional deep neural net, good accuracy, fast
cifar       | deep neural net          | Images 32x32   | Convolutional deep neural net, very good for small images
nin         | deep neural net          | Images 224x224 | 'Network in Network' convolutional deep neural net, good accuracy, very fast
googlenet   | deep neural net          | Images 224x224 | 'GoogleNet', convolutional deep neural net, good accuracy
resnet_10   | deep neural net          | Image 224x224  | 'ResNet', 10-layers deep residual convolutional neural net, top accuracy
resnet_18   | deep neural net          | Image 224x224  | 'ResNet', 18-layers deep residual convolutional neural net, top accuracy
resnet_32   | deep neural net          | Image 224x224  | 'ResNet', 32-layers deep residual convolutional neural net, top accuracy
resnet_50   | deep neural net          | Image 224x224  | 'ResNet', 50-layers deep residual convolutional neural net, top accuracy
resnet_101  | deep neural net          | Image 224x224  | 'ResNet', 101-layers deep residual convolutional neural net, top accuracy
resnet_152  | deep neural net          | Image 224x224  | 'ResNet', 152-layers deep residual convolutional neural net, top accuracy
crnn	    | deep neural net	       | Images   		| Convolutional network plus CTC head for OCR
crnn_resnet_18 | deep neural net	       | Images   		| Convolutional network plus CTC head for OCR with ResNet-18 base
crnn_resnet_50 | deep neural net	       | Images   		| Convolutional network plus CTC head for OCR with ResNet-50 base
crnn_resnext_50 | deep neural net	       | Images   		| Convolutional network plus CTC head for OCR with ResNext-50 base
enet | deep neural net	       | Images   		| Convolutional network for segmentation
mobilenet_v2 | deep neural net	       | Images   		| Lightweight network for image classification
mobilenet_v2_ssd | deep neural net	       | Images   		| Lightweight network for object detection
pspnet_50 | deep neural net	       | Images   		| Convolutional network for segmentation with ResNet-50 base
pspnet_101 | deep neural net	       | Images   		| Convolutional network for segmentation with ResNet-101 base
pspnet_vgg16 | deep neural net	       | Images   		| Convolutional network for segmentation with VGG-16 base
refinedet_512 | deep neural net	       | Images   		| Convolutional network for object detection with VGG-16 base
refinedet_vovnet27slim_512 | deep neural net	       | Images   		| Convolutional network for object detection with VovNet-27-slim base
refinedet_vovnet39_512 | deep neural net	       | Images   		| Convolutional network for object detection with VovNet-39 base
resnet_10_ssd | deep neural net	       | Images   		| Convolutional network for object detection with ResNet-10 base
resnet_18_ssd | deep neural net	       | Images   		| Convolutional network for object detection with ResNet-18 base
resnet_34_ssd | deep neural net	       | Images   		| Convolutional network for object detection with ResNet-34 base
segnet | deep neural net	       | Images   		| Convolutional network for segmentation
se_net | deep neural net	       | Images   		| Convolutional network for segmentation
se_resnet_50 | deep neural net	       | Images   		| Convolutional network for image classification
se_resnet_101 | deep neural net	       | Images   		| Convolutional network for image classification
se_resnet_152 | deep neural net	       | Images   		| Convolutional network for image classification
se_resnext_50 | deep neural net	       | Images   		| Convolutional network for image classification
se_resnext_101 | deep neural net	       | Images   		| Convolutional network for image classification
se_resnet_50_ssd | deep neural net	       | Images   		| Convolutional network for image classification
shufflenet | deep neural net	       | Images   		| Lightweight network for image classification
squeezenet | deep neural net	       | Images   		| Lightweight network for image classification
squeezenet_ssd | deep neural net	       | Images   		| Lightweight network for object detection
ssd_300 | deep neural net	       | Images   		| Convolutional network for object detection
ssd_300_res_128 | deep neural net	       | Images   		| Convolutional network for object detection wit ResNet tip
ssd_512 | deep neural net	       | Images   		| Convolutional network for object detection
ssd_512_res_128 | deep neural net	       | Images   		| Convolutional network for object detection with ResNet tip
unet | deep neural net	       | Images   		|  Convolutional network for segmentation
vdcnn_17 | deep neural net	       | Images   		| Convolutional network for text classification
vdcnn_9 | deep neural net	       | Images   		| Convolutional network for text classification
vgg_16 | deep neural net	       | Images   		| Convolutional network for image classification

## Parameters

- Caffe

Parameter  | Type            | Optional                 | Default | Description
---------  | ----            | --------                 | ------- | -----------
nclasses   | int             | no (classification only) | N/A     | Number of output classes ("supervised" service type)
ntargets   | int             | no (regression only)     | N/A     | Number of regression targets
template   | string          | yes                      | empty   | Neural network template, from "lregression", "mlp", "convnet", "alexnet", "googlenet", "nin"
layers     | array of int    | yes                      | [50]    | Number of neurons per layer ("mlp" only)
layers     | array of string | yes                      | [1000]  | Type of layer and number of neurons peer layer: XCRY for X successive convolutional layers of Y filters with activation layers followed by a max pooling layer, an int as a string for specifying the final fully connected layers size, e.g. \["2CR32","2CR64","1000"\] ("convnet" only), ["AR5", "A5" "L50","A50", "L50"] means an affine / linear / innerproduct layer of size 5, followed by a ReLU, followed by a linear layer of size 5, then 1 layer of LSTMs with hidden size of 50, then one linear layer of size 50 then another lstm layer. ["L100","L100", "T", "L300"] means an lstm autoencoder with encoder composed of 2 LSTM layers of hidden size 100 and decoder is one LSTM layer of hidden size 300 ("recurrent" only)
activation | string          | yes                      | relu    | Unit activation ("mlp" and "convnet" only), from "sigmoid","tanh","relu","prelu"
dropout    | real            | yes                      | 0.5     | Dropout rate between layers ("mlp" and "convnet" only)
regression | bool            | yes                      | false   | Whether the model is a regressor
crop_size  | int             | yes                      | N/A     | Size of random image crops as input images
rotate     | bool            | yes                      | false   | Whether to apply random rotations to input images
mirror     | bool            | yes                      | false   | Whether to apply random mirroring of input images

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
