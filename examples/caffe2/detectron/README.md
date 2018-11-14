# Detectron and DeepDetect


### Table of Contents

**[Requirements](#requirements)**

**[Get a model](#get-a-model)**

**[Create a service](#create-a-service)**

**[Predict](#predict)**

### Requirements

DeepDetect must be compiled with the ```-DUSE_CAFFE2=ON``` flag
Detectron must be installed if you wish to convert .pkl/.yaml files (default format of Detectron models) into .pb/.pbtxt (caffe2 format used by DeepDetect)

### Get a model

- #### In caffe2 format

You can find some of them [here](https://github.com/caffe2/models/tree/master/detectron)

Both ```init_net.pb``` and ```predict_net.pb``` files must be present in your model repository (the other files aren't necessary)

- #### In Detectron format

Download the weights (model_final.pkl) from the [model zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md)

And find the corresponding training [configuration file](https://github.com/facebookresearch/Detectron/tree/master/configs) (*.yaml)

Then convert them using the [tool](https://github.com/facebookresearch/Detectron/blob/master/tools/convert_pkl_to_pb.py) provided in the Detectron repository

And finally, place the .pb files into the model repository:
- ```model_init.pb``` corresponds to ```init_net.pb```
- ```model.pb``` corresponds to ```predict_net.pb```

**WARNING: Those files cannot be used to infer masks, only bounding boxes**

###### Example

Here is a script to download and convert a **Fast R-CNN**:

```
#!/bin/sh

#Model R-50-C4 (with 2x the default learning-rate schedule)
WEIGHTS=https://s3-us-west-2.amazonaws.com/detectron/35857281/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml.01_34_56.ScPH0Z4r/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl

#Config e2e_faster_rcnn_R-50-C4_2x.yaml
CONFIG=https://raw.githubusercontent.com/facebookresearch/Detectron/master/configs/12_2017_baselines/e2e_faster_rcnn_R-50-C4_2x.yaml

#Paths
DETECTRON=/home/foo/Detectron
WORKSPACE=/home/foo/tmp
REPOSITORY=/home/foo/rcnn_model

mkdir $WORKSPACE
cd $WORKSPACE
wget $WEIGHTS -O weights.pkl
wget $CONFIG -O config.yaml
python $DETECTRON/tools/convert_pkl_to_pb.py --out_dir . --cfg config.yaml DOWNLOAD_CACHE . TRAIN.WEIGHTS weights.pkl TEST.WEIGHTS weights.pkl

mkdir $REPOSITORY
cp model_init.pb $REPOSITORY/init_net.pb
cp model.pb $REPOSITORY/predict_net.pb
rm weights.pkl config.yaml model_def.png model.pbtxt model.pb model_init.pb
rmdir $WORKSPACE
```

- #### With mask support

To be able to generate masks, the model need two more files

They can be [generated](convert_pkl_to_mask_net.py) using the same '.pkl' and '.yaml' as before

The newly created files (also called 'predict_net.pb' and 'init_net.pb') must be used **in addition** to the classic ones

(See "[create a service](#create-a-service)" for more details)

###### Example

```
#!/bin/sh

#Model R-50-C4 (with 2x the default learning-rate schedule)
WEIGHTS=https://s3-us-west-2.amazonaws.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl

#Config e2e_faster_rcnn_R-50-C4_2x.yaml
CONFIG=https://raw.githubusercontent.com/facebookresearch/Detectron/master/configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml

#Paths
DETECTRON=/home/foo/Detectron
DEEPDETECT=/home/foo/deepdetect
WORKSPACE=/home/foo/tmp
REPOSITORY=/home/foo/rcnn_model

mkdir $WORKSPACE
cd $WORKSPACE
wget $WEIGHTS -O weights.pkl
wget $CONFIG -O config.yaml
python $DETECTRON/tools/convert_pkl_to_pb.py --out_dir . --cfg config.yaml DOWNLOAD_CACHE . TRAIN.WEIGHTS weights.pkl TEST.WEIGHTS weights.pkl

mkdir $REPOSITORY
cp model_init.pb $REPOSITORY/init_net.pb
cp model.pb $REPOSITORY/predict_net.pb

mkdir $REPOSITORY/mask
python $DEEPDETECT/examples/caffe2/detectron/convert_pkl_to_mask_net.py --out_dir $REPOSITORY/mask --cfg config.yaml --wts weights.pkl

rm weights.pkl config.yaml model_def.png model.pbtxt model.pb model_init.pb
rmdir $WORKSPACE
```

- #### With its class labels

If you do not have access to the training dataset or the classes that were used during the training, it may imply that the default 80-classes COCO dataset was used. Its classes are listed [here](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/dummy_datasets.py)

You can also [generate](generate_coco_corresp_file.py) a 'corresp.txt' file in your model repository

### Create a service

- #### With the default configuration

To use a detectron model, the following flags must be set:
- ```'mllib': 'caffe2'```
- ```'connector': 'image'```

```
curl -X PUT "http://localhost:8080/services/my_service" -d '{
  "description": "detectron",
  "type": "supervised",
  "mllib": "caffe2",
  "model": { "repository": "my_model" },
  "parameters": {
    "input": {
      "connector": "image",
      "height": 1000,
      "width": 1000,
      "mean": [128, 128, 128]
    },
    "mllib": { "gpu": true }
  }
}'
```

- #### With the same configuration as the python version

In this case, you want the images to be scaled but not stretched.
You can do that with the flags ```scale_min``` and ```scale_max```.
(Images are scaled until their width and height are at least 'scale_min' and and most 'scale_max'. If it's not possible, 'scale_min' is ignored).

You can find them in the .yaml file:
- Either TEST.SCALE or the first value of TRAIN.SCALES can be used as a 'scale_min'
- Either TEST.MAX_SIZE or TRAIN.MAX_SIZE can be used as a 'scale_max'

You can also use the default scales of Detectron (600 and 1000) by setting ```'scale': true```.

Finally, you can find the 'mean values' of the model inside the .yaml file, under the name PIXEL_MEANS.
Or use the default 'mean values' of Detectron: ```'mean': [102.9801, 115.9465, 122.7717]```

```
curl -X PUT "http://localhost:8080/services/my_service" -d '{
  "description": "detectron",
  "type": "supervised",
  "mllib": "caffe2",
  "model": { "repository": "my_model" },
  "parameters": {
    "input": {
      "connector": "image",
      "scale_min":800,
      "scale_max":1333,
      "mean": [102.9801, 115.9465, 122.7717]
    },
    "mllib": { "gpu": true }
  }
}'
```

**WARNING: Using scaled inputs will force a batch size of 1**

- #### With masks as outputs

Because masks are generated using more files (see "[get a model](#get-a-model)" for more details), an additional path must be given to the API when registering the repository

If your model repository looks like this:

* my_model
   * predict_net.pb
   * init_net.pb
   * mask
     * predict_net.ob
     * init_net.pb

Then the following flag must be set:
- ```'repository': 'my_model'```
- ```'extensions': [{'repository' : 'my_model/mask', 'type' : 'mask'}]```

Note that the repository is optional. In not set, a subdirectory whose name is the type will be used. It means that in the current case, the flags can be simplified to:
- ```'repository': 'my_model'```
- ```'extensions': [{'type' : 'mask'}]```

```
curl -X PUT "http://localhost:8080/services/my_service" -d '{
  "description": "detectron",
  "type": "supervised",
  "mllib": "caffe2",
  "model": {
    "repository": "my_model",
    "extensions': [{"type" : "mask"}]
  },
  "parameters": {
    "input": { "connector": "image" },
    "mllib": { "gpu": true }
  }
}'
```

### Predict

- #### Bounding boxes

The only flag to set is ```'bbox': true```:

```
curl -X POST "http://localhost:8080/predict" -d '{
  "service": "my_service",
  "parameters": {
    "output": {
      "bbox": true,
      "best": 1,
      "confidence_threshold": 0.7
    }
  },
  "data": [ "my_image_01.jpg", "my_image_02.jpg", "my_image_03.jpg" ]
}'
```

- #### Masks

The only flag to set is ```'mask': true```:

```
curl -X POST "http://localhost:8080/predict" -d '{
  "service": "my_service",
  "parameters": {
    "output": {
      "mask": true,
      "best": 1,
      "confidence_threshold": 0.7
    }
  },
  "data": [ "my_image_01.jpg", "my_image_02.jpg", "my_image_03.jpg" ]
}'
```

The API will then return a new field in addition to the bounding box:
```
{
  "cat": "boat",
  "prob": 0.997,
  "bbox": {
    "xmin": 318.044,
    "ymin": 284.642,
    "xmax': 465.683,
    "ymax': 360.258
  },
  "mask": {
    "format": "HW",
    "width': 148,
    "height": 77,
    "data": [0, 0, 0, 1, 0, 1, 1, ..., 0, 0]
  }
}
```

###### Example

You can use [this](plot_masks.py) script as an example on how to use the API results

Start DeepDetect, register your service, and run :
```
./plot_masks.py                                \
   --host=localhost --port=8080                \
   --service=service_name --threshold=0.8      \
   --pdf=/path/to/the/output/file.pdf          \
   /path/to/image1.jpg /path/to/image2.jpg ...
```
