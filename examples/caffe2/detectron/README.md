### Requirements

DeepDetect must be compiled with the ```-DUSE_CAFFE2=ON``` flag
Detectron must be installed if you wish to convert .pkl/.yaml files (default format of Detectron models) into .pb/.pbtxt (caffe2 format)

### Download models

#### Option 1 - In their caffe2 format

You can find some of them here :
https://github.com/caffe2/models/tree/master/detectron

Both ```init_net.pb``` and ```predict_net.pb``` files must be present in your model repository (the other files aren't necessary)

#### Option 2 - In their Detectron format

Download the weights (model_final.pkl) from the model zoo:
ttps://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md

And find the corresponding training configuration (*.yaml):
https://github.com/facebookresearch/Detectron/tree/master/configs

Then convert them using this script:
https://github.com/facebookresearch/Detectron/blob/master/tools/convert_pkl_to_pb.py

And finally, place the .pb files into the model repository:
- ```model_init.pb``` corresponds to ```init_net.pb```
- ```model.pb``` corresponds to ```predict_net.pb```

###### Example

Here is a script to download and convert a **Fast R-CNN**:
```
#!/bin/bash

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

### Retrieve the class names

If you do not have access to the training dataset or the classes that were used during the training, it may imply that the default 80-classes COCO dataset was used. Its classes are listed here:
https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/dummy_datasets.py

The following python script allows you to create a 'corresp.txt' file in your model repository:
```
#!/usr/bin/env python

REPOSITORY='/home/foo/my_model'

import detectron.datasets.dummy_datasets as dummy_datasets
classes = dummy_datasets.get_coco_dataset().classes
corresp = '\n'.join('{} {}'.format(i, classes[i]) for i, _ in enumerate(classes))
open(REPOSITORY + '/corresp.txt', 'w').write(corresp)
```

### Create the service

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

If you want the same behavior as the python version, images must be scaled but not stretched.
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

### Predict

The only flag to set is 'bbox':
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
