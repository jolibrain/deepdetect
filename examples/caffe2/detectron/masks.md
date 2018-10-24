### Masks

***This is a step by step example on how to get masks with DeepDetect using a Detectron model***

```
#!/bin/sh 


# Start DeepDetect and define the following variables with their correct values


export DD_REPO=/home/foo/deepdetect # DeepDetect Repository
export DD_PYTHONPATH=$DD_REPO/build/python_path # Build of DeepDetect
export DD_URL=http://localhost:8080 # API URL
export WORKSPACE=/home/foo/tmp # Some location to download files
export IMAGE=https://upload.wikimedia.org/wikipedia/commons/b/be/Cattle_dog_with_tennis_ball.jpg # Image URL
# Weights of your model
export DTRON_WTS=https://s3-us-west-2.amazonaws.com/detectron/35859007/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml.01_49_07.By8nQcCH/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl
# Configuration of your model
export DTRON_CFG=https://raw.githubusercontent.com/facebookresearch/Detectron/master/configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml


# Then prepare your workspace


export PYTHONPATH=$PYTHONPATH:$DD_PYTHONPATH/pytorch:$DD_PYTHONPATH/detectron
mkdir $WORKSPACE
mkdir $WORKSPACE/inputs
mkdir $WORKSPACE/outputs
mkdir $WORKSPACE/detectron_model
cd $WORKSPACE


# Download the files


wget $DTRON_WTS -O detectron_model/weights.pkl
wget $DTRON_CFG -O detectron_model/config.yaml
wget $IMAGE -O inputs/dogs.jpg


# Convert the model


python $DD_REPO/examples/caffe2/detectron/convert_pkl_to_pb.py \
    --out_dir deepdetect_model \
    --mask_dir deepdetect_model/mask \
    --cfg detectron_model/config.yaml \
    --wts detectron_model/weights.pkl \
    --coco


# Register the service


curl -X PUT "http://localhost:8080/services/mask" -d '{
  "description": "detectron",
  "type": "supervised",
  "mllib": "caffe2",
  "model": {
    "repository": "'$WORKSPACE'/deepdetect_model",
    "extensions": [{"type": "mask"}]
  },
  "parameters": {
    "input": {
      "connector": "image",
      "mean": [102.9801, 115.9465, 122.7717],
      "height": 800,
      "width": 1216
    },
    "mllib": { "gpu": true }
  }
}'


# Predict


curl -X POST "http://localhost:8080/predict" -d '{
  "service": "mask",
  "parameters": {
    "output": {
      "mask": true,
      "best": 1,
      "confidence_threshold": 0.7
    }
  },
  "data": ["'$WORKSPACE'/inputs/dogs.jpg"]
}' > outputs/dogs_mask.json


# The masks are arrays of 0 (nothing) and 1 (mask) with the same shape and position as the corresponding bounding box.
# You can easily plot this using python


python -c '
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the image
img = np.array(Image.open("inputs/dogs.jpg").convert("RGBA"))

# Plot it
fig, ax = plt.subplots(1)
ax.imshow(img)

# Load the predictions
items = json.load(open("outputs/dogs_mask.json"))["body"]["predictions"][0]["classes"]

# Loop over them
for item in items:
    xmin = int(item["bbox"]["xmin"])
    ymin = int(item["bbox"]["ymin"])
    width = item["mask"]["width"]
    height = item["mask"]["height"]
    mask = np.array(item["mask"]["data"]).reshape(height, width)
    
    # Format the mask
    mask = mask.astype(float) * 255 # Set the mask to a 255 capped value
    mask = np.stack((mask,) * 4, -1) # Copy it 4 times (one per ARGB channel)
    mask[...,-1] *= 0.9 # Apply a transparancy coefficient
    mask[...,:-1] *= np.random.uniform(0.3, 0.7, 3) # Apply a random color
    
    # Put the mask in an image-sized buffer
    buff = np.zeros(img.shape, dtype="uint8")
    buff[ymin:ymin+height, xmin:xmin+width] = mask

    # Plot everything
    ax.imshow(buff)
    ax.add_patch(patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor="g", facecolor="none"))
    ax.text(xmin, ymin, "{} {:.2f}".format(item["cat"], item["prob"]))

plt.show() # Show'
```
