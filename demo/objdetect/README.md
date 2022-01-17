### Object detection demo

This is a small demo of object detection via the Python client.

It uses a pre-trained VOC0712 model to detect 21 different classes of objects in images. A bounding box is returned for every detected object, along with its class and a confidence.

To run the code on your own image:

- Install the pre-trained model:

```bash
mkdir model
cd model
wget https://deepdetect.com/models/voc0712_dd.tar.gz
tar xvzf voc0712_dd.tar.gz
cd ..
```

- Start a DeepDetect server:

```bash
./dede
```

- Try object detection on an image

```bash
python objdetect.py --image /path/to/yourimage.jpg --confidence-threshold 0.1
```

- Alternatively, try the chain version. It will draw the bounding boxes on the image and return it.

```bash
python chaindetect.py --image /path/to/yourimage.jpg --save-path image.png
```

Notes:

- The VOC0712 model originates from https://github.com/weiliu89/caffe/tree/ssd and may not be very accurate on standard pictures

- You can predict over batches of images with only slight modifications of the Python code of this demo
