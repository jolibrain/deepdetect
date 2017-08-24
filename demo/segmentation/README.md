### Segmentation demo

This is a small demo of image segmentation via the Python client to DeepDetect server.

It uses a custom pre-trained model that segments 150 different classes of objects in images. A class is predicted for every pixel in the input image.

To run the code on your own image:

- Install the pre-trained model:

```
mkdir model
wget https://deepdetect.com/models/model_deeplab_ade20k.tar.gz
tar xvzf model_deeplab_ade20k.tar.gz
cd ..
```

- Start a DeepDetect server:

```
./dede
```

- Try image segmentation on an image

```
python segment.py --nclasses 150 --model-dir /path/to/model/model_deeplab_ade20k/ --image /path/to/image.jpg
```

