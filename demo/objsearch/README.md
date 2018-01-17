### Object similarity search demo

This is a small Python demo that uses DeepDetect for detecting and getting a fingerprint of objects in an image, then building an in-image object search application.

It does four things:

- use a DeepDetect object detection service to find object bounding boxes in an image
- get an embedding (i.e. vector code) for every detected object in the image
- indexes all objects with [annoy](https://github.com/spotify/annoy), an approximate nearest neighbors C++/Python library
- search by objects within an image, and returns closest objects and images in the index

To run the code on your own collection of images:

- install Annoy:
  ```
  pip install annoy
  ```
  or go look at https://github.com/spotify/annoy

- Install the pre-trained model:

```
mkdir model
cd model
wget https://deepdetect.com/models/voc0712_dd.tar.gz
tar xvzf voc0712_dd.tar.gz
cp VGG_VOC0712_SSD_300x300_iter_60000.caffemodel model
cp corresp.txt model
cd ..
```
  **make sure that the `model` repository is in the same repository as the script `objsearch.py`**

- start a DeepDetect server:
  ```
  ./dede
  ```

- index your collection of images:
  ```
  python objsearch.py --index /path/to/your_images_repo
  ```

- search for similar images:
  ```
  python objsearch.py --search /path/to/your_image
  ```

Notes : in 2017_12_14 deploy, roi_pool layer is fed only with conv4_3 (not relued!).
