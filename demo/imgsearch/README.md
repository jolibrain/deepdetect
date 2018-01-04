### Image similary search demo

This is a small Python demo of an image similarity search application.

It does two things:

- use a DeepDetect image classification service in order to generate a numerical or binary code for every image
- indexes images with [annoy](https://github.com/spotify/annoy), an approximate nearest neighbors C++/Python library
- search by images, even for new images, not previously indexed, and return the closest images

To run the code on your own collection of images:

- install Annoy:
  ```
  pip install annoy
  ```
  or go look at https://github.com/spotify/annoy

- create a model repository with the pre-trained image classification network of your choice. Here we are using a pre-trained GoogleNet, but you can also use a built-in ResNet or [other provided models](http://www.deepdetect.com/applications/model/):
  ```
  mkdir model
  cd model
  wget http://www.deepdetect.com/models/ggnet/bvlc_googlenet.caffemodel
  ```
  
  **make sure that the `model` repository is in the same repository as the script `imgsearch.py`**

- start a DeepDetect server:
  ```
  ./dede
  ```

- index your collection of images:
  ```
  python imgsearch.py --index /path/to/your/images --index-batch-size 64
  ```
  Here `index-batch-size` controls the number of images that are processed at once.
  The index file is then `index.ann` in the repository. `names.bin` indexes the filenames.
  
  **Index and name files are erased upon every new indexing call**

- search for similar images:
  ```
  python imgsearch.py --search /path/your/image.png --search-size 10
  ```
  Here `search-size` controls the number of approximate neighbors.


### Running with a dockerised image

- Start your docker container instance as per https://github.com/beniz/deepdetect/tree/master/docker but mount your images to a volume

  `docker run -d -p 8080:8080 -v /full/path/to/your/images:/images beniz/deepdetect_cpu`
 
  There is an additional `--model_repo` argument to specify the path to the model, this is relative to the volume mount in the docker container.
  `--model_repo /model` with the additional argument on `docker run` of `-v /path/to/your/model:/model`


Notes:

- The search uses a deep convolutional net layer as a code for every image. Using top layers (e.g. `loss3/classifier` with GoogleNet) uses high level features and thus image similarity is based on high level concepts such as whether the image contains a lakeshore, a bottle, etc... Using bottom or mid-range layers (e.g. `pool5/7x7_s1` with GoogleNet) makes image similarity based on lower level, potentially invariant, universal features such as lightning conditions, basic shapes, etc... Experiment and see what is best for your application.

- Annoy is a nice piece of code but in experiments the index building step becomes very memory inefficient and time-consuming around a million of images. If this is an issue, get in touch, as they are other, more complicated, ways to index and perform the search and scale.

- The code in `imgsearch.py` allows for more options such as whether to use `binarized` codes, `angular` or `euclidean` metric for similar image retrieval, and control of the accuracy of the search through `ntrees`.
