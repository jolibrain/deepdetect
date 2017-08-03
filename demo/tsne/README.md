### Clustering with T-SNE demo

This is a small demo of clustering the MNIST dataset test set via the Python client. The dataset contains 10000 images in CSV format. The final clustering thus has 10000 points.

To run the code:
- Start a DeepDetect server:

```
./dede
```

- Try the clustering:
```
python demo_tsne.py
```

The script downloads the dataset, and that can take a few seconds, then starts the clustering. Expect around a minute total before the scatter plot appears on screen:

![alt tag](https://deepdetect.com/dd/examples/tsne_mnist_test.png)
