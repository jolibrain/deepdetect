### Clustering with T-SNE demo

This is a small demo of clustering the MNIST dataset test set via the Python client.

To run the code:
- Start a DeepDetect server:

```
./dede
```

- Try the clustering:
```
python demo_tsne.py
```

The script downloads the dataset, and that can take a few seconds, then starts the clustering. Expect around a minute total before the scatter plot appears on screen.