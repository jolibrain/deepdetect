# Pytorch tools

This directory contains a set of tools to help when working with libtorch backend.

* `trace_torchvision.py`

Utility script to trace the models included in torchvision. Requires torchvision to be installed:
```
pip3 install --user torchvision
```

* `trace_pytorch_transformers.py`

Utility script to trace NLP models from Huggingface's pytorch-transformers. Requires pytorch-transformers 1.1:
```
pip3 install --user pytorch-transformers==1.1
```
At the moment a CUDA model can not be convertible to CPU and vice versa. This may change with a future version of pytorch-transformers.
