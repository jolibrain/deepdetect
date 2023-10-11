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

* `trace_yolox.py`

Requirements:

```
pip3 install thop
pip3 install loguru
```
and custom YOLOX repository:
```
git clone https://github.com/jolibrain/YOLOX.git
git checkout jit_script
```

Then to export a pretrained model before using it with DeepDetect:

```
python3 trace_yolox.py yolox-m -o /path/to/output/ --yolox_path /path/to/YOLOX/ --backbone_weights /path/to/pretrained/yolox_m.pth --num_classes 2 --img_width 300 --img_height 300
```

## Export segformer model

First install our version of mmsegmentation with a modified export script
```bash
git clone https://github.com/beniz/mmsegmentation.git
git checkout feat_add_num_classes_export_control
CUDA_HOME=/usr/local/cuda11.7 pip install "mmcv-full==1.5.0"
pip install -e -V .
```

Then run the script:
```bash
cd tools
python3 pytorch2torchscript.py \
    ../configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py \
    --checkpoint /opt/platform/models/pretrained/segformer/orig/segformer_mit-b0_512x512_160k_ade20k.pth \
    --output-file /data1/louisj/models/pretrained/segformer/segformer_b0_512_cls3.pt \
    --num_classes 3 --show
```

Checkpoints can be downloaded [here](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer)

To export only the backbone, use `--only_backbone`. Then you do not need to specify the number of classes.
