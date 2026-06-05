# Embedded Python examples

These scripts use the in-process `deepdetect` package, not the HTTP
`dd_client`. Build and install the Python wheel as described in
`docs/python_library_spec.md`, then configure `LD_LIBRARY_PATH` for the
DeepDetect SDK and its dependencies.

The examples use two-class TorchScript models by default:

```text
/data1/beniz/models/dd/yolox/yolox-nano_cls2.pt
/data1/beniz/models/dd/segformer/segformer-b0-cls2.pt
```

Override either path with `--weights`. Each script copies its initial model
into a writable `--repository`. Existing `.pt` files in that repository are
kept so inference can use a checkpoint produced by training.

CPU is the default. Pass `--gpu` when the Python package is linked against a
CUDA-enabled DeepDetect SDK.

For SDKs reporting CUDA 13 with cuDNN 8, the YOLOX training script
and SegFormer training script automatically set
`TORCH_CUDNN_V8_API_DISABLED=1`. This avoids cuDNN
execution-plan failures such as `GET was unable to find an engine`. Such SDKs
still require the cuDNN 8 CUDA 12 cuBLAS dependency in `LD_LIBRARY_PATH`.

## Dataset lists

YOLOX train and test lists contain an image path and an annotation path:

```text
/data/images/0001.jpg /data/annotations/0001.txt
/data/images/0002.jpg /data/annotations/0002.txt
```

Each annotation line is `class xmin ymin xmax ymax`. DeepDetect reserves class
`0` for background or negative samples, so foreground boxes in these
two-class examples use class `1`:

```text
1 42 35 180 210
```

SegFormer lists contain an image path and a single-channel class-index mask:

```text
/data/images/0001.jpg /data/masks/0001.png
/data/images/0002.jpg /data/masks/0002.png
```

Mask pixels must contain values in `0..nclasses-1`. `train_segformer.py`
requires `--nclasses` and validates the masks before loading the model. The
TorchScript model's segmentation head must have the same output-channel count.

## Training

```bash
python examples/python/train_yolox.py \
  --train-data /data/detection/train.txt \
  --test-data /data/detection/test.txt \
  --iterations 100 \
  --gpu

python examples/python/train_segformer.py \
  --train-data /data/segmentation/train.txt \
  --test-data /data/segmentation/test.txt \
  --nclasses 2 \
  --iterations 100 \
  --gpu
```

Training blocks and prints the final response by default. Add `--async` to
print job status while polling. `--timeout` cancels an asynchronous job that
does not finish in time, and Ctrl-C also requests cancellation.

## Inference

Use the training repository to run a newly trained checkpoint:

```bash
python examples/python/predict_yolox.py image.jpg \
  --repository deepdetect-models/yolox-train \
  --confidence-threshold 0.25 \
  --output detections.png \
  --gpu

python examples/python/predict_segformer.py image.jpg \
  --repository deepdetect-models/segformer-train \
  --output segmentation-overlay.png \
  --gpu
```

Both scripts print the complete normalized JSON payload. YOLOX optionally
draws bounding boxes. SegFormer writes an overlay and a sibling
`*_mask.png`. For multiple input images, `--output` is treated as a directory.
