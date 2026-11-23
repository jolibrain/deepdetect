# YOLOX and SegFormer inference in C++

These examples run a single image on the CPU with NCNN and OpenCV. YOLOX-S prints bounding boxes. SegFormer-B0 writes a PNG of class IDs at the original image size. Both use one inference thread and FP32. The target needs the native libraries and the converted model pair. Python and PNNX are used on the export host.

## Build

Run the commands from the DeepDetect repository root. The standalone build uses an installed NCNN 20261020 SDK, OpenCV 4 development files, CMake and a C++17 compiler.

```sh
cmake -S tools/ncnn -B build/ncnn-examples -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR=/path/to/ncnn/install/lib/cmake/ncnn
cmake --build build/ncnn-examples --parallel 1
bin=build/ncnn-examples
```

If you already have a DeepDetect build configured with `USE_NCNN=ON`, you can use its NCNN dependency directly.

```sh
cmake -S . -B build-ncnn -DBUILD_TOOLS=ON
cmake --build build-ncnn --target ncnn_yolox ncnn_segformer --parallel 1
bin=build-ncnn/tools/ncnn
```

The same commands apply on Ubuntu x86-64 and ARM64. Build the binaries for the target architecture. Copy the same `model.ncnn.param` and `model.ncnn.bin` to either target. Use a separate model directory for each export. For a fresh DeepDetect build, see the [NCNN build guide](../../docs/ncnn.md#build-on-ubuntu).

## Run an existing model

Set `image` to a local image path. Each model directory below contains the corresponding pair of files named `model.ncnn.param` and `model.ncnn.bin`.

```sh
image=/path/to/photo.jpg
"$bin/ncnn_yolox" models/yolox "$image"
"$bin/ncnn_segformer" deepdetect models/segformer-dd "$image" mask-dd.png
"$bin/ncnn_segformer" ade20k models/segformer-ade "$image" mask-ade.png
```

**YOLOX** prints one whitespace-separated row per detection in the order `class_id score xmin ymin xmax ymax`. Rows are sorted by decreasing score. Foreground IDs start at 1. Coordinates are in original-image pixels. No rows means no detection passed the threshold. The example expects the decoded YOLOX-S head at 640×640. It uses a top-left BGR letterbox padded with 114, multiplies objectness by the best foreground score, filters at 0.25 and applies class-aware NMS at IoU 0.45. [YOLOX C++ code](yolox.cc)

**SegFormer** writes an 8-bit grayscale PNG where each byte is a class ID. The two-class mask contains values 0 and 1, so it appears nearly black in a normal image viewer. Select the profile matching the checkpoint. [SegFormer C++ code](segformer.cc)

| Profile | Input | How the mask is made |
| --- | --- | --- |
| `deepdetect` | Two-class DeepDetect B0, 480×480 BGR, cubic resize, scale 0.0039 | Argmax over the two logits, then resize class IDs with nearest-neighbor interpolation. |
| `ade20k` | NVIDIA ADE20K B0, 512×512 RGB, linear resize, scale 1/255 | Bilinearly resize each of the 150 logit planes to the original image dimensions, then take argmax. |

Both SegFormer profiles subtract ImageNet means and divide by ImageNet standard deviations after scaling. The ADE20K runner keeps the current best score and class for each pixel as it processes each plane. These examples run standalone. For YOLOX inference via the DeepDetect service or CLI, use a DeepDetect-exported detector bundle as described in the [service guide](../../docs/ncnn.md#use-a-bundle-from-c).

## Export models

The recipes below produce the files expected by the run commands. Put the official Linux `pnnx` executable on your export host's `PATH`. Use fresh output directories. [PNNX 20261020 release](https://github.com/pnnx/pnnx/releases/tag/20261020)

### YOLOX-S

An existing `deepdetect export RUN --format ncnn` bundle with the decoded YOLOX head and 640×640 input can be passed directly to `ncnn_yolox`. To start with the official COCO checkpoint instead, use a Python environment with PyTorch, torchvision, ONNX and the YOLOX v0.3.0 model dependencies. Install the upstream `yolox` package in that environment, or add its checkout root to `PYTHONPATH`. [DeepDetect worker export instructions](../../docs/ncnn.md#export-from-pytorch) · [Upstream YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0)

```sh
mkdir -p models/yolox
curl -fL https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth \
  -o models/yolox/yolox_s.pth
python3 tools/ncnn/export_yolox.py \
  models/yolox/yolox_s.pth models/yolox/model.onnx
(cd models/yolox && pnnx model.onnx 'inputshape=[1,3,640,640]' fp16=0 device=cpu)
```

For a compatible custom YOLOX-S checkpoint, pass `--classes N` to the export script, where `N` counts foreground classes. The provider exports 8,400 rows of center coordinates, box dimensions, objectness and class probabilities. [Export script](export_yolox.py) · [Decoded-head provider](../torch/yolox_ncnn.py)

### SegFormer-B0, DeepDetect two-class checkpoint

The public checkpoint is already TorchScript. PNNX can convert it directly.

```sh
mkdir -p models/segformer-dd
curl -fL https://www.deepdetect.com/downloads/platform/pretrained/torch/segformer/segformer-b0-cls2.pt \
  -o models/segformer-dd/model.pt
(cd models/segformer-dd && pnnx model.pt 'inputshape=[1,3,480,480]' fp16=0 device=cpu)
```

### SegFormer-B0, NVIDIA ADE20K checkpoint

Use an export environment with PyTorch and Hugging Face Transformers. The script loads the pretrained model in evaluation mode and traces its logits output at 512×512. Use `--model /path/to/checkpoint-directory` for a local copy. [Trace script](trace_segformer.py) · [NVIDIA checkpoint](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)

```sh
mkdir -p models/segformer-ade
python3 tools/ncnn/trace_segformer.py models/segformer-ade/model.pt
(cd models/segformer-ade && pnnx model.pt 'inputshape=[1,3,512,512]' fp16=0 device=cpu)
```
