# NCNN CPU inference

DeepDetect runs standard NCNN image models through the C++ server and the embedded C++ runtime. Conversion runs on a separate Linux host. The target needs DeepDetect and its native dependencies. PyTorch, PNNX and Python are only needed for workflows that use them on the host.

For standalone image inference, the [YOLOX-S and SegFormer-B0 C++ examples](../tools/ncnn/README.md) include model export, build and run commands.

## Build on Ubuntu

Use the [source build prerequisites](source.md). The following builds the server and the self-contained NCNN tests on x86-64 or AArch64.

```sh
cmake -S . -B build-ncnn -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CPU_ONLY=ON -DUSE_TORCH=OFF -DUSE_PYTORCH_WORKER=OFF \
  -DUSE_HDF5=OFF -DUSE_NCNN=ON -DBUILD_NCNN_TESTS=ON
cmake --build build-ncnn --target dede ut_ncnnapi --parallel 1
ctest --test-dir build-ncnn -R '^ut_ncnnapi$' --output-on-failure
```

CMake accepts an installed SDK of exactly the pinned version through `ncnn_DIR`. Otherwise it builds the checksummed source release with Vulkan disabled. The `build.sh -a cpu -b ncnn` profile also selects the backend. The existing `armv7` profile remains available. Native platform qualification covers 64-bit ARM. [NCNN 20261020](https://github.com/Tencent/ncnn/releases/tag/20261020)

## Export from PyTorch

Install the DeepDetect CLI and the model's Python dependencies on the conversion host. Put the official Linux x86-64 or AArch64 PNNX executable on `PATH`, or pass its path with `--pnnx`. The exporter verifies the release binary and requests FP32 conversion. [PNNX 20261020](https://github.com/pnnx/pnnx/releases/tag/20261020)

For a PyTorch worker run with standard YOLOX-S weights, use the worker's Python environment with the upstream YOLOX models and this repository's `tools/torch` on its import path. The provider reconstructs YOLOX-S, loads the selected weights strictly, and exports the decoded head before confidence filtering and NMS. Supply the training input dimensions and the DeepDetect class count, including the reserved background. For standard COCO weights the count is 81.

```sh
export PYTHONPATH="$PWD/tools/torch${PYTHONPATH:+:$PYTHONPATH}"
python3 -m deepdetect.cli.main export /path/to/run --format ncnn \
  --checkpoint best --output /path/to/yolox-ncnn \
  --set service_mllib.onnx_export.provider=yolox_ncnn:yolox_s \
  --set service_mllib.ncnn_output=yolox --set service_mllib.threads=1 \
  --set service_input.rgb=false --set service_input.ncnn_letterbox=true
```

The same values can be saved in the run's `export.inference` configuration. Training-time ONNX providers use `mllib.onnx_export.provider`. The exporter can reuse a saved ONNX bundle when its recorded checkpoint names and SHA-256 hashes match the selected checkpoint. A cached graph still needs to meet the requested output contract. A worker with a different architecture needs its own inference provider.

For a standalone TorchScript or ONNX classifier, supply a YAML file with the inference contract. This example matches a model with ImageNet RGB normalization and 1,000 output logits.

```yaml
width: 224
height: 224
nclasses: 1000
service_input:
  rgb: true
  interp: linear
  scale: 0.00392156862745098
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
service_mllib:
  ncnn_output: logits
  threads: 1
predict_output: {best: 5}
```

```sh
python3 -m deepdetect.cli.main export resnet18.ts --format ncnn \
  --config classifier.yaml --output resnet18-ncnn
```

Use `--labels corresp.txt` for an optional text file containing `class_id label` on each line. `--test-inference` exercises a relocated copy using the installed NCNN-enabled DeepDetect Python binding. The published directory contains one native parameter/weight pair, `service.json`, `inference.yaml`, `export.json` with hashes, and any labels supplied.

### RT-DETRv4-S worker checkpoint

The integrated adapter uses the upstream S deployment model at 640×640 with 300 queries and contiguous foreground IDs starting at 1. Set the run's foreground class count plus background in `nclasses`. From the repository root, use the export environment with the worker dependencies and the matching upstream checkout:

```sh
export RTDETRV4_REPO=/path/to/RT-DETRv4
export PYTHONPATH="$PWD/extern/pytorch_workers${PYTHONPATH:+:$PYTHONPATH}"
python3 -m deepdetect.cli.main export /path/to/run --format ncnn --checkpoint best \
  --output /path/to/rtdetrv4-ncnn \
  --set service_mllib.onnx_export.provider=rtdetrv4.worker:ncnn_export \
  --set service_mllib.rtdetrv4.config_path=configs/rtv4/rtv4_hgnetv2_s_coco.yml \
  --set service_mllib.ncnn_output=rtdetrv4 --set service_mllib.threads=1 \
  --set service_input.rgb=true --set service_input.interp=linear \
  --set service_input.scale=0.00392156862745098 --set service_input.ncnn_letterbox=false
```

These preprocessing values match the official checkpoint. Preserve the training contract for fine-tuned weights. For a standalone ONNX graph produced by this provider, use the same service settings in a YAML config and pass the ONNX file and `--config` to the exporter. The output is one NCNN parameter/weight pair. The DeepDetect backend registers the custom CPU layers, runs the encoder and all decoder blocks, and applies the request's confidence threshold and box limit. The detector needs no ONNX Runtime on the target.

The exporter also lowers fixed-K FP32 TopK in other TorchScript and ONNX image models. It supports largest/smallest selection along a physical tensor axis. Consumers of indices need compatible int32 semantics. Dynamic K and unrelated unsupported decoder operations require additional lowering. See the [NCNN export skill](../tools/ncnn/ncnn-export/SKILL.md) for the layer interface and extension instructions.

## Run SegFormer through the service and CLI

Choose the config for the checkpoint. The DeepDetect two-class model takes a 480×480 BGR image, takes the argmax on the output grid, then resizes the class IDs with nearest-neighbor interpolation. The NVIDIA ADE20K model takes a 512×512 RGB image and bilinearly resizes each of its 150 logit planes to the original image size before taking the argmax. Changing the resize order can move class boundaries. Both profiles use fixed-size stretch preprocessing. [Model download and tracing instructions](../tools/ncnn/README.md#segformer-b0-deepdetect-two-class-checkpoint)

```sh
python3 -m deepdetect.cli.main export models/segformer-dd/model.pt --format ncnn \
  --config tools/ncnn/segformer-deepdetect.yaml --output bundles/segformer-dd
python3 -m deepdetect.cli.main export models/segformer-ade/model.pt --format ncnn \
  --config tools/ncnn/segformer-ade20k.yaml --output bundles/segformer-ade

python3 -m deepdetect.cli.main infer ncnn-segmenter photo.jpg \
  --config bundles/segformer-dd/inference.yaml --output results/dd
python3 -m deepdetect.cli.main infer ncnn-segmenter photo.jpg \
  --config bundles/segformer-ade/inference.yaml --output results/ade
```

Each CLI invocation writes a PNG of zero-based class IDs and a color overlay. To use HTTP, set the repository path in the exported `service.json` and send the following requests. Image paths must be readable by the server.

```sh
curl -fsS -X PUT http://localhost:8080/services/segformer \
  -H 'Content-Type: application/json' --data-binary @bundles/segformer-dd/service.json
curl -fsS -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"service":"segformer","data":["/data/photo.jpg"],"parameters":{"output":{}}}'
```

Each prediction returns the source `uri`, its original `imgsize` and `vals`, a row-major array of `width × height` class IDs. Set `parameters.output.confidences` to `["best"]` for the winning softmax probability map. For a thresholded mask, omit `confidences` and pass `confidence_threshold` (CLI `--confidence-threshold`). Pixels below the threshold receive class ID 0. The native service supports the CLI's center-merged tiled inference and optional best-confidence maps. Merge modes that request per-class probability planes require a backend that returns those planes.

## Use a bundle from C++

Set `model.repository` in the generated `service.json` to the bundle directory visible to the server or embedded process. Its default `.` resolves against the process working directory. Pass that JSON to `deepdetect::Runtime::create_service` or the service HTTP endpoint, then call predict with ordinary image paths. For example, with the server running from the bundle directory and the image available to the server:

```sh
curl -fsS -X PUT http://localhost:8080/services/objects \
  -H 'Content-Type: application/json' --data-binary @service.json
curl -fsS -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"service":"objects","parameters":{"output":{"bbox":true,"confidence_threshold":0.25,"nms_threshold":0.45}},"data":["/data/photo.png"]}'
```

The Python CLI can use the same bundle via `infer ncnn-classifier`, `infer ncnn-detector` or `infer ncnn-segmenter` with `--config BUNDLE/inference.yaml`. Use the exported image dimensions. Each extractor handles one image, including in multi-image requests.

| Output contract | Expected native output |
| --- | --- |
| `logits` | One FP32 score per class. DeepDetect applies softmax. |
| `probabilities` | One FP32 probability per class. DeepDetect preserves the scores. |
| `yolox` | Decoded rows of center x, center y, width, height, objectness and foreground scores. Box geometry uses network-input pixels. DeepDetect multiplies objectness by the best class score, applies class-aware NMS and maps boxes back to source pixels. |
| `rtdetrv4` | Ranked class, score and normalized xyxy rows. Boxes are scaled to original image pixels. Multiple classes can share a box. |
| `segmentation` | Unpacked FP32 logits of shape `[nclasses, height, width]`. Set `ncnn_segmentation_resize` to `class_ids` (default) for argmax first, or `logits` for resize first. |
| `ssd` | Standard six-column DetectionOutput rows containing class, score and normalized corners. |

Export targets a fixed spatial shape and one FP32 image input. A graph with multiple outputs needs an inference wrapper that selects the intended tensor. Unsupported operators or a different YOLO head layout require an export adapter. The optional YOLOX letterbox resizes with linear interpolation and pads the bottom and right with 114. Exported normalization is `(pixel * scale - mean) / std` in the configured channel order.

## Existing repositories

Standard matching `.param` and `.bin` files load directly. The backend discovers a unique graph endpoint or accepts explicit `inputBlob` and `outputBlob` service parameters. Old image services retain their normalization convention until `ncnn_normalize` is enabled. Re-export Caffe sources before deployment. Time-series and recurrent graphs using the old DeepDetect NCNN patches require the previous runtime or a new model implementation.

## Qualification of the 20261020 upgrade

The same exported FP32 model files ran through the embedded C++ service on Ubuntu 24.04 x86-64 and on an AArch64 DGX Spark. The reference image was the YOLOX v0.3.0 `assets/dog.jpg`, decoded once and saved as PNG. ResNet-18 used torchvision's ImageNet weights, direct 224×224 resizing and the normalization above. YOLOX-S used the upstream COCO weights, a 640×640 top-left letterbox, a confidence threshold of 0.25 and an NMS threshold of 0.45. The reference was the original PyTorch model before conversion.

| Measurement | Ubuntu x86-64 | Ubuntu AArch64 |
| --- | ---: | ---: |
| Maximum absolute error over 1,000 ResNet-18 logits | 0.0000191 | 0.0000212 |
| Maximum coordinate error over 8,400 decoded YOLOX rows, input pixels | 0.00916 | 0.00925 |
| Maximum objectness or class score error across those rows | 0.00000880 | 0.0000197 |
| Maximum returned box-corner error, original image pixels | 0.000092 | 0.000245 |
| ResNet-18 median warm latency / peak process RSS | 52 ms / 265 MiB | 21 ms / 258 MiB |
| YOLOX-S median warm latency / peak process RSS | 419 ms / 239 MiB | 170 ms / 274 MiB |

Both platforms reproduced the ordered top five ImageNet classes and all five detections. Latency covers image decode, preprocessing, C++ inference and JSON output for one local file. Each model ran in its own process with one NCNN thread, one warmup and five measured requests. The x86-64 run used the upstream shared SDK in a two-CPU container. AArch64 used a source-built static NCNN in a one-CPU cgroup. This is a portability and numerical parity check on one image. Dataset-level classification accuracy and detection mAP require separate evaluation.
