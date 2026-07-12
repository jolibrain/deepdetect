# ViTPose PyTorch Worker

This is a self-contained DeepDetect PyTorch worker for ViTPose-style keypoint
training. It lives under `extern/pytorch_workers` because PyTorch workers are
loaded through entrypoint files, but it does not require an externally cloned
ViTPose repository at runtime.

The default `topdown` head consumes one DeepDetect-generated affine crop per
annotated object and predicts one heatmap per keypoint. The optional `slots`
head keeps the full-image fixed-slot model with Hungarian assignment.

Top-down target rows use DeepDetect bbox order:

```text
cls xmin ymin xmax ymax x1 y1 x2 y2 ...
```

Set both `service_mllib.vitpose.head` and `mllib.vitpose.head` to `slots` to use
legacy rows containing keypoint coordinates only. Missing keypoints are
represented as `-1 -1` in both modes.

## Train

```shell
PYTHONPATH=bindings/python python3 -m deepdetect.cli.main train vitpose \
  --config extern/pytorch_workers/vitpose/config.yaml \
  --train-data /path/to/train.txt \
  --test-data /path/to/test.txt \
  --repository runs/vitpose \
  --iterations 1000 \
  --test-interval 100 \
  --batch-size 2 \
  --gpu --gpuid 0 \
  --terminal verbose \
  --output-format jsonl
```

`mllib.vitpose.bbox_scale_factor` controls how much context is retained around
each top-down box and defaults to `1.25`. Set `mllib.vitpose.max_objects` and
`service_mllib.vitpose.max_objects` to the same value for slot models.

## Evaluation Metrics

Each completed test set reports `loss_testN`, `heatmap_loss_testN`,
`mean_keypoint_error_px_testN`, `visible_keypoints_testN`, and
`pose_samples_testN`, where `N` is the test set index. Slot models also report
`objectness_loss_testN`. Losses are reduced over the whole test set, with
heatmap MSE weighted by the count of visible keypoints. Mean keypoint error is
the Euclidean distance between raw heatmap peaks and visible targets in source
image pixels; slot predictions use their Hungarian loss assignment.

## Photometric Augmentation

Photometric augmentation is supported for `connector_tensor_pull` training
batches only. It leaves keypoint targets and top-down crop metadata unchanged;
test batches are never augmented.

```yaml
augmentation:
  noise:
    prob: 0.01
  distort:
    prob: 0.01
```

These options retain the shared C++ augmentation behavior. `noise.prob` is
sampled independently for each legacy noise effect. Any nonzero `distort.prob`
enables its complete legacy distortion chain for each training crop.

## MAE Backbone Initialization

`--weights` accepts a regular DeepDetect/ViTPose checkpoint or a MAE-style ViT
checkpoint. Bare MAE weights are detected automatically and initialize only
the ViT backbone; the pose heatmap head remains newly initialized.

```shell
PYTHONPATH=bindings/python python3 -m deepdetect.cli.main train vitpose \
  --config extern/pytorch_workers/vitpose/config.yaml \
  --weights /data1/beniz/models/mae/mae_pretrain_vit_base.pth \
  --train-data /path/to/train.txt \
  --repository runs/vitpose-mae
```

## Inference

```shell
PYTHONPATH=bindings/python python3 -m deepdetect.cli.main infer vitpose \
  /path/to/image.jpg \
  --bbox-files /path/to/image-bboxes.txt \
  --config extern/pytorch_workers/vitpose/config.yaml \
  --repository runs/vitpose \
  --gpu --gpuid 0 \
  --confidence-threshold 0.25
```

Top-down bbox sidecars contain one `cls xmin ymin xmax ymax` row per object and
are required one-for-one with input images. Predictions are returned as
`classes[]`; top-down entries contain the source bbox, class id, and keypoints.
Slot entries contain the retained objectness score and keypoints.
