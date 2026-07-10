# ViTPose PyTorch Worker

This is a self-contained DeepDetect PyTorch worker for ViTPose-style keypoint
training. It lives under `extern/pytorch_workers` because PyTorch workers are
loaded through entrypoint files, but it does not require an externally cloned
ViTPose repository at runtime.

The worker consumes DeepDetect keypoint tensor batches, predicts a fixed number
of object slots, and trains with Hungarian assignment between slots and
ground-truth objects.

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

Set `mllib.vitpose.max_objects` and `service_mllib.vitpose.max_objects` to the
same value when training multi-object slot models.

## Inference

```shell
PYTHONPATH=bindings/python python3 -m deepdetect.cli.main infer vitpose \
  /path/to/image.jpg \
  --config extern/pytorch_workers/vitpose/config.yaml \
  --repository runs/vitpose \
  --gpu --gpuid 0 \
  --confidence-threshold 0.25
```

Predictions are returned as `classes[]`, one entry per kept object slot. Each
class entry contains `prob`, `cat`, and `keypoints[]`.
