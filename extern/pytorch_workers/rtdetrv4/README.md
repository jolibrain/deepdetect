# RT-DETRv4 External PyTorch Worker

This adapter lets DeepDetect use an external RT-DETRv4 checkout through the
generic `external-pytorch-detector` CLI profile. The adapter code is local to
this directory; the upstream RT-DETRv4 model code is not vendored here.

## Files

- `worker.py`: DeepDetect worker adapter.
- `config.yaml`: default-style CLI config for training and inference.
- `manifest.json`: adapter metadata and upstream requirements.

## Prerequisites

Prepare:

- a DeepDetect Python environment with the PyTorch worker backend available;
- an upstream RT-DETRv4 checkout;
- an RT-DETRv4 config file from that checkout;
- detection list files in DeepDetect format, with image paths and bbox paths;
- optionally, a pretrained RT-DETRv4 checkpoint.

The worker expects the upstream checkout through either
`service_mllib.rtdetrv4.repo_path` or `RTDETRV4_REPO`. It expects the upstream
model config through `service_mllib.rtdetrv4.config_path`.

## Train

From the repository root, run the source CLI:

```shell
PYTHONPATH=bindings/python python3 -m deepdetect.cli.main train external-pytorch-detector \
  --config extern/pytorch_workers/rtdetrv4/config.yaml \
  --train-data /path/to/train.txt \
  --test-data /path/to/test.txt \
  --repository runs/rtdetrv4 \
  --nclasses 2 \
  --iterations 1000 \
  --test-interval 100 \
  --batch-size 2 \
  --gpu --gpuid 0 \
  --terminal verbose \
  --output-format jsonl \
  --set service_mllib.rtdetrv4.repo_path=/path/to/rtdetrv4 \
  --set service_mllib.rtdetrv4.config_path=/path/to/rtdetrv4/config.yaml \
  --set service_mllib.rtdetrv4.pretrained_model=/path/to/checkpoint.pth
```

For CPU smoke tests, replace `--gpu --gpuid 0` with `--no-gpu`. For an
installed wheel, replace `PYTHONPATH=bindings/python python3 -m
deepdetect.cli.main` with `deepdetect`.

The config uses `mllib.data_source: connector_tensor_pull`, so image loading,
basic preprocessing, bbox tensor packing, and configured augmentation are
handled by DeepDetect before batches reach `worker.py`.

## Monitor

Use JSONL stdout events for automation. The repository also contains the latest
run state and worker artifacts:

```shell
PYTHONPATH=bindings/python python3 -m deepdetect.cli.main job status runs/rtdetrv4 \
  --output-format json
```

Useful files include:

- `runs/rtdetrv4/config.yaml`: effective CLI config;
- `runs/rtdetrv4/run.json`: latest run manifest and status;
- `runs/rtdetrv4/metrics.jsonl`: persisted metric stream;
- `runs/rtdetrv4/pytorch_worker_config.json`: worker-side effective config;
- `runs/rtdetrv4/connector_manifest.json`: connector tensor-pull manifest.

## Inference

Inference uses the same external worker entrypoint and loads the trained
checkpoint from the model repository:

```shell
PYTHONPATH=bindings/python python3 -m deepdetect.cli.main infer external-pytorch-detector \
  /path/to/image.jpg \
  --config extern/pytorch_workers/rtdetrv4/config.yaml \
  --repository runs/rtdetrv4 \
  --service-name python-rtdetrv4-infer \
  --nclasses 2 \
  --gpu --gpuid 0 \
  --confidence-threshold 0.25 \
  --visualize \
  --output runs/rtdetrv4-predictions \
  --set service_mllib.rtdetrv4.repo_path=/path/to/rtdetrv4 \
  --set service_mllib.rtdetrv4.config_path=/path/to/rtdetrv4/config.yaml
```

Add more image paths after the first image to run batched inference. Use
`--batch-size N` to control DeepDetect predict batch size.

## Notes

- DeepDetect class id `0` is background. The adapter converts foreground labels
  to zero-based RT-DETRv4 labels during training and converts predictions back
  to DeepDetect one-based foreground labels.
- The adapter patches common RT-DETRv4 config fields such as class count,
  teacher/distillation settings, and pretrained backbone behavior.
- If upstream imports fail, verify `service_mllib.rtdetrv4.repo_path`,
  `RTDETRV4_REPO`, and the upstream Python package dependencies.
