# RT-DETRv4 External PyTorch Worker

This adapter lets DeepDetect use an external RT-DETRv4 checkout through the
generic `external-pytorch-detector` CLI profile. The adapter code is local to
this directory; the upstream RT-DETRv4 model code is not vendored here.

## Files

- `worker.py`: DeepDetect worker adapter.
- `config.yaml`: default-style CLI config for training and inference.
- `finetune-m.yaml`, `finetune-l.yaml`: full-detector fine-tuning recipes.
- `manifest.json`: adapter metadata and upstream requirements.

## Prerequisites

Prepare:

- a DeepDetect Python environment with the PyTorch worker backend available;
- an upstream RT-DETRv4 checkout;
- an RT-DETRv4 config file from that checkout;
- the official full [M](https://drive.google.com/file/d/1O-YpP4X-quuOXbi96y2TKkztbjroP5mX)
  or [L](https://drive.google.com/file/d/1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8)
  detector checkpoint, downloaded from the [upstream release](https://github.com/RT-DETRs/RT-DETRv4);
  use a trusted local checkpoint because PyTorch deserializes it;
- DeepDetect detection lists with image and bbox paths (not keypoint or
  image-only lists).

The worker expects the upstream checkout through either
`service_mllib.rtdetrv4.repo_path` or `RTDETRV4_REPO`. It expects the upstream
model config through `service_mllib.rtdetrv4.config_path`.

## Train

From the repository root, activate an environment with a built DeepDetect
wheel, choose the M or L recipe, then override its example paths. The source
package under `bindings/python` has no compiled `_native` extension; do not
put it on `PYTHONPATH` when launching training. `--weights` must point to the
full detector checkpoint of the matching flavor. `--nclasses` counts
background as class 0, so a two-class foreground dataset uses `--nclasses 3`.

```shell
source ~/venv/bin/activate
env -u PYTHONPATH deepdetect train external-pytorch-detector \
  --config extern/pytorch_workers/rtdetrv4/finetune-m.yaml \
  --train-data /path/to/train.txt \
  --test-data /path/to/val.txt \
  --weights /path/to/RTv4-M-hgnet.pth \
  --repository runs/rtv4-m \
  --nclasses 3 \
  --gpu --gpuid 0 \
  --terminal verbose \
  --output-format jsonl
```

For L, use `finetune-l.yaml`, the official L checkpoint, and a separate
repository. Set `service_mllib.rtdetrv4.repo_path` if the upstream checkout
is elsewhere. For CPU smoke tests, replace `--gpu --gpuid 0` with `--no-gpu`.
The wheel supplies the native runtime; the external worker is loaded from
`extern/pytorch_workers/rtdetrv4/worker.py` in this source checkout.

The worker loads every backbone, encoder, and decoder tensor from the selected
checkpoint. It reinitializes only the classification tensors whose dimensions
change with `--nclasses`; incompatible or backbone-only checkpoints fail.
The recipes use AdamW with lower backbone LR and no weight decay on bias or
normalization parameters. The DeepDetect worker runs a constant LR and its
standard augmentation; this is detector fine-tuning, not a reproduction of
upstream's teacher/distillation training schedule.

The config uses `mllib.data_source: connector_tensor_pull`, so image loading,
basic preprocessing, bbox tensor packing, and configured augmentation are
handled by DeepDetect before batches reach `worker.py`.

The default `config.yaml`, including when used with RT-DETRv4-S (HGNetV2-B0),
enables mirroring, rotation, cutout, perspective, translation, zoom, noise,
and color distortion. The M/L fine-tuning recipes use the same augmentation
settings. Cropping is disabled to preserve the configured input dimensions;
rotation requires square inputs. Augmentation applies only to training.

To continue a run, set `--iterations` to the **new total optimizer step**.
For example, a run stopped at step 1000 resumes at step 1001 and adds 1000
steps with `--iterations 2000`. The model and optimizer state come from the
latest complete checkpoint pair in the same repository. The saved `config.yaml`
can be reused; turn off its original `repository_override` value so the run is
preserved:

```shell
env -u PYTHONPATH deepdetect train external-pytorch-detector \
  --config runs/rtv4-m/config.yaml \
  --resume latest --iterations 2000 \
  --set repository_override=false
```

Do not pass `--repository-override` on resume. Numbered checkpoints and new
metric events keep their global step numbers.

## Monitor

Use JSONL stdout events for automation. The repository also contains the latest
run state and worker artifacts:

```shell
env -u PYTHONPATH deepdetect job status runs/rtv4-m \
  --output-format json
```

Useful files include:

- `runs/rtv4-m/config.yaml`: effective CLI config;
- `runs/rtv4-m/run.json`: latest run manifest and status;
- `runs/rtv4-m/metrics.jsonl`: persisted metric stream;
- `runs/rtv4-m/pytorch_worker_config.json`: worker-side effective config;
- `runs/rtv4-m/connector_manifest.json`: connector tensor-pull manifest.

## Inference

Inference uses the same external worker entrypoint and loads the trained
checkpoint from the model repository:

```shell
env -u PYTHONPATH deepdetect infer external-pytorch-detector \
  /path/to/image.jpg \
  --config extern/pytorch_workers/rtdetrv4/finetune-m.yaml \
  --repository runs/rtv4-m \
  --service-name python-rtdetrv4-infer \
  --nclasses 3 \
  --gpu --gpuid 0 \
  --confidence-threshold 0.25 \
  --visualize \
  --output runs/rtv4-m-predictions
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
