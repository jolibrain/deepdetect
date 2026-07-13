# DeepDetect Python CLI Specification

## Summary

The Python CLI provides ready-to-use image training and inference workflows on
top of the in-process `deepdetect` bindings. Version 1 is intentionally scoped
to focused model profiles:

- `yolox`: object detection with bounding boxes.
- `segformer`: semantic segmentation.
- `torchvision-detector`: managed PyTorch Faster R-CNN detection.
- `external-pytorch-detector`: an external Python worker selected through
  YAML/API `mllib.entrypoint`.
- `vitpose`: a self-contained PyTorch keypoint worker selected through an
  in-tree `extern/pytorch_workers/vitpose/worker.py` entrypoint. It uses
  bbox-driven top-down pose by default; `vitpose.head: slots` retains the
  full-image Hungarian slot model.
- `sam2`: an inference-only external PyTorch worker for SAM2 static-image
  automatic masks or bbox-prompted masks. Its `sam2==1.1.0` runtime and
  checkpoint are installed/provided separately.

The canonical command shape is task first:

```shell
deepdetect train yolox ...
deepdetect train segformer ...
deepdetect train torchvision-detector ...
deepdetect train external-pytorch-detector ...
deepdetect train vitpose ...
deepdetect infer yolox ...
deepdetect infer segformer ...
deepdetect infer vitpose ...
deepdetect job status RUN_DIR
deepdetect inspect models
```

All commands are discoverable through `--help`. Long-running training emits
structured events that are easy for humans to watch and easy for agents to
parse. YAML config files are supported, and explicit CLI flags override config
values.

Inference accepts positional image paths or `--images-file`, a text file with
one image path on each non-empty line. Relative entries resolve from the list
file directory. Top-down ViTPose and SAM2 also accept `--bbox-files-file` for
an aligned list of sidecar paths; it cannot be combined with positional images
or `--bbox-files`, respectively.

## Configuration

Each train and inference command accepts:

```shell
--config CONFIG.yaml
--set key=value
--output-format jsonl|json|text
```

Configuration precedence is:

1. Model profile defaults.
2. YAML config file.
3. CLI flags.
4. `--set` overrides.

YAML is intended for repeatable runs. CLI flags are intended for discoverable
interactive use and agent function-call arguments.
Config files and `--set` map to top-level CLI option names such as
`base_lr=0.001`, `iter_size=2`, `batch_size=4`, or `visdom=true`. Training
YAML files may also define an `augmentation` mapping. This mapping is
deep-merged into the selected model profile's training augmentation defaults
without adding one command-line flag per augmentation parameter.
Class weighting is available as a YAML-only top-level `class_weights` list.

Example default-style configs are provided next to this document:

- `yolox-default.yaml`
- `segformer-default.yaml`
- `torchvision-detector-default.yaml`
- `external-pytorch-detector-default.yaml`
- `vitpose-default.yaml`
- `sam2-default.yaml`

The trainable model configs include both training and inference keys; the SAM2
config is inference-only. Replace the dataset and model paths before use:

```shell
deepdetect train yolox --config bindings/python/deepdetect/cli/yolox-default.yaml
deepdetect infer yolox image.jpg --config bindings/python/deepdetect/cli/yolox-default.yaml
deepdetect train torchvision-detector --config bindings/python/deepdetect/cli/torchvision-detector-default.yaml
deepdetect train external-pytorch-detector --config bindings/python/deepdetect/cli/external-pytorch-detector-default.yaml
deepdetect train vitpose --config bindings/python/deepdetect/cli/vitpose-default.yaml
deepdetect infer sam2 image.jpg --config bindings/python/deepdetect/cli/sam2-default.yaml
```

`external-pytorch-detector` uses the `pytorch` backend and normal DeepDetect
training flags. The YAML `service_mllib` section must select the worker file
and class, for example:

```yaml
service_mllib:
  entrypoint: extern/pytorch_workers/model_slug/worker.py
  class: DeepDetectWorker
mllib:
  data_source: connector_tensor_pull
```

The worker file may live outside the packaged `deepdetect` module. Generated
adapters should live under `extern/pytorch_workers/<model_slug>/`, which is
ignored by git except for workspace documentation.

## Training Commands

Required inputs may come from CLI flags or config:

```shell
deepdetect train yolox \
  --train-data train.txt \
  --test-data test0.txt test1.txt \
  --weights yolox-nano_cls2.pt \
  --repository runs/yolox-model \
  --nclasses 2

deepdetect train segformer \
  --train-data train.txt \
  --test-data test0.txt test1.txt \
  --weights segformer-b0-cls2.pt \
  --repository runs/segformer-model \
  --nclasses 13
```

Shared training options:

- `--test-data PATH [PATH ...]`: one or more test dataset lists. The first
  test set is reported as `test0`, the second as `test1`, and so on when the
  backend emits per-test-set metrics.
- `--gpu` / `--no-gpu`: request or disable GPU execution.
- `--gpuid ID [ID ...]`: select one or more GPU ids. Comma-separated values
  are also accepted, for example `--gpuid 0,1`. Use `--gpuid -1` for all
  GPUs. Supplying `--gpuid` implies `--gpu` unless `--no-gpu` is explicitly
  passed, which is rejected.
- `--width`, `--height`: image input size. Defaults stay profile-specific:
  YOLOX uses `640x640`, SegFormer uses `480x480`.
- `--run-name`: name for the training run. Defaults strictly to the directory
  name from `--repository`. No date, model prefix, or random suffix is added
  unless an explicit `--run-name` is provided.
- `--resume latest|best`: resume training from `--repository` instead of
  staging `--weights`. `latest` uses the newest checkpoint and solver state
  found in the repository. `best` reads `best_model.txt` and resumes the
  matching `checkpoint-N.*` and `solver-N.pt` files.
- `--repository-override`: remove all existing contents of `--repository`
  before starting a new run. The flag is explicit confirmation and writes a
  warning to stderr. It cannot be combined with `--resume`.
- `--iterations`, `--batch-size`, `--iter-size`, `--base-lr`,
  `--test-interval`: common solver controls. `--iter-size` controls gradient
  accumulation before each optimizer update.
- `--service-name`: DeepDetect service name.
- `--job-dir`: directory for CLI run manifests. Defaults to `--repository`.
  When omitted, `run.json` and `metrics.jsonl` are written directly in the
  model repository. When explicitly set, the run files are written under
  `<job-dir>/<run-name>/`.
- `--poll-interval`, `--timeout`: async monitoring controls.
- `--terminal verbose|live`: choose stdout behavior for async training.
  `verbose` is the default and preserves `--output-format`. `live` uses a
  fixed-size Rich display when stdout is a TTY and falls back to JSONL when it
  is not.
- `--sync` / `--no-sync`: use blocking DeepDetect training or the default
  async job.
- `--dataset-check full|none`: choose dataset validation mode before training.
  `full` is the default and performs the model-specific validation described
  below. `none` skips dataset validation for fast startup on datasets that have
  already been checked.

Training YAML files can override augmentation values for both YOLOX and
SegFormer:

```yaml
augmentation:
  mirror: true
  rotate: true
  crop_size: 384
  cutout: 0.25
  geometry:
    prob: 0.3
    zoom_in: false
  noise:
    prob: 0.02
  distort:
    prob: 0.02
class_weights: [1.0, 0.5, 2.0]
```

Only keys that are provided need to be listed; nested values are merged with
the profile defaults. `class_weights` is passed to DeepDetect as
`mllib_parameters.class_weights`.

Training creates a run directory containing:

- `run.json`: run ID, command, model, service name, config snapshot, job ID when
  available, status, and latest status body.
- `metrics.jsonl`: metric events extracted from DeepDetect status payloads.

Training also writes `config.yaml` at the root of `--repository`. This file is
the effective training configuration after profile defaults, YAML config,
explicit CLI arguments, and `--set` overrides have all been applied.

On resume, `--weights` is optional because the model state comes from
`--repository`. When Visdom is enabled, the CLI replays previous metric history
from `--repository/metrics.json` into the run environment before streaming new
points, then ignores already-seen history points from the first live status
poll.

The CLI emits events such as `run_started`, `dataset_check`,
`training_status`, `metric`, and `run_finished`. Dataset checks are deliberately
structured as extension points. With `--dataset-check full`, list files are
checked for basic structure; YOLOX validates image paths, bbox sidecar paths,
bbox line format, class ranges, and numeric bbox values; SegFormer validates
mask values unless `--skip-mask-validation` is passed. With
`--dataset-check none`, the CLI emits a skipped dataset-check event and leaves
dataset validation to the DeepDetect backend.

For PyTorch workers, `measure.flops` and service `model_stats.flops` are
estimated from the first real model forward pass using the native PyTorch
profiler. The value is reported per sample in that model forward. For top-down
pose workers this means per cropped instance; for full-image models it means
per input image. Unsupported operators are non-fatal and leave the value
unavailable until a worker reports a supported estimate.

### Visdom Metric Sink

Training can also stream scalar losses and metrics to Visdom:

```shell
deepdetect train yolox ... --visdom --run-name ring-hand-yolox
```

Visdom options:

- `--visdom` / `--no-visdom`: enable or disable Visdom streaming.
- `--visdom-server`: server URL, default `http://localhost`.
- `--visdom-port`: server port, default `8097`.
- `--visdom-base-url`: base URL, default `/`.
- `--visdom-offline-ok` / `--no-visdom-offline-ok`: continue with a warning
  if Visdom is unavailable, default enabled.
- `--visdom-save` / `--no-visdom-save`: call `vis.save([run_name])` on close.
- `--visdom-results` / `--no-visdom-results`: upload sampled test-set
  prediction overlays, default enabled when Visdom is enabled.
- `--visdom-results-count`: number of sampled result images per test set,
  default `10`. Use `0` to disable result image uploads.
- `--visdom-results-seed`: random seed for deterministic result-image
  resampling.

The Visdom environment name is the training `run_name`. Loss-like metrics
whose names contain `loss` are each sent to their own line plot. mAP metrics
are split by metric name, for example `map`, `map-05`, `map-50`, and `map-90`
use separate windows. Per-test-set metrics such as `map-50_test0` and
`map-50_test1` are sent to the same `map-50` window with traces named `test0`
and `test1`. Other numeric metrics are grouped into scale-compatible line
plots. Non-numeric metrics are ignored with a one-time warning. Non-finite
numeric values such as `NaN` and `Inf` are silently skipped for Visdom.
Result images are resampled from every test set at each evaluation interval.
Detection results draw predicted boxes and labels on input-sized images;
segmentation results show mask overlays on input-sized images.

## Inference Commands

Inference accepts one or more images and supports batches:

```shell
deepdetect infer yolox image1.jpg image2.jpg \
  --weights yolox-nano_cls2.pt \
  --repository runs/yolox-model \
  --batch-size 2 \
  --confidence-threshold 0.25 \
  --benchmark

deepdetect infer segformer image1.png image2.png \
  --weights segformer-b0-cls2.pt \
  --repository runs/segformer-model \
  --batch-size 2 \
  --visualize \
  --output outputs/
```

Shared inference options:

- `--gpu` / `--no-gpu`: request or disable GPU execution.
- `--gpuid ID [ID ...]`: select one or more GPU ids. Comma-separated values
  are also accepted, for example `--gpuid 0,1`. Use `--gpuid -1` for all
  GPUs. Supplying `--gpuid` implies `--gpu` unless `--no-gpu` is explicitly
  passed, which is rejected.
- `--width`, `--height`: image input size. Defaults stay profile-specific:
  YOLOX uses `640x640`, SegFormer uses `480x480`.
- `--batch-size`: number of images per DeepDetect predict call.
- `--benchmark` / `--no-benchmark`: emit or suppress benchmark statistics.
- `--warmup`: run unmeasured warmup predictions before benchmarking.
- `--visualize` / `--no-visualize`: produce or suppress visual outputs.
- `--output`: file or directory for visual outputs.

Prediction events include the image path, per-image wall-clock inference time,
and the raw DeepDetect prediction payload, including available confidence
values.

## Monitoring

Version 1 uses DeepDetect async training plus structured stdout as the primary
agent monitoring path. Agents can monitor the active process by reading JSONL
events and can inspect the latest persisted status with:

```shell
deepdetect job status MODEL_REPOSITORY
```

An embedded REST status endpoint is intentionally deferred. A future HTTP
adapter should reuse the same run manifest and DeepDetect job polling logic.

## Extension Points

The implementation is split into small modules so future model families and
agent interfaces can be added without changing the command contract:

- Model profiles define service, train, and predict parameter defaults.
- Dataset checks return structured results and may be expanded into richer
  validators.
- Metric sinks accept metric events and can later forward them to TensorBoard,
  MLflow, Weights & Biases, or a DeepDetect-specific dashboard.
- Visualization helpers are optional and only run when requested.
