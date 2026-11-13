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
- `external-pytorch-image`: an external self-supervised image worker selected
  through YAML/API `mllib.entrypoint`. DeepDetect passes manifest paths and the
  worker loads images directly in Python.
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
deepdetect train external-pytorch-image ...
deepdetect train vitpose ...
deepdetect infer yolox ...
deepdetect infer segformer ...
deepdetect infer vitpose ...
deepdetect export RUN_DIR --output EXPORT_DIR
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
Native C++ detection profiles use the dataset-level detection mAP v2
implementation by default. Use `--detection-map-version 1` or the top-level
YAML field `detection_map_version: 1` to retain the historical native metric.
The direct DeepDetect API field is
`parameters.output.detection_map_version`. PyTorch workers continue to use
their Python detection evaluator.

Both the native version 2 and Python detection evaluators automatically report
per-class AP for foreground classes present in the test ground truth. For
example, `map-50_1` is class `1` AP at IoU `0.50`, while `map_1` is that
class's AP averaged over the requested IoU thresholds. Python worker metrics
retain their test-set suffix, for example `map-50_1_test0`. Background class
`0`, prediction-only classes, and classes absent from a test set are omitted.
When native results from multiple test sets are aggregated, each per-class
metric is averaged only over the test sets containing that class.

Example default-style configs are provided next to this document:

- `yolox-default.yaml`
- `segformer-default.yaml`
- `torchvision-detector-default.yaml`
- `external-pytorch-detector-default.yaml`
- `external-pytorch-image-default.yaml`
- `vitpose-default.yaml`
- `sam2-default.yaml`

The trainable model configs include both training and inference keys; the SAM2
config is inference-only. Replace the dataset and model paths before use:

```shell
deepdetect train yolox --config bindings/python/deepdetect/cli/yolox-default.yaml
deepdetect infer yolox image.jpg --config bindings/python/deepdetect/cli/yolox-default.yaml
deepdetect train torchvision-detector --config bindings/python/deepdetect/cli/torchvision-detector-default.yaml
deepdetect train external-pytorch-detector --config bindings/python/deepdetect/cli/external-pytorch-detector-default.yaml
deepdetect train external-pytorch-image --config bindings/python/deepdetect/cli/external-pytorch-image-default.yaml
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

`external-pytorch-image` deliberately has no `mllib.data_source` setting.
DeepDetect forwards the train/test manifest paths to the worker, which is
responsible for decoding and transforming each image. Manifests contain an
image path and may contain one optional label field; generic validation checks
the files, while the worker decides whether labels have a useful objective.
Model-specific masked patches should be created by the Python model or dataset,
not by a generic C++ connector. Reconstruction workers persist visual outputs
under their model repository rather than returning encoded images through the
prediction API.

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
  staging `--weights`. `latest` selects a matching checkpoint and solver state
  from the repository. `best` reads `best_model.txt` and resumes the
  matching `checkpoint-N.*` and `solver-N.pt` files. Resume takes precedence
  over all initialization weights. An explicitly supplied `--weights` is
  ignored with a warning; profile and YAML default weights are ignored
  silently. Python workers define `latest` as the highest numbered iteration
  with both `checkpoint-N.pt` and `solver-N.pt`, with the corresponding
  `*-latest.pt` aliases accepted only when no numbered pair exists.
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
- `--cancel-timeout`: grace period for asynchronous `pytorch` workers after a
  cooperative cancellation request. It defaults to 30 seconds and must be
  positive.
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
- `training-observability.json`: versioned, framework-neutral index describing
  the metadata, metric stream, and visual-artifact index.
- `artifacts.jsonl`: append-only index of saved prediction overlays when visual
  results are enabled.

Training also writes `config.yaml` at the root of `--repository`. This file is
the effective training configuration after profile defaults, YAML config,
explicit CLI arguments, and `--set` overrides have all been applied.

`command.txt` is also written at the root of `--repository` on each training
invocation, including resumes. It contains a shell-quoted command using the
current Python interpreter and `-m deepdetect.cli.main`, preceded by `cd` to
the original working directory so relative paths still resolve. The command
preserves the supplied CLI arguments; referenced input files, YAML configs,
and environment variables must still be available when rerunning it.

On resume, `--weights` is ignored because the model state comes from
`--repository`. When Visdom is enabled, the CLI replays previous metric history
from `--repository/metrics.json` into the run environment before streaming new
points, then ignores already-seen history points from the first live status
poll.

Legacy native detection runs resumed without an explicit
`detection_map_version` use version 2. Their existing `map*` history may
therefore contain version 1 values followed by version 2 values; select version
1 explicitly when a single historical metric definition is required.

The CLI emits events such as `run_started`, `dataset_check`,
`training_status`, `metric`, and `run_finished`. Dataset checks are deliberately
structured as extension points. With `--dataset-check full`, list files are
checked for basic structure; YOLOX validates image paths, bbox sidecar paths,
bbox line format, class ranges, and numeric bbox values; SegFormer validates
mask values unless `--skip-mask-validation` is passed. With
`--dataset-check none`, the CLI emits a skipped dataset-check event and leaves
dataset validation to the DeepDetect backend.

For asynchronous profiles using the `pytorch` worker backend, the first
Ctrl-C requests cooperative cancellation so the worker can write its final
checkpoint. Monitoring continues with a `cancelling` status, and another
Ctrl-C forces worker process-group termination. If the worker does not finish
within `--cancel-timeout`, the CLI escalates automatically and reports
`terminating`. Cooperative and forced shutdown finish as `cancelled` and
`terminated`, respectively, and return exit code 130. If training finishes
while cancellation is racing with completion, the run remains `finished` and
returns zero. Blocking `--sync` training and other backends retain their
existing interrupt behavior.

### Agent Observation CLI

The optional `training-observability` package is installed with the wheel and
adds a JSON-first, read-only CLI. Install `deepdetect[observability]` when PNG
rendering is needed:

```shell
training-observe summary REPOSITORY
training-observe metrics REPOSITORY --name train_loss
training-observe plots REPOSITORY
training-observe render REPOSITORY --plot loss-train-loss --output /tmp/train-loss.png
training-observe artifacts REPOSITORY --step latest
```

Metric reads use the flushed `metrics.jsonl` stream, not a live Visdom server.
`render` only writes its requested output file and never changes the run. The
generic `training_observability.RunWriter` can emit the same manifest, metrics,
and artifact index from diffusion, reinforcement-learning, or other trainers.
Existing DeepDetect runs without `training-observability.json` remain readable
through the legacy `run.json` / `metrics.jsonl` / `visdom-results` layout.

Other trainers need only create their own run root and emit scalars/artifacts
through the standalone package:

```python
from training_observability import RunWriter

run = RunWriter("runs/diffusion-001", run_id="diffusion-001")
run.metric("train/loss", 0.081, step=12000)
run.artifact(kind="sample-grid", path="samples/12000.png", step=12000)
```

For PyTorch workers, `measure.gflops` and service `model_stats.gflops` are
estimated from the first real model forward pass using the native PyTorch
profiler. The value is reported in billions of floating-point operations
(GFLOPs) per sample in that model forward. For top-down pose workers this means
per cropped instance; for full-image models it means per input image.
Unsupported operators are non-fatal and leave the value unavailable until a
worker reports a supported estimate.

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
and `test1`. Per-class metrics share that window: `map-50_1` uses trace
`class 1`, and `map-50_1_test0` uses `class 1 / test0`. Existing global trace
names are unchanged so resumed Visdom histories remain continuous. Other
numeric metrics are grouped into scale-compatible line plots. Non-numeric
metrics are ignored with a one-time warning. Non-finite numeric values such as
`NaN` and `Inf` are silently skipped for Visdom.
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

Large semantic-segmentation images can be inferred without resizing the full
source to the model input size. Tiled mode keeps one DeepDetect service alive,
predicts overlapping model-sized crops in batches, and stitches hard class
labels using nearest-tile-center ownership:

```shell
python3 -m deepdetect.cli.main infer segformer \
  images/*.jpg \
  --weights segformer-b0-cls2.pt \
  --repository runs/segformer-model \
  --service-name segmentation \
  --nclasses 2 \
  --width 512 \
  --height 512 \
  --tiled \
  --confidence-maps \
  --tile-overlap 128 \
  --batch-size 8 \
  --gpu --gpuid 0 \
  --output outputs/ \
  --output-format text
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

SegFormer tiled-inference options:

- `--tiled`: enable Python-side tiled semantic-segmentation inference. This
  requires `--output`, supports at most 256 classes, and is rejected by
  non-segmentation profiles.
- `--tile-overlap PIXELS`: overlap on both axes. It must satisfy
  `0 <= overlap < min(width, height)`. When omitted, the overlap is 25% of the
  smaller tile dimension.
- `--preview-max-side PIXELS`: bound the largest side of the overlay preview;
  the default is `4096`.
- `--confidence-maps`: request DeepDetect's per-pixel winning-class softmax
  confidence and write full-resolution 16-bit confidence maps. It requires
  `--tiled`.
- In tiled mode, `--width` and `--height` are both the tile dimensions and
  model input dimensions, while `--batch-size` is the number of tiles in each
  DeepDetect prediction call.

For each source image, tiled inference atomically writes
`<stem>_mask.png`, a full-resolution paletted class-index mask, and
`<stem>_overlay_preview.png`, a bounded review image. Binary preview reduction
preserves foreground pixels so thin features remain visible. The CLI emits a
compact `tile_plan`, one `tile_progress` event per batch, a final `prediction`
summary with dimensions, tile count, class histogram, and inference time, and
normal artifact events. It never emits the full stitched pixel array. Tiled
benchmark events additionally contain total tile count, average milliseconds
per tile and per source image, and tile throughput.

With `--confidence-maps`, tiled inference also writes
`<stem>_confidence.png`, the winning-class softmax confidence, as a
full-resolution 16-bit grayscale PNG. For binary segmentation it additionally
writes `<stem>_foreground_probability.png`, the class-1 probability derived
exactly from the winning class and confidence. Convert any stored pixel value
back to probability with `value / 65535`. Confidence arrays remain excluded
from stdout events.

The equivalent YAML keys are `tiled`, `tile_overlap`, and
`preview_max_side`, and `confidence_maps`.

Prediction events include the image path, per-image wall-clock inference time,
and, for ordinary inference, the raw DeepDetect prediction payload including
available confidence values. Tiled prediction events use the compact summary
described above.

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

### External semantic segmentation workers

`external-pytorch-segmentation` selects `backend="pytorch"`,
`task="segmentation"`, a 512×512 RGB image connector and two classes by default.
It needs `service_mllib.entrypoint` and optionally `service_mllib.class` and
`service_mllib.python` (the latter defaults to the CLI interpreter). It does not
require native SegFormer weights. See `external-pytorch-segmentation-default.yaml`
and the semantic-result contract in `docs/pytorch_worker_backend.md`.

Use `service_mllib.segmentation` for worker settings and `mllib.segmentation`
for training overrides. Mask checking merges these namespaces and supports
`ignore_label` (255 by default for this external profile), `label_mapping`, and
explicit RGB `color_mapping`. Existing segmentation profiles retain their
previous default label range. Model loading, losses and preprocessing belong
to the external worker; CLI run manifests, JSONL metrics, masks/overlays, and
saved validation image/JSON pairs use the normal segmentation flows.


### Segmentation ground-truth comparison rows

The training visualizer requests `test_predictions.ground_truth=true` for
segmentation. Workers may attach a `ground_truth` object to each sampled result:
`imgsize` matching the prediction, flattened integer `vals`, and `ignore_label`
(default 255). These must be the targets actually evaluated, after preprocessing.
Optional `mask_convention` metadata distinguishes support/dilated targets.

When every displayed sample supplies ground truth, Visdom uses two aligned rows:
predictions above, ground truth in green below. Ignored target pixels are gray in
both rows. Ground-truth overlays are saved as `sample-NNNNNN-ground-truth.png`,
linked from each sample JSON's `ground_truth_overlay` field; the exact target
values remain in the JSON prediction payload. Existing prediction masks and
values are unchanged. Backends without this optional payload retain their
prediction-only display; the CLI does not guess targets from source annotations.

### Probability fusion and source-scale tiled segmentation

`infer ... --tiled` keeps center ownership as its default. Optional settings:

- `--tile-source-size N`: extract square N-pixel source windows and resize each
  to the configured model `--width`/`--height`. `--tile-overlap` is measured in
  source pixels; its default remains 25% of the source window's smaller side.
- `--tile-merge weighted-probability`: average each class probability with
  separable raised-cosine weights (each axis floored at `1e-3`) before argmax.
- `--tile-merge max-confidence`: choose the complete class-probability vector
  from the tile with the greatest winning-class probability. Background can
  win; this is not a foreground union. Exact ties keep the first row-major tile.
- `--tile-padding edge`: replicate image boundaries for undersized inputs.
  The default remains constant black padding. Padded outputs are excluded.

The source-scale/probability path requires confidence maps for **every class**;
unsupported backends fail clearly. Image tiles are resized with Pillow bilinear
interpolation; returned probabilities are bilinearly mapped back before fusion.
This is distinct from training pipelines that resize floating-point pixels before
quantization. Class ties use the lowest class ID. `--confidence-maps` controls
saving full-resolution 16-bit confidence maps, not the probabilities internally
required for fusion. A per-image `_tiling.json` records geometry and fusion.

Python callers may use `run_tiled_image(..., source_size=1024,
merge_modes=("weighted-probability", "max-confidence"), padding="edge")` to
produce both outputs from the same tile predictions. Multiple modes write to
mode-named subdirectories. Disk-backed temporary accumulators bound resident
memory; temporary files are removed on failure as well as success. Results are
finalized in row chunks. Whole RGB inputs, class masks, and saved confidence
maps still require memory proportional to source dimensions.

## Export a trained model

Export selects a checkpoint and generates `inference.yaml` automatically from
`run.json` and the effective `config.yaml`. There is no separate recipe file.
By default it does not launch a service, deserialize weights, or require a GPU.
It never modifies the training repository. Add `--test-inference` to verify the
exported model with an actual inference call.

```shell
python3 -m deepdetect.cli.main export runs/my-model --output models/my-model
python3 -m deepdetect.cli.main infer yolox image.jpg --config models/my-model/inference.yaml
```

All trainable CLI profiles support export. Native Torch exports include the
selected `.pt`, `.npt`, and/or `.ptw` components and required graph/class mapping
files. Worker exports contain the selected `.pt` and declared inference assets.
Worker code and Python dependencies must be installed in the destination
environment. File-entrypoint workers, including ViTPose, need an installed
module declaration; export never copies arbitrary worker source trees.

Options:

- `SOURCE --output DESTINATION`: source run and a **new**, non-overlapping export
  directory. Existing destinations are rejected, even when empty.
- `--checkpoint best|latest|ITERATION`: defaults to `best`, which strictly reads
  `best_model.txt`. `latest` selects the highest numbered checkpoint, falling
  back to `checkpoint-latest.pt` only if numbered checkpoints are absent.
  Missing best checkpoints are errors; solver files are never required.
- `--model PROFILE`, `--config PATH`: supply missing profile/effective training
  configuration. A profile conflicting with `run.json` is rejected.
- `--worker-module PACKAGE.MODULE`: installed worker module, overriding the
  saved file entrypoint/module. The saved interpreter path is discarded.
- `--asset CONFIG_KEY=SOURCE`: copy a file or directory and bind its new path to
  a scalar inference config field, e.g. `service_mllib.model_config`. CLI source
  paths resolve from the current working directory.
- `--set KEY=VALUE`, `--unset KEY`: adjust generated inference settings. Unlike
  training's deep-merge overrides, export assignments **replace the addressed
  value**, so `--set 'service_mllib.custom={}'` clears an opaque mapping.
  Unsets are applied after assignments. Export owns the selected `weights`,
  `repository`, `_export`, and worker checkpoint/lifecycle fields.
- `--dry-run`: validate and print the selection/inventory without writing.
- `--test-inference`: after export, generate one deterministic random image
  (seed 0) at the exported width/height and run inference using the exported
  configuration. Grayscale inputs are supported; top-down keypoint models
  receive a full-image bounding box. The test uses a temporary copy of the
  bundle so service-generated files do not clutter the export. It requires a
  working native runtime and the installed worker/dependencies. It uses the
  exported device settings (CPU by default); for GPU use, for example,
  `--set gpu=true --set gpuid=0`. It cannot be combined with `--dry-run`.
  This checks model loading and a prediction response, not model quality.
  Success prints `Inference test: PASSED`; a failure returns exit code 1,
  prints `inference test FAILED`, and retains the export for inspection.
  `export.json` records `inference_test` with status, input dimensions/seed,
  device and elapsed time on success, or the error on failure. JSON output
  reports `export_test_failed` on test failure rather than `export_finished`.
  Inference logs are routed to stderr to keep JSON stdout machine-readable.
- `--output-format text|json|jsonl`: defaults to a concise text success message
  identifying the destination and selected checkpoint. `--dry-run` explicitly
  reports that no files were written and lists the planned files. JSON modes
  emit `export_finished` with `status: success`, or `export_planned` with
  `status: planned`, and retain the full machine-readable inventory. The full
  completed report is also saved in `export.json`. Failures print an error on
  stderr and return a nonzero exit code; no completion message is emitted.

The generated `inference.yaml` preserves model settings and preprocessing,
uses batch size 1 and CPU by default, and contains bundle-relative paths.
Use `--gpu --gpuid 0` at inference to select GPU execution. Inference explicitly
loads the YAML with `--config`; it does not discover files by name. Declared
bundle paths resolve relative to that YAML, independently of the working
directory, before inference CLI overrides are applied. Ordinary YAML and CLI
path semantics remain unchanged. Keep the YAML together with its assets when
moving or extracting an archive.

`export.json` records the selected iteration, best-marker information (for best
selection), file sizes/SHA-256 hashes, worker module/class, declared package
requirements, and available package versions. It is provenance, not an
exhaustive environment lockfile. If requirements were not declared, export
attempts to identify the installed distribution providing the worker module.
No packages are imported, downloaded, or installed by export.

Only selected model components and inference assets are copied. Optimizer
states, other checkpoints, datasets, training configs/logs/metrics, and saved
visualizations are excluded. Output is assembled in a temporary sibling and
published after validation. Detected changes to copied source files abort the
export. Export from a stable checkpoint; do not modify it concurrently.
Atomic publication uses Linux `renameat2` with no-replace semantics, including
when another process creates the destination during export.

### External worker export metadata

New adapters should declare optional `export` metadata in their **existing
training YAML**. It is retained in the effective run configuration:

```yaml
export:
  version: 1
  module: my_worker.worker
  requirements: [my-worker==1.2.0]
  inference:
    service_mllib:
      task: detection
      class: DeepDetectWorker
      backend_parameters:
        pretrained: false
  assets:
    - key: service_mllib.backend_parameters.model_config
      source: ./model-definition.yaml
      destination: assets/model-definition.yaml
```

`inference` replaces generated values at the top level; in particular,
`service_mllib` replaces the opaque training worker configuration. The CLI then
applies `--set`/`--unset`, module selection, and asset mappings. Standard worker
lifecycle fields are removed and the checkpoint/repository are supplied by
export. Unspecified top-level inference values come from the run/profile.

Each asset has `key`, `source`, and an optional bundle-relative `destination`.
Without a destination, it goes under `assets/INDEX/BASENAME`. Metadata source
paths resolve relative to the training YAML; training persists absolute source
paths so a later export does not depend on the launch directory. When using a
legacy run or explicit `--config`, relative metadata sources resolve relative
to that configuration file. Asset fields use dotted mapping keys (not list
indices); directory assets support workers needing several related files.

Visible absolute and existing relative local paths left in generated settings
are rejected: declare them as assets, or remove training-only settings. Export
cannot discover dependencies embedded inside arbitrary checkpoint/config file
formats. Such references must already be relative to the bundled asset or
handled by the worker. Opaque worker parameters are otherwise preserved.
File symlinks are dereferenced; symlink directories inside assets, traversal,
missing files, and destination collisions are rejected.

For the existing cracks B5 run, which embeds inference model settings in its
checkpoint, this exports the recorded best (`checkpoint-3300.pt`) without the
training-only segmentation configuration:

```shell
python3 -m deepdetect.cli.main export \
  /data1/airbuscracksv2_platform/code/cracks_deepdetect_worker/runs/v5-b5-moderate-skip \
  --output /tmp/cracks-b5-export \
  --worker-module cracks_deepdetect_worker.worker \
  --set 'service_mllib.segmentation={}'

python3 -m deepdetect.cli.main infer external-pytorch-segmentation tile.png \
  --config /tmp/cracks-b5-export/inference.yaml --gpu --gpuid 0
```

Install `cracks-deepdetect-worker` and its dependencies in the destination
interpreter. Solver files and the initial pretrained encoder are unnecessary
for this worker's explicit checkpoint inference. Other workers must declare
their own inference requirements rather than assuming this checkpoint format.

Real-model relocation smoke tests are opt-in. With the native wheel and worker
installed in the test interpreter, set `DEEPDETECT_EXPORT_CRACKS_RUN` and/or
`DEEPDETECT_EXPORT_NATIVE_RUN` to completed cracks/native segmentation runs and
run `tests/test_cli_export.py -k real_model`. They compare CPU predictions
before and after tar extraction elsewhere, using this checkout's CLI with the
installed native engine. The worker test needs permission to bind a local socket.

### Best-effort ONNX export during training

ONNX conversion is optional and **off by default**. Both native Torch and Python
worker training accept this YAML configuration through the existing `mllib`
pass-through (direct API: `parameters.mllib.onnx_export`):

```yaml
mllib:
  onnx_export:
    enabled: true
    timeout_seconds: 120
    opset_version: 17
    # python: /absolute/path/to/python
    # provider: installed_module:build_onnx_model
```

Install the Python package's `onnx` extra in the export interpreter. Native Torch
uses `python`, then `DEEPDETECT_PYTHON`, then `python3`; Python workers default to
their current interpreter. Dependencies are never installed automatically.

After saving the aggregate best checkpoint (`best_model.txt`), training copies
its model components to an export-owned snapshot. Conversion runs in a separate
CPU process while training continues. There is one active attempt and one pending
candidate; newer best checkpoints replace the pending candidate. Per-test-set
best checkpoints do not trigger export. Normal training completion drains the
active and latest pending attempts; cancellation kills conversion and discards
pending work. Snapshot copying still adds checkpoint I/O overhead.

The first export failure disables all subsequent attempts for that run, including
resume. Missing dependencies/providers, unsupported models, process crashes,
timeouts, and publication errors are nonfatal export failures. Fresh non-resumed
training resets the latch. Explicit cancellation does not latch a failure; an
unexplained interrupted conversion does. Resume requeues an unexported current
best checkpoint if export has not been disabled.

Native Torch directly attempts only self-contained TorchScript `.pt` checkpoints
with one fixed-shape image tensor (batch size one, inference width/height/channels).
Native `.npt`, graph archives, split heads, and other input signatures need an
explicit provider. Worker export always requires a provider; arbitrary state
dictionaries cannot be reconstructed generically. The shared detection training
base integrates best-checkpoint notifications; custom worker loops must integrate
the SDK helper described below.

A provider is an installed `module:function` callable receiving
`(checkpoint_paths, inference_configuration)` and returning
`(cpu_eval_model, example_positional_inputs, names)`. Inputs must be a tuple of CPU
tensors. `names` may contain `input_names` and `output_names`. Load the exact saved
checkpoint without training data, pretrained downloads, or the live model. Native
configuration includes `width`, `height`, `channels`, and request `parameters`;
worker configuration includes `service`, effective `mllib`, and request
`parameters`. The provider's module must be importable in the export interpreter.

Custom workers can use `deepdetect.onnx_export.BestOnnxExporter`, passing
`backend="pytorch"`, the repository, export config, inference config, and resume
flag. Call `notify(iteration, model_checkpoint_paths)` only after committing an
aggregate best checkpoint, and only on the checkpoint-writing rank. Call
`close(cancellation=lambda: cancellation.requested)` to drain at completion and
`close(cancel=True)` on cancellation or exceptions. Always arrange cleanup with
`try/finally`. Existing `configure/train/predict` contracts do not change.

`REPOSITORY/onnx/state.json` is the atomic manifest for the latest successful
bundle and current export status. Its `bundle` directory contains `model.onnx`,
any external tensor data, and `metadata.json` recording iteration, checkpoint
hashes, input signatures, and export settings. Do not assume the ONNX artifact
matches the latest best checkpoint: compare `exported_iteration` with
`best_iteration` or inspect `stale`. Earlier successful output is retained when a
later attempt fails. Backend training status and CLI `training_status` events
expose `onnx_export` separately from metrics; `run.json` preserves it as well.

Conversion uses one fixed-shape `torch.onnx.export(..., dynamo=False)` attempt and
ONNX checker validation. DeepDetect does not provide architecture surgery,
operator patches, fallback exporters, numerical parity guarantees, or TensorRT
compatibility guarantees. Preprocessing and postprocessing are included only if
present in the exported model. Existing `deepdetect export` packages checkpoints;
it is separate from this ONNX conversion facility.
