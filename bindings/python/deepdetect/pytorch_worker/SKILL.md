---
name: deepdetect-pytorch-worker
description: Use when porting external PyTorch models through the DeepDetect worker backend or making worker adapters exportable for inference.
---

# External PyTorch Model Import Skill

Use this guide when an LLM agent needs to port, import, adapt, or test an
external PyTorch object detection model in DeepDetect through the external
PyTorch worker backend, the `external-pytorch-detector` CLI profile,
`mllib.entrypoint` or `service_mllib.entrypoint`, connector tensor pull,
generated `extern/pytorch_workers/<model_slug>/` adapters, or generic detection
worker hooks.

## Ground The Work

Read the local implementation before editing. Start with the PyTorch worker
backend docs, the external worker README, the CLI profile code, and the current
detection worker base. Use `rg` for:

- `DetectionTrainingWorkerBase`
- `external-pytorch-detector`
- `service_mllib`
- `mllib.entrypoint`
- `connector_tensor_pull`
- `test_predictions`

Inspect the target upstream repository before designing the adapter. Identify
its package requirements, model factory, YAML or config system, training loss
path, postprocessor, checkpoint format, device handling, distributed assumptions,
download or pretrained defaults, and label and box conventions.

Verify whether a needed extension point already exists. If a core change is
required, make it generic for external detection workers rather than naming the
target model.

## Keep The Boundary

Keep committed DeepDetect changes reusable:

- Use the generic `external-pytorch-detector` profile.
- Do not add a first-slice hard-coded `<target>-detector` model profile.
- Load target code through `mllib.entrypoint` or `service_mllib.entrypoint`.
- Preserve the public worker contract: `DeepDetectWorker.configure`, `train`,
  and `predict`.
- Put generated target adapter code in `extern/pytorch_workers/<model_slug>/`
  unless the user explicitly asks to commit model-specific code.
- Treat `AGENTS.md` as operational CLI and monitoring guidance, not the place
  for a long model-porting workflow.

Target-specific adapter directories are local workspaces by default. The core
repository should not import them unless the user selects them in YAML or API
parameters.

## Build The Adapter

Create this layout for a generated external worker:

```text
extern/pytorch_workers/<model_slug>/
  worker.py
  config.yaml
  manifest.json
  README.md
  notes.md        # optional
```

The manifest should record upstream repository URL, local checkout path, commit
or tag when known, license, dependencies, entrypoint, class name, expected config
path, checkpoint compatibility, and generation notes.

The README is required. It should include a quickstart with the concrete train
and inference CLI commands for the adapter, required upstream checkout/config
settings, checkpoint expectations, key environment variables, and any
model-specific label or bbox conversion notes.

Prefer subclassing the existing detection training base for object detectors.
Implement only the target-specific pieces:

- Backend import and dependency checks.
- Model and postprocessor construction.
- Optional target config patching, such as class count, disabled teacher models,
  disabled pretrained downloads, or single-process defaults.
- DeepDetect batch-to-upstream target conversion.
- Upstream prediction-to-DeepDetect output conversion.
- Checkpoint load/save compatibility.
- Model-specific optimizer construction only when the default optimizer is not
  suitable.

Make conversions explicit. DeepDetect detection data normally uses pixel `xyxy`
boxes and one-based foreground labels, with class `0` reserved for background.
Many DETR-style models expect normalized `cxcywh` boxes and zero-based labels.
Convert labels and boxes in both directions and keep the mapping visible in code.

When useful, preserve or synthesize generic target metadata: `orig_size`, `size`,
`area`, and `iscrowd`. Keep existing `boxes`, `labels`, and `image_id` behavior
compatible with torchvision-style detection workers.

Raise typed worker errors:

- Use dependency errors for missing upstream repos, missing Python packages,
  invalid config paths, and import failures.
- Use contract errors for invalid worker classes, malformed batches, unsupported
  prediction schemas, and impossible target conversions.
- Let training errors carry real model failures after dependency and contract
  checks have passed.

## Add Generic Core Changes Only

Commit DeepDetect core edits only when they help more than one external detector.
Examples of acceptable generic work:

- Clearer runtime errors for external entrypoint loading.
- Tests that an external worker can be loaded from outside the packaged
  `deepdetect` module.
- Generic hooks for target conversion, prediction conversion, checkpoint formats,
  and optimizer construction.
- Dataset metadata enrichment needed by DETR-style detectors.
- CLI support that passes `service_mllib.entrypoint`, `service_mllib.class`, and
  target-specific YAML values through normal training flows.

Do not commit a target adapter, target config, or target dependency workaround
as a DeepDetect core change unless the user explicitly requests it.

## Validate

For committed generic backend work, add focused tests for the reusable behavior:

- Runtime loading from a temporary external entrypoint path.
- Failure mapping to dependency, contract, or launch errors.
- Preservation of current torchvision detection behavior.
- Prediction, target conversion, checkpoint, and optimizer hooks.
- CLI `external-pytorch-detector` config pass-through.

For generated target adapters, add local tests with fake upstream modules when
possible. Gate real-upstream tests behind an environment variable such as
`<MODEL>_REPO`.

Run Python tests from the repository with the local package on `PYTHONPATH`, for
example:

```shell
PYTHONPATH=bindings/python python3 -m pytest bindings/python/tests/test_pytorch_worker_runtime.py bindings/python/tests/test_cli.py
```

For manual CLI smoke tests, use the source CLI or project wrapper and select the
external detector profile:

```shell
PYTHONPATH=bindings/python python3 -m deepdetect.cli.main train external-pytorch-detector \
  --config extern/pytorch_workers/<model_slug>/config.yaml \
  --train-data train.txt \
  --test-data test.txt \
  --repository runs/<model_slug>-smoke \
  --nclasses 2 \
  --iterations 10 \
  --test-interval 5 \
  --batch-size 1 \
  --terminal verbose \
  --output-format jsonl
```

Add `--gpu --gpuid <id>` when CUDA is required. Add `--visdom --visdom-results`
when visual monitoring is needed, and inspect `sink_warning`, `run.json`,
`metrics.jsonl`, and saved files under `REPOSITORY/visdom-results/`.

## Triage Failures

If upstream import fails, install or document the missing dependency and improve
the adapter's dependency error message. If upstream code assumes distributed
training, prefer adapter-local single-process setup or disable the upstream path
that calls distributed APIs before the process group exists.

If evaluation fails after training begins, inspect device placement, cached CPU
tensors, postprocessor size arguments, label offsets, and box units. Prediction
conversion bugs often appear first during test intervals or visual result
generation.

If metrics look plausible but rendered visual results are wrong, inspect the
saved image and JSON pairs before changing training settings. Check test-set
ordering, sample indices, coordinate sizes, confidence thresholds, class ids,
and whether the backend shuffled evaluation data.

## Make New Workers Exportable

The CLI generates a relocatable `inference.yaml` with `deepdetect export RUN
--output DESTINATION`; do not add a separate recipe file. See
[CLI export documentation](../cli/CLI_SPEC.md#export-a-trained-model) for the
metadata schema and legacy-run overrides.

For new adapters, declare `export` metadata in the existing training YAML:
installed `module`, package `requirements`, inference-only settings under
`inference`, and any auxiliary `assets` with their configuration field keys.
Training preserves this metadata in the run's effective `config.yaml`. An
`inference.service_mllib` mapping replaces opaque training worker settings;
include the architecture/preprocessing parameters needed to reconstruct the
model unless the checkpoint authoritatively stores them.

Use an importable installed module for deployment. Keep a file-entrypoint shim
for development if useful, but declare its installed module explicitly. Load
an explicit `mllib.weights` checkpoint for inference without requiring solver
state, training data, the original checkout, or pretrained downloads. Keep the
existing configure/train/predict worker contract; no export hook is required.

Declare auxiliary files/directories rather than retaining absolute paths in
inference settings. References embedded inside those assets must be relative
or resolved by the adapter; the CLI does not rewrite arbitrary file formats.
Do not copy the upstream source tree into model exports.

Validate an exported checkpoint with a tar/extract relocation test: use a
second directory and different working directory, make original training paths
unavailable, and compare predictions on the same inputs. Verify that only the
selected checkpoint and declared assets are needed with the worker package
installed. Record any precision/device requirements in the adapter README.
Use `export --test-inference` for a basic load/run check on a random image at
the exported dimensions; inspect `inference_test` in `export.json`. This smoke
test does not replace comparison on real inputs or model-quality assessment.
