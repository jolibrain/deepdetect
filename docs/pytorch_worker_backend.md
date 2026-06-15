# Managed Python PyTorch Worker Backend

## Summary

This document specifies a new DeepDetect backend for Python-native PyTorch
training and inference. The backend is exposed as `mllib: "pytorch"` and is
implemented as a managed Python subprocess supervised by the DeepDetect C++
server.

The goal is to keep DeepDetect as the stable API, service lifecycle, model
repository, connector contract, metric, and deployment control plane while
allowing modern PyTorch models to be implemented in Python. This is intended
for research and development workflows where new architectures are often only
available as Python code, and where TorchScript or C++ reimplementation is not
a durable training path.

The current `mllib: "torch"` backend remains the existing C++/libtorch
backend. The new backend must not change existing `torch` service behavior.

## Goals

- Run PyTorch training code in Python while preserving DeepDetect service,
  train, status, cancel, predict, connector, output, and repository semantics.
- Make object detection and segmentation training compatible with current
  DeepDetect CLI monitoring, metrics, Visdom, and saved visual result flows.
- Provide a worker structure that LLM agents can reliably author or adapt from
  external academic repositories.
- Support single-machine multi-GPU training by delegating distributed execution
  to Python-side launchers such as `torchrun` or `accelerate`.
- Define a protocol that can later accept CPU and GPU tensor references from
  DeepDetect C++ input connectors without redesigning worker scripts.

## Non-Goals For V1

- No GPU zero-copy implementation.
- No multi-node distributed training.
- No automatic production export to TensorRT, ONNX, or AOTInductor.
- No concurrent prediction against a model while offline training is active.
- No requirement that Python workers implement raw socket/protocol handling.
- No replacement or deprecation of the existing C++ libtorch backend.

## Backend Surface

### Service Creation

A Python worker service is created with `mllib: "pytorch"`:

```json
{
  "mllib": "pytorch",
  "description": "torchvision detection finetune",
  "type": "supervised",
  "model": {
    "repository": "/data/models/detector"
  },
  "parameters": {
    "input": {
      "connector": "image",
      "width": 640,
      "height": 640,
      "rgb": true,
      "bbox": true
    },
    "mllib": {
      "task": "detection",
      "entrypoint": "/data/workers/train_worker.py",
      "class": "DeepDetectWorker",
      "gpu": true,
      "gpuid": [0, 1],
      "distributed": {
        "enabled": true,
        "launcher": "torchrun",
        "nproc_per_node": 2
      },
      "backend_parameters": {
        "model": "fasterrcnn_mobilenet_v3_large_fpn",
        "pretrained": true
      }
    },
    "output": {
      "bbox": true,
      "measure": ["map", "map-50", "map-90"],
      "best_bbox": 100,
      "confidence_threshold": 0.1
    }
  }
}
```

Required `parameters.mllib` fields:

- `task`: one of `classification`, `detection`, `segmentation`, `embedding`,
  or `custom`.
- `entrypoint` or `module`: Python file or importable module containing the
  worker class.
- `class`: Python worker class name. Defaults to `DeepDetectWorker` if omitted.
- `backend_parameters`: model-specific opaque JSON passed unchanged to the
  Python worker.

Standard DeepDetect fields remain authoritative:

- `model.repository`
- `parameters.input`
- `parameters.output`
- `gpu`
- `gpuid`
- service name, type, description, and async job behavior

### Backend Registration

The C++ API should add a separate service branch for `mllib == "pytorch"`,
parallel to the existing `torch` branch. The service type should be a new
`PytorchWorkerLib` template instantiation over existing input and output
connectors where possible:

- `ImgPytorchInputFileConn` for image tasks.
- Future connector wrappers for text, CSV, time series, or multimodal tasks.
- Existing `SupervisedOutput` for classification, detection, segmentation, and
  regression-like supervised predictions.

The first implementation may reuse existing input connector classes to parse
and validate parameters, then generate a connector manifest for the worker.

## Data Boundary

### V1: Connector Manifest, Python-Side Loading

V1 uses DeepDetect connectors as the source of truth for data semantics, but
the Python worker reads sample files and builds tensors.

Flow:

```text
DeepDetect API request
  -> C++ validates service and connector parameters
  -> C++ resolves train/test manifests and writes connector_manifest.json
  -> Python worker reads images, labels, bbox files, and masks
  -> Python worker builds PyTorch datasets and tensors
  -> Python worker reports metrics/results through the protocol
```

The C++ side must provide:

- resolved train and test dataset list paths;
- resolved per-sample image paths and target sidecar paths when available;
- input connector parameters such as width, height, rgb, scale, mean, std,
  bbox, segmentation, and shuffle;
- class count and optional class names;
- output connector parameters such as requested measures, `bbox`,
  `confidence_threshold`, `best_bbox`, and test prediction sampling;
- repository path and run/job identifiers.

The Python worker is responsible for applying equivalent transforms. The
worker must write the effective connector manifest into the model repository
so training and inference preprocessing can be audited.

### V1.1: C++ Connector-Produced CPU Tensors

V1.1 moves actual sample reading and preprocessing into the C++ input
connectors for selected connectors. The worker receives CPU tensor references
or shared memory references instead of paths.

Flow:

```text
DeepDetect C++ input connector reads and transforms
  -> C++ emits batch TensorRef objects
  -> Python worker converts TensorRef to torch.Tensor
  -> Python worker trains/evaluates
```

V1.1 must preserve the worker class contract. A worker that currently consumes
path-backed samples should be able to consume tensor-backed samples through a
dataset adapter without changing model, optimizer, metric, or checkpoint code.

### Reserved V2: GPU Tensor References

The protocol reserves a GPU tensor boundary for later work. V1 must not
implement CUDA IPC, DLPack ownership, or CUDA stream negotiation, but the
message schema should avoid blocking them.

Reserved `TensorRef` shape:

```json
{
  "kind": "tensor_ref",
  "device": "cpu",
  "dtype": "float32",
  "shape": [4, 3, 640, 640],
  "layout": "strided",
  "strides": [1228800, 409600, 640, 1],
  "storage": {
    "type": "shared_memory",
    "name": "dd-pytorch-job-7-batch-42",
    "offset": 0,
    "nbytes": 19660800
  },
  "lifetime": {
    "owner": "deepdetect",
    "valid_until_ack": "batch_done"
  },
  "cuda": {
    "ipc_handle": null,
    "stream": null
  }
}
```

## Worker Protocol

### Transport

The C++ supervisor starts the Python worker and communicates through a local
Unix domain socket. Messages are framed JSON. Large payloads are passed by
reference, never embedded in control JSON.

Worker launch command receives:

- socket path;
- config path;
- service name;
- repository path;
- job id when launched for training;
- GPU visibility through environment variables;
- distributed launcher settings when requested.

Recommended environment:

- `CUDA_VISIBLE_DEVICES` reflects selected `gpuid`.
- `DEEPDETECT_SERVICE_NAME`
- `DEEPDETECT_REPOSITORY`
- `DEEPDETECT_WORKER_CONFIG`
- `DEEPDETECT_WORKER_SOCKET`
- `DEEPDETECT_JOB_ID`

### Supervisor To Worker Messages

`hello`

Requests protocol negotiation.

```json
{
  "id": 1,
  "method": "hello",
  "params": {
    "protocol_version": 1,
    "deepdetect_version": "unknown"
  }
}
```

`configure`

Sends service configuration, connector manifest, output configuration, and
backend parameters.

`train_start`

Starts a training job. Contains train data, test data, solver parameters, and
output request.

`train_status`

Requests latest status snapshot.

`train_cancel`

Requests cooperative cancellation.

`predict`

Runs prediction on path-backed samples or tensor references.

`shutdown`

Gracefully terminates the worker service.

### Worker To Supervisor Messages

`hello_result`

Declares worker capabilities:

```json
{
  "id": 1,
  "result": {
    "protocol_version": 1,
    "capabilities": {
      "train": true,
      "predict": true,
      "task": "detection",
      "metric_frequency": "optimizer_step",
      "template": "torchvision_fasterrcnn_mobilenet",
      "output_format": "deepdetect_detection_v1",
      "cpu_tensor_input": false,
      "gpu_tensor_input_reserved": true,
      "distributed_launcher": true
    }
  }
}
```

`status`

Reports current state:

```json
{
  "event": "status",
  "job": 7,
  "payload": {
    "phase": "train",
    "iteration": 42,
    "iterations": 10000,
    "elapsed_time_ms": 12345,
    "remain_time": 456789,
    "gpu": [0, 1]
  }
}
```

`metric`

Reports one scalar metric:

```json
{
  "event": "metric",
  "job": 7,
  "payload": {
    "name": "train_loss",
    "value": 1.234,
    "iteration": 42
  }
}
```

`artifact`

Reports a checkpoint, config, exported model, or visual result.

`failure`

Reports a typed failure. See "Failure Handling".

`heartbeat`

Confirms liveness. The supervisor treats missed heartbeats as a timeout.

## Python Worker Runtime

Workers should not implement sockets or message framing directly. They should
implement a small class consumed by a DeepDetect Python worker runtime.

Normative class contract:

```python
class DeepDetectWorker:
    def configure(self, context):
        """Receive repository, connector, output, backend, and runtime config."""

    def build_datasets(self):
        """Create train/test datasets from connector manifest or TensorRef adapters."""

    def build_model(self):
        """Create and return the torch.nn.Module."""

    def build_optimizer(self):
        """Create optimizer and optional scheduler/scaler."""

    def train_one_iteration(self, batch):
        """Run one optimizer step and return scalar metrics."""

    def evaluate(self, test_set):
        """Run evaluation for one test set and return metrics and sample predictions."""

    def predict(self, batch):
        """Return predictions in a DeepDetect-compatible output schema."""

    def save_checkpoint(self, iteration):
        """Write model and optimizer state into the repository."""

    def load_checkpoint(self, path):
        """Restore model and optimizer state."""
```

The runtime provides:

- `context`: immutable service/job configuration;
- `reporter`: status, metric, artifact, log, heartbeat, and failure reporting;
- `cancellation`: cooperative cancellation flag;
- `rank`: distributed rank metadata;
- helper functions for checkpoint paths and repository artifact paths.

### Worker SDK

Phase 2 introduces a small Python SDK in
`deepdetect.pytorch_worker.sdk`. Worker authors should subclass
`DeepDetectWorkerBase` and use:

- `WorkerContext` for repository and backend configuration access;
- `WorkerReporter` for status, scalar metric, artifact, and log events;
- `Cancellation` for cooperative training shutdown;
- `WorkerContractError`, `MetricContractError`,
  `PredictionContractError`, `DatasetContractError`, and
  `WorkerDependencyError` for typed failures.

Minimal worker skeleton:

```python
from deepdetect.pytorch_worker.sdk import DeepDetectWorkerBase


class DeepDetectWorker(DeepDetectWorkerBase):
    def configure(self, context):
        super().configure(context)
        return {"worker": "my-worker"}

    def train(self, params, *, reporter, cancellation):
        for iteration in range(1, 11):
            if cancellation.requested:
                return {"status": "cancelled", "iteration": iteration - 1}
            reporter.status(phase="train", iteration=iteration, test_active=0)
            reporter.metric("train_loss", 1.0 / iteration, iteration=iteration)
        return {"status": "finished", "iteration": 10}

    def predict(self, params):
        return {"results": [{"uri": "sample", "probs": [1.0], "cats": ["ok"]}]}
```

The runtime validates SDK outputs at the Python boundary before sending them to
C++:

- metrics must be finite numeric scalars with non-empty names;
- status, artifact, log, train, configure, and predict payloads must be JSON
  serializable;
- predict must return `{"results": [...]}`;
- detection bboxes must contain finite `xmin`, `ymin`, `xmax`, and `ymax`.

Existing non-SDK workers that implement compatible `configure`, `train`, and
`predict` methods remain loadable. Contract failures are reported through the
stable protocol categories before crossing into C++.

## Normative Object Detection Worker Template

The first template should be a real file that future workers can copy:

```text
bindings/python/deepdetect/pytorch_worker/builtin/vision/detection/torchvision_fasterrcnn.py
```

The template implements object detection finetuning with torchvision.

Default model:

```python
torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn
```

The template must:

- create a torchvision detection model;
- replace the box predictor for `nclasses`;
- preserve label `0` as background;
- train with model-owned losses;
- keep optimizer and gradient accumulation inside the worker;
- leave scheduler, AMP, and distributed wrappers as later extensions;
- report metrics every optimizer step;
- evaluate every configured test interval;
- return predictions in DeepDetect detection schema;
- save checkpoints and optimizer state in the DeepDetect repository.

### Detection Dataset Contract

The template defines `DeepDetectDetectionDataset`.

Input manifest sample:

```json
{
  "sample_index": 12,
  "image": "/data/images/img001.jpg",
  "target": "/data/labels/img001.txt",
  "test_index": 0
}
```

Bbox sidecar format:

```text
class xmin ymin xmax ymax
```

Rules:

- class `0` is background and is not used for positive boxes;
- positive labels must be in `1..nclasses-1`;
- boxes are absolute pixel coordinates;
- empty bbox files produce empty `boxes` and `labels` tensors;
- invalid boxes raise `dataset_contract_error`;
- transforms must preserve box/image consistency.

Each `__getitem__` returns:

```python
image, target
```

where `image` is a `[3, H, W]` tensor and `target` contains:

```python
{
    "boxes": FloatTensor[N, 4],
    "labels": Int64Tensor[N],
    "image_id": Int64Tensor[1],
    "area": FloatTensor[N],
    "iscrowd": UInt8Tensor[N],
}
```

This follows the torchvision detection finetuning contract, where boxes are
`[x0, y0, x1, y1]` and label `0` is background.

### Training Loop Contract

The template runtime drives training roughly as:

```python
for iteration in range(start_iteration, max_iterations):
    batch = next(train_iterator)
    metrics = worker.train_one_iteration(batch)
    reporter.status(iteration=iteration, phase="train")
    for name, value in metrics.items():
        reporter.metric(name, value, iteration=iteration)
    if iteration % test_interval == 0:
        for test_index, test_set in enumerate(test_sets):
            worker.evaluate(test_set)
    if cancellation.requested:
        break
```

`train_one_iteration` must:

- put model in train mode;
- move images and targets to the selected device;
- call the model to get torchvision loss dict;
- sum losses into `train_loss`;
- run backward and optimizer step;
- support gradient accumulation by reporting only after the optimizer step;
- return finite scalar metrics only.

Required per-optimizer-step metrics:

- `train_loss`
- every finite scalar from torchvision loss dict, for example:
  - `loss_classifier`
  - `loss_box_reg`
  - `loss_objectness`
  - `loss_rpn_box_reg`
- `learning_rate`

If AMP is enabled, the worker uses `torch.cuda.amp` or the current supported
PyTorch equivalent internally. AMP scaler state is part of checkpoints.

### Evaluation Contract

`evaluate(test_set)` must:

- set model to eval mode;
- report:
  - `test_active = 1`
  - `test_set_index`
  - `test_sets_total`
  - `test_processed`
  - `test_total`
- emit `test_active = 0` when evaluation completes;
- emit metrics with `_testX` suffix for multiple test sets when an evaluator is
  configured;
- emit sampled `test_predictions` for visual result sinks.

The torchvision detector worker emits detection metrics for every test set:

- `map`
- `map-05`
- `map-50`
- `map-90`

Metric names are suffixed with `_testX` when reported by the worker, for example
`map-50_test0`. It also emits test progress and sampled predictions for visual
result sinks. A later common evaluator slice can move this calculation out of
the worker without changing the reporting channel.

### Prediction Schema

Sampled training visualizations use the DeepDetect detection output shape:

```json
{
  "sample_index": 12,
  "imgsize": {
    "width": 1920,
    "height": 1080
  },
  "classes": [
    {
      "cat": "ring",
      "prob": 0.91,
      "bbox": {
        "xmin": 100.0,
        "ymin": 50.0,
        "xmax": 200.0,
        "ymax": 180.0
      }
    }
  ]
}
```

Coordinates are in the `imgsize` coordinate system. If predictions are produced
on resized model inputs, the worker must map them back to the original image
coordinate system or set `imgsize` to the exact prediction coordinate size.

The worker `/predict` method returns the internal supervised-output connector
shape consumed by the C++ bridge:

```json
{
  "results": [
    {
      "uri": "/data/images/img001.jpg",
      "loss": 0.0,
      "probs": [0.91],
      "cats": ["1"],
      "bboxes": [
        {
          "xmin": 100.0,
          "ymin": 50.0,
          "xmax": 200.0,
          "ymax": 180.0
        }
      ]
    }
  ]
}
```

## LLM Worker Authoring Rules

These rules are intended for future Codex or agent skills that port external
academic repositories into DeepDetect-compatible workers.

Agents should:

- preserve the class contract;
- avoid implementing protocol/socket code in model workers;
- isolate external repository code behind:
  - `build_model`
  - dataset adapter
  - loss/metric adapter
  - prediction adapter
- keep optimizer, scheduler, AMP, and distributed wrappers inside worker code;
- report scalar losses every optimizer step through `reporter.metric`;
- check cancellation in training and evaluation loops;
- write checkpoints through `save_checkpoint`;
- convert all predictions to DeepDetect schemas before returning;
- keep label mapping explicit and preserve background label `0`;
- write enough metadata to the repository to reproduce preprocessing and class
  mappings.

Agents should not:

- silently change DeepDetect class ids;
- use hard-coded absolute dataset paths;
- report non-finite metrics;
- report per-microbatch metrics as optimizer-step metrics;
- let external code own process exit or signal handling;
- write predictions in a framework-specific schema without an adapter.

Porting checklist:

- worker class imports successfully;
- required methods exist;
- dataset returns valid tensors and targets;
- labels preserve DeepDetect class semantics;
- losses are finite and reported every optimizer step;
- evaluation reports test progress and test-set-specific metrics;
- predictions render with existing DeepDetect visual result tooling;
- checkpoints include model, optimizer, scheduler, scaler, iteration, and class
  mapping;
- cancellation exits cleanly;
- dependency and model-code failures are typed.

## Failure Handling

The supervisor and runtime classify failures with stable categories.

Configuration and launch:

- `configuration_error`: invalid DeepDetect or worker parameters.
- `worker_launch_error`: Python executable, script, permissions, socket, or
  launcher setup failure.
- `dependency_error`: missing import, incompatible PyTorch/torchvision/CUDA
  version, missing optional evaluator.
- `worker_contract_error`: missing required class/method or incompatible method
  signature.

Data and protocol:

- `connector_error`: invalid connector config or unresolved data path.
- `dataset_contract_error`: dataset sample cannot become a valid model target.
- `protocol_error`: malformed message, unsupported protocol version, schema
  mismatch.
- `metric_contract_error`: metric is non-scalar, non-finite, or unserializable.
- `prediction_contract_error`: prediction cannot be converted to DeepDetect
  output schema.

Runtime:

- `training_error`: Python exception in training or evaluation.
- `distributed_error`: rank failure, rendezvous failure, or nonzero launcher
  status.
- `resource_error`: OOM, disk full, GPU unavailable, process killed by signal.
- `timeout_error`: heartbeat or request timeout.
- `cancelled`: user-requested termination.
- `internal_error`: uncategorized supervisor/runtime bug.

Each failure payload includes:

- category;
- message;
- retryable flag;
- traceback summary when available;
- worker exit code or signal when available;
- rank id for distributed jobs;
- stderr/log tail.

Mapping to DeepDetect API:

- input and dataset failures map to service input errors when they are caused
  by user data;
- configuration and contract failures map to service bad request;
- train/predict conflict uses the existing conflict response;
- resource failures map to resource exhaustion where possible;
- crashes, protocol failures, and uncategorized runtime failures map to
  internal mllib errors.

## Model Repository Contract

The worker writes or updates:

- `config.json` or `config.yaml`: effective worker config.
- `connector_manifest.json`: data and preprocessing contract.
- `class_mapping.json`: DeepDetect class id to label mapping.
- `checkpoint-latest.pt`: model and optimizer state.
- `checkpoint-N.pt`: periodic checkpoints.
- `best_model.txt`: best checkpoint metadata when a best metric is configured.
- `metrics.jsonl`: optional worker-side metric mirror.
- `exports/`: optional deployment artifacts.
- `visual_results/` or existing DeepDetect visual result paths when requested.

The C++ supervisor remains responsible for the DeepDetect service config blob
and normal API status files.

## Distributed Training

V1 is single-machine only. Multi-GPU is supported by Python launchers.

Supported launcher modes:

- `python`: single process.
- `torchrun`: one process per local rank.
- `accelerate`: optional, when configured by the worker environment.

The supervisor starts the configured launcher and treats the whole launcher
process tree as one DeepDetect training job. Rank-specific errors are reported
through `distributed_error` with rank id when known.

Only rank 0 should report user-visible metrics, artifacts, and checkpoints
unless the worker explicitly declares otherwise.

## Testing Requirements

Specification-level tests:

- `mllib: "pytorch"` service creation routes separately from `mllib: "torch"`.
- invalid worker class produces `worker_contract_error`.
- missing worker script produces `worker_launch_error`.
- missing torchvision dependency produces `dependency_error`.
- malformed bbox target produces `dataset_contract_error`.
- non-finite metric produces `metric_contract_error`.
- malformed prediction produces `prediction_contract_error`.

Integration tests:

- dummy worker starts, trains for a few iterations, and emits metrics.
- torchvision template emits per-optimizer-step losses.
- empty bbox files produce empty detection targets.
- multiple test sets emit `_test0`, `_test1` metrics.
- test progress includes processed and total sample counts.
- sampled detection predictions render with existing visualization tools.
- cancellation terminates the process tree and reports `cancelled`.
- worker crash includes traceback summary and stderr tail.
- fake distributed rank failure reports `distributed_error`.

Compatibility tests:

- existing `mllib: "torch"` tests are unchanged.
- existing JSON API status and error conventions remain recognizable.
- DeepDetect CLI metric sinks can consume metrics emitted by the new backend
  once surfaced through training status.

## Phased Implementation

Phase 1:

- add `mllib: "pytorch"` routing behind `USE_PYTORCH_WORKER`;
- implement the C++ worker supervisor and length-prefixed JSON protocol;
- implement the Python protocol runtime;
- ship a dummy worker that imports `torch`, emits status/metrics, handles
  cancellation, and returns a fixed prediction;
- validate async train/status/cancel, failure mapping, and basic predict
  plumbing without real dataset/model code.

Phase 2:

- generate full connector manifests;
- add the normative torchvision detection worker template;
- define the class-based worker authoring SDK for real trainers;
- integrate worker metrics into existing training status and history;
- surface `test_predictions`;
- support saved visual result artifacts;
- add CLI profile support for the new backend.

Phase 3:

- add CPU tensor reference input path for selected connectors;
- keep path-backed manifest loading as fallback.

Phase 4:

- prototype GPU tensor references and stream/lifetime management;
- evaluate whether zero-copy matters for target workloads.
