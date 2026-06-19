# External PyTorch Worker Workspace

`extern/pytorch_workers/<model_slug>/` is a local workspace for generated or
hand-authored PyTorch worker adapters. The directory is ignored by git so model
ports can depend on external repositories without adding model-specific code to
DeepDetect core.

Select an adapter by setting `service_mllib.entrypoint` and
`service_mllib.class` in a CLI YAML config or API service parameters. Worker
classes must implement the public `DeepDetectWorker.configure/train/predict`
contract, usually by subclassing
`deepdetect.pytorch_worker.builtin.vision.detection.base.DetectionTrainingWorkerBase`
for object detection.

Each generated adapter should include:

- `worker.py`: the worker class selected by `service_mllib.entrypoint`;
- `config.yaml`: a repeatable CLI config for the adapter;
- `manifest.json`: upstream repository, checkout, license, dependencies,
  selected entrypoint, and generation notes;
- optional notes for manual validation.
