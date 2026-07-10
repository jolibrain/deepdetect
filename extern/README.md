# External PyTorch Worker Workspace

`extern/pytorch_workers/<model_slug>/` is a local workspace for generated,
hand-authored, or tracked PyTorch workers. Most model-specific worker
directories are ignored by git by default so experiments can depend on external
repositories without adding model code to DeepDetect core. Some workers are
explicitly unignored and tracked when their model code is intentionally
self-contained in this tree.

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
- `README.md`: quickstart train and inference commands, required upstream
  paths, checkpoint expectations, environment variables, and model-specific
  conversion notes;
- optional notes for manual validation.

Tracked self-contained workers must not require an externally cloned upstream
repository at runtime. They should vendor or adapt the required model code under
their own worker directory and document the upstream license and provenance in
their manifest and README.
