# DeepDetect Agent Instructions

## DeepDetect CLI Training And Monitoring

These instructions are for agent LLMs operating in this repository. They
describe how to launch DeepDetect CLI training jobs, monitor them through
structured outputs, and assess model quality through scalar metrics and saved
visual results.

### Command Style

- Prefer the in-repo module entrypoint while developing:

  ```shell
  python3 -m deepdetect.cli.main ...
  ```

  Use the installed `deepdetect ...` command only when the target wheel or
  environment is intentionally under test.
- If the local development environment is expected, activate it before running
  Python commands, for example:

  ```shell
  source ~/venv/bin/activate
  ```

- Training command shape is task first:

  ```shell
  python3 -m deepdetect.cli.main train yolox ...
  python3 -m deepdetect.cli.main train segformer ...
  python3 -m deepdetect.cli.main job status REPOSITORY
  python3 -m deepdetect.cli.main inspect models
  ```

- Prefer YAML configs for repeatable runs. CLI flags override config values,
  and `--set key=value` overrides both. The CLI writes the truly applied
  training config to `REPOSITORY/config.yaml`; treat that file as the delivered
  run configuration.
- Use `--gpu --gpuid ID [ID ...]` to select GPUs. Comma-separated values are
  accepted, for example `--gpuid 0,1`. Use `--gpuid -1` for all GPUs.
- Use `--iter-size N` for gradient accumulation.
- Use `--test-data PATH [PATH ...]` for multiple test sets. Metrics from the
  first set are suffixed with `_test0`, the second with `_test1`, and so on.
- Use YAML-only fields for detailed augmentation and class weighting instead
  of adding ad hoc command-line flags:

  ```yaml
  augmentation:
    mirror: true
    rotate: true
    crop_size: 384
  class_weights: [1.0, 0.5, 2.0]
  ```

### Training Examples

Detection with YOLOX:

```shell
python3 -m deepdetect.cli.main train yolox \
  --train-data train.txt \
  --test-data test0.txt test1.txt \
  --weights yolox.pt \
  --repository runs/yolox-model \
  --nclasses 2 \
  --iterations 10000 \
  --test-interval 100 \
  --batch-size 4 \
  --iter-size 1 \
  --gpu --gpuid 0 \
  --terminal verbose \
  --output-format jsonl \
  --visdom \
  --visdom-results
```

Segmentation with SegFormer:

```shell
python3 -m deepdetect.cli.main train segformer \
  --train-data train.txt \
  --test-data test0.txt test1.txt \
  --weights segformer.pt \
  --repository runs/segformer-model \
  --nclasses 13 \
  --iterations 10000 \
  --test-interval 100 \
  --batch-size 2 \
  --gpu --gpuid 0 \
  --terminal verbose \
  --output-format jsonl \
  --visdom \
  --visdom-results
```

Use `--dataset-check none` only when the dataset was already validated or when
fast startup is more important than early data errors.

### Agent Monitoring

- For machine monitoring, use `--terminal verbose --output-format jsonl`.
  Parse stdout as newline-delimited JSON events.
- Do not scrape `--terminal live` as an automation interface. Live mode is a
  human terminal dashboard and may redraw in place. If live mode is requested
  in a non-TTY context, the CLI falls back to verbose JSONL.
- Important stdout events include:
  - `run_started`: run directory, repository, command, and initial metadata.
  - `dataset_check`: dataset validation status.
  - `training_status`: current DeepDetect async status and `measure` payload.
  - `metric`: extracted scalar metric event.
  - `sink_warning`: non-fatal sink or visualization warning.
  - `run_finished`: final run status.
- Read durable run artifacts directly:
  - `REPOSITORY/run.json`: latest run manifest, job id, status, and latest
    status body.
  - `REPOSITORY/metrics.jsonl`: persisted metric stream.
  - `REPOSITORY/config.yaml`: effective config after all overrides.
- Use job status for a persisted status snapshot:

  ```shell
  python3 -m deepdetect.cli.main job status REPOSITORY --output-format json
  ```

- Track losses such as `train_loss`, `total_loss`, `cls_loss`, `conf_loss`,
  `iou_loss`, and `l1_loss`.
- Track evaluation metrics such as `map`, `map-05`, `map-50`, `map-90`,
  `acc`, `meaniou`, and false-positive metrics. With multiple test sets,
  compare suffixed metrics like `map-50_test0` and `map-50_test1`.
- Treat `elapsed_time_ms`, `test_active`, `test_processed`, and similar
  progress fields as status/progress signals, not model quality metrics.

### Visdom Monitoring

- Enable Visdom with `--visdom`. Result images are enabled by default when
  Visdom is enabled; keep `--visdom-results` explicit in agent-authored
  commands for clarity.
- `--visdom-results-count` controls the number of sampled result images per
  test set. The default is `10`; use `0` to disable result image uploads.
- `--visdom-results-seed` controls deterministic sampling. Samples are
  refreshed at every test interval.
- The Visdom environment is the run name. The CLI does not delete Visdom
  environments at the end of training. Use `--visdom-save` when the Visdom
  server should persist the environment.
- For multiple test sets, the same metric is plotted in the same Visdom window
  with separate traces named `test0`, `test1`, and so on.
- If Visdom is unreachable and `--visdom-offline-ok` is enabled, training
  continues and emits `sink_warning` events. If visual monitoring is required,
  inspect these warnings immediately.

### Visual Result Assessment

- Detection result images draw predicted bounding boxes, class labels, and
  confidences on the original image pixels. Box colors represent classes.
- Segmentation result images draw class overlays on the original image pixels.
- Visdom receives resized display copies with the largest side capped for
  practical viewing. Durable repository artifacts are the source of truth for
  review after the run.
- Saved visual artifacts are written under:

  ```text
  REPOSITORY/visdom-results/iteration-XXXXXX/testN/
  ```

  Each rendered sample has an image file and a JSON file containing the source
  image path, sample index, and raw prediction payload.
- When metrics are high but rendered results look wrong, inspect the saved
  image/JSON pairs before changing model settings. Check test-set ordering,
  sample indices, coordinate sizes, class ids, confidence thresholds, and
  whether the backend is shuffling test data.

### Troubleshooting Rules

- If no visual samples appear, verify `--visdom`, `--visdom-results`,
  `--visdom-results-count > 0`, Visdom connectivity, and the presence of
  backend `test_predictions` in training status.
- If live terminal test progress is missing, fall back to `training_status`,
  `run.json`, and `metrics.jsonl`; these are the automation-grade sources.
- If backend status fields, test prediction payloads, or C++ training behavior
  were changed, rebuild the native DeepDetect library or wheel before testing
  the CLI.
- Keep CLI changes and monitoring investigations scoped. Do not delete or
  reset existing run repositories unless the user explicitly asks.
