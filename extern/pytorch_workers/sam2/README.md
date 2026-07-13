# SAM2 Image Inference Worker

This external PyTorch worker exposes static-image SAM2 inference through the
DeepDetect CLI. It does not support training, video tracking, point prompts,
or mask prompts.

Install its runtime into the Python interpreter used by the worker. The
dependency is deliberately not bundled in DeepDetect wheels:

```shell
SAM2_BUILD_CUDA=0 python3 -m pip install -r extern/pytorch_workers/sam2/requirements.txt
```

`SAM2_BUILD_CUDA=0` avoids the optional custom CUDA post-processing extension.
The worker still runs without it. The selected `sam2==1.1.0` PyPI package is an
external runtime choice; verify it against local deployment policy before use.

Download a matching checkpoint separately and provide it explicitly. The
default `tiny` variant expects `sam2.1_hiera_tiny.pt`.

```shell
PYTHONPATH=bindings/python python3 -m deepdetect.cli.main infer sam2 image.jpg \
  --config extern/pytorch_workers/sam2/config.yaml \
  --weights /models/sam2.1_hiera_tiny.pt \
  --repository runs/sam2-tiny \
  --gpu --gpuid 0 \
  --output results
```

Without `--bbox-files`, the worker runs automatic mask generation. The output
contains an overlay and one original-resolution binary PNG per accepted mask.
The PNG values are exactly `0` and `1`; they are not a semantic class map.
The prediction response contains the same masks as uncompressed COCO RLE under
`classes[].mask`.

For box prompts, provide one sidecar for each input image:

```text
cls xmin ymin xmax ymax
```

Coordinates are source-image pixels. `cls` is preserved in prediction metadata
but SAM2 does not classify masks. An empty sidecar produces no masks for that
image.

`--batch-size` performs true batched image-encoder inference for bbox-prompted
images through SAM2's `set_image_batch` API. Choose the largest value that fits
GPU memory. Automatic mask generation remains image-serial because the
upstream automatic generator only exposes `generate(image)`; its
`points_per_batch` setting controls prompt batching within each image.

```shell
PYTHONPATH=bindings/python python3 -m deepdetect.cli.main infer sam2 image1.jpg image2.jpg \
  --config extern/pytorch_workers/sam2/config.yaml \
  --weights /models/sam2.1_hiera_tiny.pt \
  --bbox-files image1.txt image2.txt \
  --repository runs/sam2-boxes \
  --gpu --gpuid 0 \
  --output results
```

Set `service_mllib.sam2.variant` to `small`, `base_plus`, or `large` only with
the matching SAM2.1 checkpoint. Automatic generator options, including
`max_masks` (`0` means unlimited), are configured below
`service_mllib.sam2.automatic`.

## Manifest Runner

`tools/run_sam2_manifest.py` runs bbox-prompted inference for a two-column
`paths.txt` manifest while keeping a single DeepDetect SAM2 service alive. Each
non-empty row must contain an image path and its bbox sidecar path. Relative
paths resolve from the parent of the manifest directory, which matches a
`trainA/paths.txt` layout with sibling dataset directories:

```text
ring/img/sample.png ring/bbox/sample.txt
```

```shell
PYTHONPATH=bindings/python python3 tools/run_sam2_manifest.py \
  /datasets/trainA/paths.txt results/sam2 \
  --weights /models/sam2.1_hiera_tiny.pt \
  --gpu --gpuid 0
```

The runner validates that every sidecar contains exactly one
`cls xmin ymin xmax ymax` row, writes masks under `<output>/masks/` and visual
overlays as JPEG files under `<output>/overlays/`, shows a `tqdm` progress bar
as overlays complete, and records completed artifacts in
`<output>/.sam2-manifest-run/completed.jsonl`. A later invocation resumes only
entries whose recorded overlay is still present. Use `--overwrite` to clear
that completion state and process every entry again; it does not delete output
artifacts. Use `--data-root` when relative paths are rooted somewhere other
than the manifest's parent directory. The selected Python interpreter must
already have the DeepDetect bindings and SAM2 runtime installed; pass a
different one through `--python` when needed.

Masks are binary by default. Pass `--class-mask-values` to replace every
nonzero mask pixel with the sidecar's `cls` value. Background remains `0`, so
this mode accepts class IDs from `1` through `255`. Enabling it on a resumed
output converts completed binary masks in place without rerunning SAM2. Use
`--overwrite` without `--class-mask-values` to regenerate binary masks after a
class-valued run.

For outputs generated before class-valued masks were supported, repair the
existing PNG files in place by joining the original image/bbox manifest with
the generated image/mask manifest:

```shell
python3 tools/fix_sam2_class_masks.py \
  /datasets/trainA/paths.txt /results/sam2/paths.txt --dry-run
python3 tools/fix_sam2_class_masks.py \
  /datasets/trainA/paths.txt /results/sam2/paths.txt
```

The repair is atomic and idempotent. `--limit` is available for verification.
Output rows whose source image is missing or absent from the input manifest are
counted under `skipped_missing_images`. Missing output mask files are counted
and removed from the output `paths.txt` atomically. In `--dry-run` mode the
removal is reported without modifying the manifest.
When SAM2 is still appending output rows, run the repair again after inference
finishes.

At startup, `<output>/paths.txt` is reconciled with valid resume state and then
kept open in append mode while inference runs. Each completed image is flushed
to the file immediately as one dataset pair:

```text
/absolute/path/to/image.png /output/masks/image_sam2_mask_0001.png
```

It contains only entries with an existing binary mask, including the subset
produced by `--limit`, and can be used directly as an image/mask dataset list.
Existing completed artifacts from the former flat output layout are migrated
into `masks/` and `overlays/` when the runner next starts. A per-output lock
rejects concurrent runner invocations that would otherwise share staging and
completion files.
