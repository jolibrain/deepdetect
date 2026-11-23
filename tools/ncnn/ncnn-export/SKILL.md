---
name: ncnn-export
description: Export a DeepDetect image model or PyTorch worker checkpoint to a native NCNN service, diagnose PNNX conversion failures, and implement or reuse CPU operator workarounds.
---

# Export a model to NCNN

Work from the DeepDetect checkout. The [backend guide](../../../docs/ncnn.md) gives the pinned converter, service contracts and CLI commands. Use the worker's Python environment on the export host. The target runs the C++ backend.

## Establish the contract

- Identify the checkpoint, evaluation/deploy transforms, one FP32 image input, fixed dimensions, channel order, resize and normalization. Compare the tensor and final predictions on the same image. Record class offsets, coordinate units and when postprocessing runs.
- For a worker, reuse a matching saved ONNX graph or provide an importable `(checkpoint_paths, inference_configuration)` callable returning a CPU evaluation module, tuple of example inputs and input/output names. A state dict needs the original architecture. The [worker skill](../../../bindings/python/deepdetect/pytorch_worker/SKILL.md) describes the checkpoint and provider interfaces. Keep NCNN-specific conversion out of training.
- Use `python3 -m deepdetect.cli.main export SOURCE --format ncnn --output NEW_DIR`. Standalone ONNX and TorchScript sources require `--config`; worker runs use their recorded config and selected checkpoint. The [exporter](../../../bindings/python/deepdetect/cli/ncnn_export.py) stages files, verifies PNNX and publishes the bundle. Set `service_mllib.ncnn_output` to the actual output contract.

## Diagnose and lower unsupported operations

Compare the source graph, PNNX intermediate with shapes and attributes, emitted NCNN param and native load/extract result. A successful PNNX exit alone can leave an unregistered layer or an incorrect axis. Compare the intermediate before and after a suspect operator. Keep conversion isolated to the export host.

- For fixed-K FP32 ranking, use `DeepDetectTopK` in [ops](../../../src/backends/ncnn/ops/). The [TopK lowering](../../../bindings/python/deepdetect/cli/ncnn_topk.py) recovers attributes before PNNX drops them. NCNN parameters are `0=K`, `1=axis`, `2=largest` and `3=outputs` (0 values, 1 indices, 2 both). Axes index the physical NCNN tensor outermost first; -1 means width. Optional parameters 4 and 5 check the expected axis length and inner stride. Indices are zero-based int32. Sorted results and lower-index ties are deterministic. Confirm the consumer accepts the index representation.
- Put reusable runtime operators in `src/backends/ncnn/ops/` as CPU `ncnn::Layer` implementations. Give exported types a `DeepDetect` prefix and register them in `ops.cc` before model loading. List new translation units in `src/CMakeLists.txt`. Validate parameter values, tensor rank, packing, allocation and output arity. Respect NCNN channel padding and the extractor's allocator. Keep per-request state in `forward`.
- Put export-time lowering in the Python CLI, retaining source attributes, blob connections and binary weight order. A generic op should not depend on a model's layer names or class count. Keep model-specific topology, indexing conventions and repairs in the adapter. The [RT-DETRv4 adapter](../../../bindings/python/deepdetect/cli/ncnn_rtdetrv4.py) shows how to join converted sections into one pair, including explicit splits for shared blobs. Its attention sum-axis repair applies to the verified S graph and pinned PNNX version.
- For unsupported semantics, surface the exact op, dtype or layout and preserve the source and previous export. Distinguish a converter failure from a native numerical mismatch.

## Verify the delivered bundle

Exercise service creation and prediction through the embedded runtime or HTTP. Use `--test-inference` with a matching NCNN-enabled binding for a relocation smoke test. Check scores and decoded geometry against the original PyTorch model with identical preprocessing. Add a small operator test for axes, indices, ties or shape errors that the full model would conceal. Run `ut_ncnnapi` and `bindings/python/tests/test_cli_ncnn_export.py` when changing conversion or runtime behavior.
