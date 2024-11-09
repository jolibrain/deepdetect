# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.26.0](https://github.com/jolibrain/deepdetect/compare/v0.25.0...v0.26.0) (2024-11-09)


### Features

* generate mask for diffusion models in trt ([5c16ad0](https://github.com/jolibrain/deepdetect/commit/5c16ad0b4afec2a294769f883d201ace4786552c))
* opencv optional build ([ce9b9a7](https://github.com/jolibrain/deepdetect/commit/ce9b9a74a02cecf90a8a711c5fd70f65d7f2e7f9))
* **output:** add false positive metrics for detection ([bec49c4](https://github.com/jolibrain/deepdetect/commit/bec49c43210360f8118e6e80418711f85f602a0c))
* **torch:** add JIT FusionStrategy selection ([d2331be](https://github.com/jolibrain/deepdetect/commit/d2331be5fff65cff8eafc3f379b3fd5279f188d8))
* **torch:** Added script to trace HuggingFace Transformers CLIP models based on https://huggingface.co/openai/clip-vit-large-patch14-336 ([c2373ea](https://github.com/jolibrain/deepdetect/commit/c2373ea60144560ded246354cf2a5adf16e49366))


### Bug Fixes

* **torch:** errors in input connector are caught correctly ([ac09c52](https://github.com/jolibrain/deepdetect/commit/ac09c52af467a2863119b555f8a40dd244f65a6e))
* **torch:** segmentation with test_batch_size > 1 ([0d8d3da](https://github.com/jolibrain/deepdetect/commit/0d8d3da83fb673ddbdb6c8bbf32b98f1114d8b30))
* Update ut-chain.cc to pass tests ([87ca92c](https://github.com/jolibrain/deepdetect/commit/87ca92cbd3b0f733fb8686841cf2215524100101))
