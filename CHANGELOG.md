# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.23.0](https://github.com/jolibrain/deepdetect/compare/v0.22.1...v0.23.0) (2022-09-28)


### Features

* add crnn resnet native template ([ec1f8ad](https://github.com/jolibrain/deepdetect/commit/ec1f8ad4640ab4ef9c0109b101b4d1ba1e10f869))
* add deepdetect version to config variables for external projects ([be79e54](https://github.com/jolibrain/deepdetect/commit/be79e543a5f7c73949e1d5fbe97a4d2890548c3c))
* **dlib:** update dlib backend ([12d181f](https://github.com/jolibrain/deepdetect/commit/12d181f5bccbbea9473853475086781f439f29e6))
* **torch:** add multilabel classification ([90d536e](https://github.com/jolibrain/deepdetect/commit/90d536e60bd5a2b748da6f51305df4332d984977))
* **torch:** allow multigpu for traced models ([6b3b9c0](https://github.com/jolibrain/deepdetect/commit/6b3b9c08b2590456cfa19f6344f8569291950bea))
* **torch:** best model is computed over all the test sets ([fbedf80](https://github.com/jolibrain/deepdetect/commit/fbedf80605a8228424a39b7ce99ed2635572e20f))
* **torch:** update torch to 1.12 ([7172314](https://github.com/jolibrain/deepdetect/commit/717231409f341ee871a4b3baa53a4bfb74e7c7d6))
* **yolox:** export directly from trained dd repo to onnx ([a612539](https://github.com/jolibrain/deepdetect/commit/a612539cee8d49a2e5a68351caa958013a7163b4))


### Bug Fixes

* adamw default weight decay with torch backend ([eb0cf83](https://github.com/jolibrain/deepdetect/commit/eb0cf83d8eabb6481b57a90a7db4313d0a5fc399))
* add missing headers in predict_out.hpp ([b23298f](https://github.com/jolibrain/deepdetect/commit/b23298f6b8ebc888eba28e0f2333f6a59ddeff1c))
* **docker:** add libcupti to gpu_torch docker ([1a5cd09](https://github.com/jolibrain/deepdetect/commit/1a5cd090d75f2fa4a0626a292f5d5f2a4de878c6))
* enable caffe chain with DTO & custom actions ([d3e722e](https://github.com/jolibrain/deepdetect/commit/d3e722ed0f3d7cbccdd645c4c147b824e8063020))
* exported yolox have the correct number of classes ([4dac269](https://github.com/jolibrain/deepdetect/commit/4dac269a0496d52026c4d82dc9514e3790237e02))
* missing ifdef ([e8a70cf](https://github.com/jolibrain/deepdetect/commit/e8a70cf5f9a39cdf9275f0874f9ff716913e3872))
* missing path to cub headers in tensorrt-oss build for jetson nano ([00df9fd](https://github.com/jolibrain/deepdetect/commit/00df9fdfce78af7a87ce6d515d80a653d47a9ded))
* **oatpp:** oatpp-zlib memory leak ([fccd9a6](https://github.com/jolibrain/deepdetect/commit/fccd9a622dea9bd3bbbf6e40a12ba05dd9f57e80))
* prevent a buggy optimization in traced fasterrcnn ([dab88ca](https://github.com/jolibrain/deepdetect/commit/dab88cae82f76b65012ddc23c5546f79c719de08))
* reload best metric correctly after resume ([c15c502](https://github.com/jolibrain/deepdetect/commit/c15c502319085f062b018ac26263d7b0790ffed0))
* **torch:** OCR predict with native model ([24aa37c](https://github.com/jolibrain/deepdetect/commit/24aa37c79448738f753bb22721fd75b29a5b6563))
* **torch:** remove invalid argument ([50e3f0b](https://github.com/jolibrain/deepdetect/commit/50e3f0ba60b7fcd1b511d6b5f3331137a81f57a8))
