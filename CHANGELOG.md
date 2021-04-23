# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.16.0](https://github.com/jolibrain/deepdetect/compare/v0.15.0...v0.16.0) (2021-04-23)


### Features

* **torch:** add confidence threshold for classification ([0e75d88](https://github.com/jolibrain/deepdetect/commit/0e75d88fb949fc2e0e23ed744b5752df8b581d5a))
* **torch:** add more backbones to traced detection models ([f4d05e1](https://github.com/jolibrain/deepdetect/commit/f4d05e1ea9f419832cc4c63c5367f7615ef22b2f))
* **torch:** allow FP16 inference on GPU ([705d3d7](https://github.com/jolibrain/deepdetect/commit/705d3d77c8f325d7707cafb422e948b9cc3ac7f7))
* **torch:** madgrad optimizer ([0657d82](https://github.com/jolibrain/deepdetect/commit/0657d82cd05d575cb6d45c2f122946626d7457a8))
* **torch:** training of detection models on backend torch ([b920999](https://github.com/jolibrain/deepdetect/commit/b9209991a4e44a45d9bacaed182fd7ecacaed369))


### Bug Fixes

* **torch:** default gradient clipping to true when using madgrad ([5979019](https://github.com/jolibrain/deepdetect/commit/5979019c27cb5e84ddcb38f40bbd962c32d7003f))
* remove dirty git flag on builds ([6daa4f5](https://github.com/jolibrain/deepdetect/commit/6daa4f5343fb31afbf0efd7330da7513b652e539))
* services names were not always case insentitive ([bee3183](https://github.com/jolibrain/deepdetect/commit/bee318356c2bf056247073c73f580016970f379f))
* **chains:** cloning of image crops in chains ([2e62b7e](https://github.com/jolibrain/deepdetect/commit/2e62b7e6f3f75d2de08e8c6088c5a2da7b320d39))
* **ml:** refinedet image dimensions configuration via API ([20d56e4](https://github.com/jolibrain/deepdetect/commit/20d56e4ac6ab4691c32187137b520996160c8d59))
* **TensorRT:** fix some memory allocation weirdness in trt backend ([4f952c3](https://github.com/jolibrain/deepdetect/commit/4f952c3fbc2f8da03ebc66644e125576d9b12fee))
* **timeseries:** throw if no data found ([a95e7f9](https://github.com/jolibrain/deepdetect/commit/a95e7f936c35cbe5cf24779fe5e899667b7f6e6c))
* **torch:** allow partial or mismatching weights loading only if finetuning ([23666ea](https://github.com/jolibrain/deepdetect/commit/23666ea49ece302477e1f2d8f88edc41366ff213))
* **torch:** Fix underflow in CSVTS::serialize_bounds ([c8b11b6](https://github.com/jolibrain/deepdetect/commit/c8b11b66b4b264ac16b3b2357fbd66293c01f99d))
* **torch:** fix very long ETA with iter_size != 1 ([0c716a6](https://github.com/jolibrain/deepdetect/commit/0c716a60b2742c70ad715ad4e9b23a3f4d035a77))
* **torch:** parameters are added only once to solver during traced model training ([86cbcf5](https://github.com/jolibrain/deepdetect/commit/86cbcf5f41f868a6472bb3df46015db34b61f1a2))
* **torch:** select correct mltype ([e674fcc](https://github.com/jolibrain/deepdetect/commit/e674fcce4812cf499a6e2eaaa8f98b34d35d5ffa))
