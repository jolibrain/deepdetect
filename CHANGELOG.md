# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.17.0](https://github.com/jolibrain/deepdetect/compare/v0.16.0...v0.17.0) (2021-05-10)


### Features

* introduce predict output parameter ([c9ee71a](https://github.com/jolibrain/deepdetect/commit/c9ee71af5167ee24fa285fb0812bbc267a72970b))
* use DTO for NCNN init parameters ([2ee11f0](https://github.com/jolibrain/deepdetect/commit/2ee11f07b7f8c60ca3baefd9b09f20df30a45863))
* **ml:** data augmentation for object detection with torch backend ([95942b9](https://github.com/jolibrain/deepdetect/commit/95942b9174170099ca1e2a4b87a5e5f758943a37))
* **ml:** Visformer architecture with torch backend ([40ec03f](https://github.com/jolibrain/deepdetect/commit/40ec03f77d0107a4b758b1103d265fffc904812a))
* **torch:** add batch size > 1 for detection models ([91bde66](https://github.com/jolibrain/deepdetect/commit/91bde66ba01e0a0f2e75d32062583c4fe018022b))
* **torch:** image data augmentation with random geometric perspectives ([d163fd8](https://github.com/jolibrain/deepdetect/commit/d163fd88bbc966aedc1a213f25e1d9d16664f822))


### Bug Fixes

* **build:** docker builds with tcmalloc ([6b8411a](https://github.com/jolibrain/deepdetect/commit/6b8411a3989f8131c6745c921fe96629246570d3))
* **doc:** api traced models list ([342b909](https://github.com/jolibrain/deepdetect/commit/342b909b74d6e24f1b0c440086e8ce8057e6fd83))
* **graph:** loading weights from previous model does not fail ([5e7c8f6](https://github.com/jolibrain/deepdetect/commit/5e7c8f6c8a0ddcd1cc2e2bd93278e2262a6d80ff))
* **torch:** fix faster rcnn model export for training ([cbbbd99](https://github.com/jolibrain/deepdetect/commit/cbbbd99cb1fffe5fda5ff7aeeef47a853c35e615))
* **torch:** retinanet now trains correctly ([351d6c6](https://github.com/jolibrain/deepdetect/commit/351d6c6aafde52d821ab50853a37595438778556))
* **torch:** torchvision models can be used with greyscale images ([2921050](https://github.com/jolibrain/deepdetect/commit/292105084e4c37b46f849347e659b38a0df872a9))
