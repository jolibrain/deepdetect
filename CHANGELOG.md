# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.19.0](https://github.com/jolibrain/deepdetect/compare/v0.18.0...v0.19.0) (2021-09-06)


### Features

* add DTO schemas to swagger automatic doc ([9180ff4](https://github.com/jolibrain/deepdetect/commit/9180ff4b8f0d71995bffff58cd497121ae3ea98a))
* add z-normalisation option ([82d7cc5](https://github.com/jolibrain/deepdetect/commit/82d7cc57011180d2836efffed919f68200d1ff24))
* **dto:** add custom dto vector type ([01222db](https://github.com/jolibrain/deepdetect/commit/01222db2bc8663a959de57e8c27a715d97add163))
* **torch:** add ADAMP variant of adam in RANGER (2006.08217) ([e26ed77](https://github.com/jolibrain/deepdetect/commit/e26ed77744e302c8fbae597f51864c78a411a903))
* **trt:** add return cv::Mat instead of vector for GAN output ([4990e7b](https://github.com/jolibrain/deepdetect/commit/4990e7bc39e663ed1a96af2391d1d9e4e3b21f55))
* torch segmentation model prediction ([d72a138](https://github.com/jolibrain/deepdetect/commit/d72a138b7f39aa300f273e252d20fd0afb473369))


### Bug Fixes

* always depend on oatpp ([f262114](https://github.com/jolibrain/deepdetect/commit/f262114381d7a06ba99d5c7fc679a2188d7133b6))
* **test:** tar archive was decompressed at each cmake call ([910a0ee](https://github.com/jolibrain/deepdetect/commit/910a0ee5080260f2dbda8f78698e3db14fa5fe5c))
* **torch:** predictions handled correctly when data count > 1 ([5a95c29](https://github.com/jolibrain/deepdetect/commit/5a95c29a8a100f1a6dec4427a041a98185a19d2c))
* **trt:** detect architecture and rebuild model if necessary ([5c9ff89](https://github.com/jolibrain/deepdetect/commit/5c9ff896b3bc868f4ba493af7db7d432ff587722))
* **TRT:** fix build wrt new external build script ([7121dfe](https://github.com/jolibrain/deepdetect/commit/7121dfed3fdcce3672342a62ee38770c011cb709))
* **TRT:** make refinedet great again, also upgrades to TRT8.0.0/TRT-OSS21.08 ([bdff2ae](https://github.com/jolibrain/deepdetect/commit/bdff2aedc2e0f2cb5e4110bda928f53e1c4cbdb4))
* CI on Jetson nano with lighter classification model ([1673a99](https://github.com/jolibrain/deepdetect/commit/1673a99ecc922e01dd7cc8845098291ef46a8902))
* dont rebuild torchvision everytime ([4f17897](https://github.com/jolibrain/deepdetect/commit/4f178973aac93e9616fe7d9449c1326c402b2ef8))
* remove linking errors on oatpp access_log ([ed276b3](https://github.com/jolibrain/deepdetect/commit/ed276b30385be690923404f4052a30fbde94e5f1))
* torch build with custom spdlog and C++14 ([4435540](https://github.com/jolibrain/deepdetect/commit/44355402f6f5f2b9b5093625e0e08b0f448565ea))
