# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.22.0](https://github.com/jolibrain/deepdetect/compare/v0.21.0...v0.22.0) (2022-05-23)


### Features

* **cpp:** torch predict to DTO ([b88f22a](https://github.com/jolibrain/deepdetect/commit/b88f22a214cf8a59c4df70cbedbc8854d7a189bc))
* sliding object detection script ([0e3df67](https://github.com/jolibrain/deepdetect/commit/0e3df679941d50c3c79d2a9b4604c26999f3e9a3))
* tensorrt object detector top_k control ([655aa48](https://github.com/jolibrain/deepdetect/commit/655aa483c0f129a0a07c0da9be1f1ab8a465f1be))
* **torch:** bump to torch 1.11 and torchvision 0.12 ([5d312d0](https://github.com/jolibrain/deepdetect/commit/5d312d02c12ad8d9a0a1b0d6605e1e17ec1e53d4))
* **torch:** ocr model training and inference ([3fc2e27](https://github.com/jolibrain/deepdetect/commit/3fc2e278974a168bac1d1fba87913a75fa8a931e))
* **trt:** update tensorrt to 22.03 ([c03aa9d](https://github.com/jolibrain/deepdetect/commit/c03aa9d515a3fa4a058174e24b917318ec91cd8f))


### Bug Fixes

* cropped model input size when publishing torch models + tests ([2dabd89](https://github.com/jolibrain/deepdetect/commit/2dabd8923c8123d07534b2cb35d424b39869f439))
* cutout and crops in data augmentation of torch models ([1ef2796](https://github.com/jolibrain/deepdetect/commit/1ef2796220a76a64bb68263443e6161d18c28f62))
* **docker:** fix libraries not found in trt docker ([86f3924](https://github.com/jolibrain/deepdetect/commit/86f3924bb67482f8c5bcc3ae7da41c9007009754))
* remove semantic commit check ([5d0f0c7](https://github.com/jolibrain/deepdetect/commit/5d0f0c774600b026b68661c1d540cd468326d3a4))
* seeded random crops at test time ([92feae3](https://github.com/jolibrain/deepdetect/commit/92feae33bb759e486ab86f606aeb41466c6e62a4))
* torch best model better or equal ([4d50c8e](https://github.com/jolibrain/deepdetect/commit/4d50c8ed8e1422c5db3a583196dfa67bdabc7615))
* torch model publish crash and repository ([6a89b83](https://github.com/jolibrain/deepdetect/commit/6a89b8332b3b117845f3f4baf54420af716674f6))
* **torch:** Fix update metrics and solver options when resuming ([9b0019f](https://github.com/jolibrain/deepdetect/commit/9b0019f54614ed909dedf63bcfb7fe1316bbb900))
* **trt:** yolox failed at second inference ([72cc87c](https://github.com/jolibrain/deepdetect/commit/72cc87cb585173fc9719c28168acb0c546a8c995))
