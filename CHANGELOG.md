# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.21.0](https://github.com/jolibrain/deepdetect/compare/v0.20.0...v0.21.0) (2022-02-22)


### Features

* add predict from video ([02872eb](https://github.com/jolibrain/deepdetect/commit/02872eb2139e20843b1fdcb16aa8cb22f4339cbc))
* add video input connector and streaming endpoints ([07644b4](https://github.com/jolibrain/deepdetect/commit/07644b43e4e443bb3662a7fd7229a8706a3229b5))
* allow pure negative samples for training object detectors with torch ([cd23bad](https://github.com/jolibrain/deepdetect/commit/cd23bad4b890404b42ae3362e6e848f2cec585e8))
* **bench:** add monitoring of transform time ([3f77d42](https://github.com/jolibrain/deepdetect/commit/3f77d42b96aa98a6e784b3225a0668542fafa55e))
* **chain:** add action to draw bboxes as trailing action ([ae0a05f](https://github.com/jolibrain/deepdetect/commit/ae0a05f32591cec5bec7ba5f768d3971943f0b3f))
* **chain:** allow user to add their own custom actions ([a470c7b](https://github.com/jolibrain/deepdetect/commit/a470c7baf5ae4f00b4ae75646a29645e529df2b7))
* **ml:** added support for segformer with torch backend ([ab03d1d](https://github.com/jolibrain/deepdetect/commit/ab03d1dd7412ff5d2aa7e02abb60a340e8b1727e))
* **ml:** random cropping for training segmentation models with torch ([ac7ce0f](https://github.com/jolibrain/deepdetect/commit/ac7ce0ffaef57f9b8a1d20107037dce27332acf4))
* random crops for object detector training with torch backend ([385122d](https://github.com/jolibrain/deepdetect/commit/385122d4eace490ab95fa7a7b9ed92121af1414e))
* segmentation of large images with sliding window, example Python script ([8528e9a](https://github.com/jolibrain/deepdetect/commit/8528e9a689f9f68e436da91b6e59b6117f6470ae))


### Bug Fixes

* bbox clamping in torch inference ([2d6efd3](https://github.com/jolibrain/deepdetect/commit/2d6efd3eacbadc0f71aa3adf35017ae080bbc9ea))
* caffe object detector training requires test set ([2e4db7e](https://github.com/jolibrain/deepdetect/commit/2e4db7ea7daade86d6e138f75b867ee662166367))
* dataset output dimension after crop augmentation ([636d455](https://github.com/jolibrain/deepdetect/commit/636d4555ff87bd5df433503a0362d621e7d38657))
* **detection/torch:** correctly normalize MAP wrt torchlib outputs ([b12d188](https://github.com/jolibrain/deepdetect/commit/b12d188e46df4511d1294311319b4b6b8ff53a53))
* model.json file saving ([809f00a](https://github.com/jolibrain/deepdetect/commit/809f00a9e22878ca1c75aa3b02aeb80b5d6b9e05))
* segmentation with torch backend + full cropping support ([e14c3f2](https://github.com/jolibrain/deepdetect/commit/e14c3f2fed8a593640f963791d2209d0308ffdb5))
* torch MaP with bboxes ([9bc840f](https://github.com/jolibrain/deepdetect/commit/9bc840f0b1055426670d64b5285701d6faceabb9))
* torch model published config file ([b0d4e04](https://github.com/jolibrain/deepdetect/commit/b0d4e0485443fb9c069bde4d2b323e13e8733d93))
* **torch:** fix unweighted dice loss ([04ef758](https://github.com/jolibrain/deepdetect/commit/04ef758d35c3b9005c483d9fed90f4decc9dc4d9))
