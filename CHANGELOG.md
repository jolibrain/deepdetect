# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.20.0](https://github.com/jolibrain/deepdetect/compare/v0.19.0...v0.20.0) (2021-12-17)


### Features

* add elapsed time to training metrics ([fe5fc41](https://github.com/jolibrain/deepdetect/commit/fe5fc41e7090d5756f99488ceb02708a58d95b7d))
* add onnx export for torchvision models ([07f69b1](https://github.com/jolibrain/deepdetect/commit/07f69b1f01af46088a00019d480b653b4b0350aa))
* add yolox export script for training and inference ([0b2f20b](https://github.com/jolibrain/deepdetect/commit/0b2f20be8211a95b1fea3a600f0d5ba17b8d339f))
* add yolox onnx export and trt support ([80b7e6a](https://github.com/jolibrain/deepdetect/commit/80b7e6a658a05046d840b0f2d0591ee865d75168))
* chain uses dto end to end ([5efbf28](https://github.com/jolibrain/deepdetect/commit/5efbf283f8056fef09512db7a11277b0f15ecd2d))
* data augmentation for training segmentation models with torch backend ([b55c218](https://github.com/jolibrain/deepdetect/commit/b55c218f3a31e7877039cd027f010dfcace56bd7))
* DETR export and inference with torch backend ([1e4ea4e](https://github.com/jolibrain/deepdetect/commit/1e4ea4e8e21759682c0355974f8da4bedfd890bd))
* full cuda pipeline for tensorrt ([93815d7](https://github.com/jolibrain/deepdetect/commit/93815d7c607560890435b6bbe2f32be8306c8380))
* noise image data augmentation for training with torch backend ([2d9757d](https://github.com/jolibrain/deepdetect/commit/2d9757d40463194db403ff6d675e3570603edecb))
* training segmentation models with torch backend ([1e3ff16](https://github.com/jolibrain/deepdetect/commit/1e3ff160b2b0796ea8dc1bd7252689c4bf7482ff))
* **ml:** activate cutout for object detector training with torch backend ([8a34aa1](https://github.com/jolibrain/deepdetect/commit/8a34aa17213ffeeea003c5223b8f4e85647fbbda))
* **ml:** distortion noise for image training with torch backend ([35a16df](https://github.com/jolibrain/deepdetect/commit/35a16dfabc4ae1148b854d81324812460d90f98a))
* **torch:** dice loss https://arxiv.org/abs/1707.03237 ([542bcb4](https://github.com/jolibrain/deepdetect/commit/542bcb49870c82d2bccfd1bf68ac2eaa76e30846))
* **torch:** manage models with multiple losses ([bea7cb4](https://github.com/jolibrain/deepdetect/commit/bea7cb46c0bfda50526b7af262b7e0ccf3d0b181))


### Bug Fixes

* **cpu:** cudnn is now on by default, auto switch it to off in case of cpu_only ([3770baf](https://github.com/jolibrain/deepdetect/commit/3770baf63c06746aaee3aa681333492a61ecde8b))
* **tensorrt:** read onnx model to find topk ([5cce134](https://github.com/jolibrain/deepdetect/commit/5cce1348b865d90a920559b8246a7129bb9e1c09))
* simsearch ivf index craft after reload, disabling mmap ([8a2e665](https://github.com/jolibrain/deepdetect/commit/8a2e665569887f040bbec624e8aa0266802c9c32))
* **tensorrt:** yolox postprocessing in C++ ([1d781d2](https://github.com/jolibrain/deepdetect/commit/1d781d25b4ad3246be46e6df52685a2197c4977c))
* **torch:** add include sometimes needed ([74487dc](https://github.com/jolibrain/deepdetect/commit/74487dc0069df0ef43dc06fbdd825b3c123c66e2))
* add mltype in metrics.json even if training is not over ([9bda7f7](https://github.com/jolibrain/deepdetect/commit/9bda7f70382279724c2d00967150e4a01f5b85fa))
* clang formatting of mlmodel ([130626b](https://github.com/jolibrain/deepdetect/commit/130626b0040f414cc70f41741d08d0005db854fa))
* **torch:** avoid crashes caused by an exception in the training loop ([667b264](https://github.com/jolibrain/deepdetect/commit/667b26416c8a2011b327108a8744a35d25d2c60b))
* **torch:** bad bbox rescaling on multiple uris ([05451ed](https://github.com/jolibrain/deepdetect/commit/05451ed1aa3827c6a51aec6e592d18be29b222ac))
* **torch:** correct output name for onnx classification model ([a03eb87](https://github.com/jolibrain/deepdetect/commit/a03eb87fcd60267deac403e33850fd38c6a7760e))
* **torch:** prevent crash during training if an exception is thrown ([4ce7802](https://github.com/jolibrain/deepdetect/commit/4ce78020982f29b62c4d04f189711abe3b3d8c65))
* torch train shuffles dataset by default ([b086b77](https://github.com/jolibrain/deepdetect/commit/b086b7717faf4ec18facd95da379ce1d73078338))
