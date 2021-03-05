# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.14.0](https://github.com/jolibrain/deepdetect/compare/v0.13.0...v0.14.0) (2021-03-05)


### Features

* **bench:** Add parameters for torch image backend ([5d24f3d](https://github.com/jolibrain/deepdetect/commit/5d24f3d4665c0c7cd21bc2ba84643c6f7830735f))
* **ml:** ViT support for Realformer from https://arxiv.org/abs/2012.11747v2 ([5312de7](https://github.com/jolibrain/deepdetect/commit/5312de770eb16408d7bea8ffbc4a6b24f35a95c9))
* **nbeats:** add parameter coefficient to backcast loss ([35b3c31](https://github.com/jolibrain/deepdetect/commit/35b3c313fd0f122c788969e93e5ffa476150e8ea))
* **torch:** add inference for torch detection models ([516eeb6](https://github.com/jolibrain/deepdetect/commit/516eeb6a56ac36aefbb3f24624344236cbb20f39))
* **torch:** Sharpness Aware Minimization (2010.01412) ([45a8408](https://github.com/jolibrain/deepdetect/commit/45a84087b5321a5c3eebcb8a6d53975d1b544478))
* **torch:** support for multiple test sets ([c0dcec9](https://github.com/jolibrain/deepdetect/commit/c0dcec9a51f86cf904809c492fc175b5951dae5b))
* **torch:** temporal transformers (encoder only) (non autoreg) ([3538eb7](https://github.com/jolibrain/deepdetect/commit/3538eb78a721b477f377d6798f707796be8319e0))
* CSV parser support for quotes and string labels ([efa4c79](https://github.com/jolibrain/deepdetect/commit/efa4c79e9fe21e9074f17ca20b020f97bd2112cb))
* new cropping action parameters in chains ([6597b53](https://github.com/jolibrain/deepdetect/commit/6597b53671b19022b2b7b32f4e2a6e0a29136f21))
* running custom methods from jit models ([73d1eef](https://github.com/jolibrain/deepdetect/commit/73d1eef00b0b41083237e7061d11ff8d4156f612))
* **torch/txt:** display msg if vocab not found ([31837ec](https://github.com/jolibrain/deepdetect/commit/31837eca5907c4cac7aa3d42f0d12b474ad673f9))
* SSD MAP-x threshold control ([acd252a](https://github.com/jolibrain/deepdetect/commit/acd252a2a448f0c1f8c497ae665cc5be7649f35d))
* use oatpp::DTO to parse img-input-connector APIData ([33aee72](https://github.com/jolibrain/deepdetect/commit/33aee72ad4450a1080dd53f40a5c2cea14a304b8))


### Bug Fixes

* **build:** pytorch with custom spdlog ([1fb19a0](https://github.com/jolibrain/deepdetect/commit/1fb19a02c700698a4dc262a6c81ef83f8c3623a6))
* **caffe/cudnn:** force default engine option in case of cudnn not compiled in ([b6dec4e](https://github.com/jolibrain/deepdetect/commit/b6dec4e30dc166e9f37246a69bb15a0d9efc6c3e))
* **chore:** typo when trying to use syslog ([374e6c4](https://github.com/jolibrain/deepdetect/commit/374e6c4b48a18f4e41eacef3cb5e13bcf325b0f7))
* **client:** Change python package name to dd_client ([b96b0fa](https://github.com/jolibrain/deepdetect/commit/b96b0fade15a46b6a89bc830f00f650b4ca7242b))
* **csvts:** read from memory ([6d1dba8](https://github.com/jolibrain/deepdetect/commit/6d1dba85fc584d7bf75d30f63c506c6e00aaa07e))
* **csvts:** throw proper error when a csv file is passed at training time ([90aab20](https://github.com/jolibrain/deepdetect/commit/90aab201df316dea23f1bc47c5ca5d300d95f12c))
* **docker:** ensure pip3 is working on all images ([a374a58](https://github.com/jolibrain/deepdetect/commit/a374a5898e73cddb0b76b4309ad59c4329359571))
* **ncnn:** update innerproduct so that it does not pack data ([9d88187](https://github.com/jolibrain/deepdetect/commit/9d88187381982c0b49170aa749caf8581532128c))
* **torch:** add error message when repository contains multiple models ([a08285f](https://github.com/jolibrain/deepdetect/commit/a08285f51b2f614d24fea08d6c62edf3c9a47e74))
* -Werror=deprecated-copy gcc 9.3 ([0371cfa](https://github.com/jolibrain/deepdetect/commit/0371cfa03bf0c42ce3a643c198e7154d426c7892))
* action cv macros with opencv >= 3 ([37d2926](https://github.com/jolibrain/deepdetect/commit/37d292683a7a8039ec77cd66ab16f21342b5f28c))
* caffe build spdlog dependency ([62e781a](https://github.com/jolibrain/deepdetect/commit/62e781a4a2f97d420d3a34cbb16da40d27d6199c))
* docker /opt/models permissions ([82e2695](https://github.com/jolibrain/deepdetect/commit/82e269589a9a8160eb1c63fbde2f8b372f0838d6))
* prevent softmax after layer extraction ([cbee659](https://github.com/jolibrain/deepdetect/commit/cbee65945d46ee5f304519bd760d92be3b00eb2f))
* tag syntax for github releases ([4de3807](https://github.com/jolibrain/deepdetect/commit/4de3807adfb13957358e41b95a09bd9ee0533a09))
* torch backend CPU build and tests ([44343f6](https://github.com/jolibrain/deepdetect/commit/44343f6236d9afc70f931b7a762d4df591325abf))
* typo in oatpp chain HTTP endpoint ([955b178](https://github.com/jolibrain/deepdetect/commit/955b178b09a015b1f147449f277c0e4945c48d3a))
* **torch:** gather torchscript model parameters correctly ([99e4dbe](https://github.com/jolibrain/deepdetect/commit/99e4dbe34e8845331a95dec3b4dd7bad3d11b03b))
* **torch:** set seed of torchdataset during training ([d02404a](https://github.com/jolibrain/deepdetect/commit/d02404a6120ef6ec599accc63e8bc25c27072e7e))
* **torch/ranger:** allow not to use lookahead ([d428d08](https://github.com/jolibrain/deepdetect/commit/d428d08e5bd40166f899c2c317a9617f0faf61a8))
* **torch/timeseries:** in case of db, correctly finalize db ([aabedbd](https://github.com/jolibrain/deepdetect/commit/aabedbd2d7c2be14360cb3df2e5a71f93805102a))
* **torch/txt:** correclty handle test sets in case of no splitting ([00036b1](https://github.com/jolibrain/deepdetect/commit/00036b1b26a273f7452fe9f1943322f9b86e745b))
