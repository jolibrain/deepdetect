# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.25.0](https://github.com/jolibrain/deepdetect/compare/v0.24.0...v0.25.0) (2024-01-10)


### ⚠ BREAKING CHANGES

* **trt:** dropped support for caffe refinedet

### Features

* allow returning images in json in base64 format ([05096fd](https://github.com/jolibrain/deepdetect/commit/05096fdabf19f23b296484d06c7b0a94a2c22112))
* build Deepdetect + pytorch MPS on Apple platforms ([aa8822d](https://github.com/jolibrain/deepdetect/commit/aa8822d671f8badc188a55a67ef1fd5f4e97bd55))
* recompose action to recreate an image from a GAN + crop ([e1118b1](https://github.com/jolibrain/deepdetect/commit/e1118b147d6395a8d8343d3ea98c3171b6f63c08))
* **torch:** add map metrics with arbitrary iou threshold ([20d8ebe](https://github.com/jolibrain/deepdetect/commit/20d8ebea3ee37748101994986aeaffc553467cd9))
* **torch:** Added param `disable_concurrent_predict` ([71cb66a](https://github.com/jolibrain/deepdetect/commit/71cb66ab9bb00ca01fba4d03f4ea4d44ebe9a1b2))


### Bug Fixes

* add more explicit error messages ([ca2703c](https://github.com/jolibrain/deepdetect/commit/ca2703c02b2644a98e6d127514b9cd48d6d92187))
* allow two chain calls with the same name to be executed simultaneously ([b26b5b9](https://github.com/jolibrain/deepdetect/commit/b26b5b98a991457730747697891ac9a4ef9a45c6))
* **chain:** empty predictions were too empty ([57bed0b](https://github.com/jolibrain/deepdetect/commit/57bed0b1360bdd1fd5fc2ae162cd7630653bd398))
* **docker:** build CPU dockers ([9e56aba](https://github.com/jolibrain/deepdetect/commit/9e56aba46b248618341ac3798aea2f2209a4a184))
* no resize when training with images ([e84c616](https://github.com/jolibrain/deepdetect/commit/e84c6161aa75a7157b60c5bb51b144768481996e))
* prevent crash when a service is deleted before finishing predict ([0ef1f46](https://github.com/jolibrain/deepdetect/commit/0ef1f469a539e722a722ff693c91f0088087ca35))
* support boolean value for service info parameters ([737724d](https://github.com/jolibrain/deepdetect/commit/737724de18af18a3da29dc79d98a650228622f4d))
* torch architecture selected correctly at docker build ([5eb7890](https://github.com/jolibrain/deepdetect/commit/5eb7890c15bc4dffcbc430f3ec4b5379d3052340))
* **torch:** black&white image now working with crnn & dataaug ([2b07002](https://github.com/jolibrain/deepdetect/commit/2b070027944affedc753b9a88c7148a4f9fa71e3))
* **torch:** concurrent_predict was always true ([edb28c1](https://github.com/jolibrain/deepdetect/commit/edb28c11fb42e06f62411766b1dc027f56d009c7))


* **trt:** update to v8.6.1 & docker 23.05 & ubuntu 22.04 ([0e516bb](https://github.com/jolibrain/deepdetect/commit/0e516bbf8d5323a7b5d3954c6764d5a4650ecdf4))
