# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## [0.15.0](https://github.com/jolibrain/deepdetect/compare/v0.14.0...v0.15.0) (2021-03-26)


### Features

* **nbeats:** default backcast loss coeff to zero, allows very short forecast length to learn smoothly ([db17a41](https://github.com/jolibrain/deepdetect/commit/db17a41401b037187b5ccf2e54464e3f6647e40d))
* **timeseries:** add MAE and MSE metrics ([847830d](https://github.com/jolibrain/deepdetect/commit/847830d8f6a011be05763b36fbf7240dd6d867e6))
* **timeseries:** do not output per serie metrics as a default, add prefix _all for displaying all metrics ([5b6bc4e](https://github.com/jolibrain/deepdetect/commit/5b6bc4e19274595741e8fd11cbfd326b0497b79f))
* **torch:** model publishing with the platform ([da14d33](https://github.com/jolibrain/deepdetect/commit/da14d33affb362aa869367fb748d5dbac1d73a10))
* **torch:** save last model at training service interruption ([b346923](https://github.com/jolibrain/deepdetect/commit/b34692395ee6c0d03b6a378d2b454a1479e52e76))
* **torch:** SWA for RANGER/torch (https://arxiv.org/abs/1803.05407) ([74cf54c](https://github.com/jolibrain/deepdetect/commit/74cf54cce30b791def7712eabd0c93c31eebf91b))
* **torch/csvts:** create db incrementally ([4336e89](https://github.com/jolibrain/deepdetect/commit/4336e893efe3c41d97b0199d300c5461fde55776))


### Bug Fixes

* **caffe/detection:** fix rare spurious detection decoding, see bug 1190 ([94935b5](https://github.com/jolibrain/deepdetect/commit/94935b5a6c9a4ab9321cca52d5050f3b520e9ff7))
* **chore:** add opencv imgcodecs explicit link ([8ff5851](https://github.com/jolibrain/deepdetect/commit/8ff585140f8784e2a91a955c53d10fcb0917369d))
* compile flags typo ([8f0c947](https://github.com/jolibrain/deepdetect/commit/8f0c947eefad3bde0defae52f4b85317a0e98f50))
* docker cpu link in readme ([1541dcc](https://github.com/jolibrain/deepdetect/commit/1541dccfdbede08ffd0ce466f7f27a171d6647a9))
* tensorrt tests on Jetson nano ([25b12f5](https://github.com/jolibrain/deepdetect/commit/25b12f573d6894a24d233a30ee85092327e0d96f))
* **nbeats:** make seasonality block work ([d035c79](https://github.com/jolibrain/deepdetect/commit/d035c794822f57be5d2aad57403cc2d7ba06738f))
* **torch:** display msg if resume fails, also fails if not best_model.txt file ([d8c5418](https://github.com/jolibrain/deepdetect/commit/d8c541838713ee2922e3d262c24fcf0bf058ce1a))
* **torch:** throw error if multiple models are provided ([efbd1f9](https://github.com/jolibrain/deepdetect/commit/efbd1f93077791299013e04bef292aba9a09afa5))
