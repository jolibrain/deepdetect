# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## 0.10.0 (2020-10-09)


### Features

* **timeseries:** MAPE, sMAPE, MASE, OWA metrics ([c1f4ef9](https://github.com/jolibrain/deepdetect/commit/c1f4ef9fa240556d4ab624bae23443a1d19f539b))
* automatically push image build for master ([19e9674](https://github.com/jolibrain/deepdetect/commit/19e9674897c1933e0dad358b9da17ce9ce5990d2))
* **build:** add script to create cppnet-lib debian package ([28247b4](https://github.com/jolibrain/deepdetect/commit/28247b435e7290fb3297a5ce90731f33f0f6fc71))
* **build:** allow to change CUDA_ARCH ([67ad43e](https://github.com/jolibrain/deepdetect/commit/67ad43e0079fa3e3beee73158a737f8b14b3f1ce))
* **dede:** Training for image classification with torch ([6e81915](https://github.com/jolibrain/deepdetect/commit/6e81915128c0c6b9d9a20184a119fd861b5e696e))
* **docker:** publish image as soon as ready ([957e07c](https://github.com/jolibrain/deepdetect/commit/957e07c9fe02ca6f92a4af72d9fc6be236af0a8d))
* **docker:** publish image as soon as ready ([5f7013d](https://github.com/jolibrain/deepdetect/commit/5f7013d17a2ab6ebd452f2642ef1bd252de93154))
* **docker:** rework Dockerfile ([8bc9ddf](https://github.com/jolibrain/deepdetect/commit/8bc9ddf6d9f5dbc9035f0a8198f8d2fb353f7aa0))
* **docker:** use prebuild cppnet-lib ([c929773](https://github.com/jolibrain/deepdetect/commit/c92977350cabbc1a180e11104ad141105d2231a1))
* **graph:** lstm autoencoder ([038a74c](https://github.com/jolibrain/deepdetect/commit/038a74c4f79c38e689108a113a944edc54d4e763))
* **nbeats:** expose hidden size param in API ([d7e5515](https://github.com/jolibrain/deepdetect/commit/d7e5515aaa428d1767fcc720483d9c5034f17022))
* add auto release tools ([98b41b0](https://github.com/jolibrain/deepdetect/commit/98b41b037f1bb8aaea4b1bba8910ee273b7ab08b))
* **imginputfile:** histogram equalization of input image ([2f0061c](https://github.com/jolibrain/deepdetect/commit/2f0061caf939e9cd4efeeac0cdfa5df7269970dd)), closes [#778](https://github.com/jolibrain/deepdetect/issues/778)
* **imginputfile:** histogram equalization of input image ([576f2d8](https://github.com/jolibrain/deepdetect/commit/576f2d8a966035425f58509a1888412c3c6acdd2)), closes [#778](https://github.com/jolibrain/deepdetect/issues/778)
* **stats:** added service statistics mechanism ([1839e4a](https://github.com/jolibrain/deepdetect/commit/1839e4a3451a9d6cab6025c0a5face956817cc70))
* **torch:** in case of timeseries, warn if file do not contain enough timesteps ([1a5f905](https://github.com/jolibrain/deepdetect/commit/1a5f9059a6551182384101af039a8be04237edd4))
* **torch:** nbeats ([f288665](https://github.com/jolibrain/deepdetect/commit/f2886654240ed3921cb9ea571b4812323a1cb8c8))
* **torch:** upgrade to torch 1.6 ([f8f7dbb](https://github.com/jolibrain/deepdetect/commit/f8f7dbbd9854e5f0459e340badf3e1d1994c211f))
* **torch,native:** extract_layer ([d37e182](https://github.com/jolibrain/deepdetect/commit/d37e182040fddda7eb0a7a9a6190ea4f79b8aa3d))
* add json output to dd_bench.py ([874fc01](https://github.com/jolibrain/deepdetect/commit/874fc0142eae604013cbff48d341cef402b55fdd))
* added bw image input support to dd_bench ([6e558d6](https://github.com/jolibrain/deepdetect/commit/6e558d61e5da4dbd1ac92f9029a20b669c84354c))
* **trains-status:** add tflops to body.measures ([af31c8b](https://github.com/jolibrain/deepdetect/commit/af31c8bda0285eb778ccbf378be38e82f6f68895)), closes [#785](https://github.com/jolibrain/deepdetect/issues/785)
* Docker images optimization ([fba637a](https://github.com/jolibrain/deepdetect/commit/fba637a7c70ddb8e73c2bdd2850e6daea36259f0))
* format the code with clang-format ([07d6bdc](https://github.com/jolibrain/deepdetect/commit/07d6bdc0227ed2f799942e6c7eb80690c8a2a16f))
* LSTM over torch , preliminary internal graph representation ([25faa8b](https://github.com/jolibrain/deepdetect/commit/25faa8bcf3d0eff6aec27642e53d67580866850a))
* update all docker images to ubuntu 18.04 ([eaf0421](https://github.com/jolibrain/deepdetect/commit/eaf04210642fa22696d1b1fa0af1e7b9e9d25c2f))


### Bug Fixes

* fix split_data in csvts connector ([8f554b5](https://github.com/jolibrain/deepdetect/commit/8f554b510e4cb7ee4df1e56c54d0bf6358e74ab4))
* **build:** CUDA_ARCH not escaped correctly ([696087f](https://github.com/jolibrain/deepdetect/commit/696087f4cafe4be555ee1b37a1587f37e67d4383))
* **build:** ensure all xgboost submodules are checkouted ([12aaa1a](https://github.com/jolibrain/deepdetect/commit/12aaa1af8ca80b82948d5b12f71582adfda1c0d1))
* **clang-format:** signed/unsigned comparaison ([af8e144](https://github.com/jolibrain/deepdetect/commit/af8e144f5ae0f8667775d837096bd3fa27ed766c))
* **clang-format:** signed/unsigned comparaison ([0ccabb6](https://github.com/jolibrain/deepdetect/commit/0ccabb62ca97008816760714e057f600befc5c1c))
* **clang-format:** typo in dataset tarball command ([04ddad7](https://github.com/jolibrain/deepdetect/commit/04ddad7820612f8ae82cc8daff4a3c14ea01a79d))
* **csvts:** correctly store and print test file names ([12d4639](https://github.com/jolibrain/deepdetect/commit/12d4639a06b26b4fc5edd36a6ed86d04633f8bed))
* **dede:** Remove unnecessary caffe include that prevent build with torch only ([a471b82](https://github.com/jolibrain/deepdetect/commit/a471b82c79b732a2c0dea6dfd675b2d0419e343c))
* **dede:** support all version of spdlog while building with syslog ([81f47c9](https://github.com/jolibrain/deepdetect/commit/81f47c9101be8ba5e24ac2206d623db1c9f605c8))
* **docker:** add missing .so at runtime ([4cc24ce](https://github.com/jolibrain/deepdetect/commit/4cc24ceb5f2c1a8c214a6795fc7fe46f7e4ea0d6))
* **docker:** add missing gpu_tensorrt.Dockerfile ([97ff2b3](https://github.com/jolibrain/deepdetect/commit/97ff2b3ec4fc447810eacabcb9ebde0091ff10cc))
* **docker:** add some missing runtime deps ([0883a33](https://github.com/jolibrain/deepdetect/commit/0883a33f95f75bff473197e4a0fd24bffc9a0e58))
* **docker:** add some missing runtime deps ([a91f35f](https://github.com/jolibrain/deepdetect/commit/a91f35f3e4669be16b0c602cb4169d272321c97f))
* **docker:** fixup base runtime image ([6238dd4](https://github.com/jolibrain/deepdetect/commit/6238dd4698c9ea674852bf895384ec7ed75f0c8b))
* **docker:** install rapidjson-dev package ([30fb2ca](https://github.com/jolibrain/deepdetect/commit/30fb2caf7971343213335e98100d3e2e8df697f9))
* **native:** do not raise exception if no template_param is given ([d0705ab](https://github.com/jolibrain/deepdetect/commit/d0705abf53d5982a8f660f9a3c6e74687630b2d7))
* **nbeats:** correctly setup trend and seasonality models (implement paper version and not code version) ([75accc6](https://github.com/jolibrain/deepdetect/commit/75accc61516890ebc4ef8f15d9e44bbc0c4b3376))
* **nbeats:** much lower memory use in case of large dim signals ([639e222](https://github.com/jolibrain/deepdetect/commit/639e22285e7a0ce907ff3e2368d0840b5763dbf8))
* **tests:** inc iteration of torchapi.service_train_image test ([4c93ace](https://github.com/jolibrain/deepdetect/commit/4c93ace24e8c3c5161bacdb2b6dad40ff87aa445))
* **torch:** Fix conditions to add classification head. ([f46a710](https://github.com/jolibrain/deepdetect/commit/f46a710d9b1c1f28774c1985bc4450386587ca95))
* **torch/timeseries:** unscale prediction output if needed ([aa30e88](https://github.com/jolibrain/deepdetect/commit/aa30e88a22d526d25361c40919e6ec4c5de90f6a))
* /api/ alias when deployed on deepdetect.com ([4736893](https://github.com/jolibrain/deepdetect/commit/4736893614b0678af04254ffe2923c3a51b03350))
* add support and automated processing of categorical variables in timeseries data ([1a9af3e](https://github.com/jolibrain/deepdetect/commit/1a9af3e32a8627d03b305676177ef46ed527aaa5))
* allow serialization/deserializationt of Inf/-Inf/NaN ([976c892](https://github.com/jolibrain/deepdetect/commit/976c892d3326b80c69e708ae20763df59c6d41ca))
* allows to specify size and color/bw with segmentation models ([58ecb4a](https://github.com/jolibrain/deepdetect/commit/58ecb4a2b772ed48ca690095d957c0e10fb61550))
* build with -DUSE_TENSORRT_OSS=ON ([39bd675](https://github.com/jolibrain/deepdetect/commit/39bd67524aa3d455c348656605cc50f0b1b5b719))
* convolution layer initialization of SE-ResNeXt network templates ([69ff0fb](https://github.com/jolibrain/deepdetect/commit/69ff0fb2b0e3755b0dfc26796e2a10a16b787b8f))
* in tensorrt builds, remove forced cuda version and unused lib output + force-select tensorrt when tensorrt_oss is selected ([9430fb4](https://github.com/jolibrain/deepdetect/commit/9430fb40b6924d57d43b602f35686f688c6afe3e))
* input image transforms in API doc ([f513f17](https://github.com/jolibrain/deepdetect/commit/f513f1750173ae76971185d5e04410091fba41e0))
* install cmake version 3.10 ([10666b8](https://github.com/jolibrain/deepdetect/commit/10666b81d734678916360b73e8fcf38d5321cdea))
* missing variant package in docker files ([dcf738b](https://github.com/jolibrain/deepdetect/commit/dcf738bdad357ef4762cfe12f60fcfc844a516ac))
* race condition in xgboost|dede build ([fd32eae](https://github.com/jolibrain/deepdetect/commit/fd32eae643884f92c47ee27ab5ece42980ae2221))
* remove unecessary limit setting call to protobuf codedstream ([ae26f59](https://github.com/jolibrain/deepdetect/commit/ae26f59651b8e725b67f9f301664683c94af5b95))
* replace db":true by db":false in json files when copying models ([06ac6df](https://github.com/jolibrain/deepdetect/commit/06ac6dfb38a943de351e64bf6d26041df07147ab))
* set caffe smooth l1 loss threshold to 1 ([0e329f0](https://github.com/jolibrain/deepdetect/commit/0e329f08803f12c7df03bf555594ddd0b84b467f))
* ssd_300_res_128 deploy file is missing a quote ([4e52a0e](https://github.com/jolibrain/deepdetect/commit/4e52a0e66f74050811f882508c7eacae11fdc3d5))
* svm prediction with alll db combinations ([d8e2961](https://github.com/jolibrain/deepdetect/commit/d8e2961bf970243052421bd1d7000bcf08e72bed))
* svm with db training ([6e925f2](https://github.com/jolibrain/deepdetect/commit/6e925f2bb0a19633b408dca1d457a246acdd4c52))
* tensorrt does not support blank_label ([7916500](https://github.com/jolibrain/deepdetect/commit/7916500876d8acab85c3f49010cfc49faa8a9186))
* typo in docker image builder ([cb5ae19](https://github.com/jolibrain/deepdetect/commit/cb5ae19b3674936281a5f6ce1959a665f09c04cd))
* unusual builds (ie w/o torch or with tsne only lead to build errors ([241bf6b](https://github.com/jolibrain/deepdetect/commit/241bf6b2cd5cf4db0ccb144152557c3c1fed2776))
* update caffe cudnn engine without template ([ca58c51](https://github.com/jolibrain/deepdetect/commit/ca58c51618b759a254c0d479e5d0e458197676a0))
* **torch:** handle case where sequence data is < wanted timestep ([b6d394a](https://github.com/jolibrain/deepdetect/commit/b6d394ac229993e21f557d58a70116823fe75a6b))
* **TRT:** refinedet ([b6152b6](https://github.com/jolibrain/deepdetect/commit/b6152b6bdc0d7f6b77a253b1c0f9a96d0fcedd7f))

### 0.9.7 (2020-04-18)

### 0.9.6 (2020-02-05)

### 0.9.5 (2019-12-30)

### 0.9.4 (2019-10-22)

### 0.9.3 (2019-08-27)


### Bug Fixes

* caffe solver selection ([c45abfc](https://github.com/jolibrain/deepdetect/commit/c45abfc8c769b10123113d96ac497950f6d4f5da))

### 0.9.2 (2019-08-02)


### Bug Fixes

* do not segfault when cannot find ncnn model files ([6aee195](https://github.com/jolibrain/deepdetect/commit/6aee195c9db4fe723e79b9505bee37a4ae4f39d7))

### 0.9.1 (2019-06-30)

## 0.9.0 (2019-05-09)


### Bug Fixes

* compilation race issue. ([4e16c56](https://github.com/jolibrain/deepdetect/commit/4e16c565a3ce5584260aad119e4ccf3f96c61528))
* deps issue for NCNN ([22f40bd](https://github.com/jolibrain/deepdetect/commit/22f40bdd18d861563ce47429718b9cb5dae36fd6))
* do not build tf unit tests when tf not requested in build ([e9974b8](https://github.com/jolibrain/deepdetect/commit/e9974b8df9ed0cb1bf8e62106289bdf263c9bf49))
* issues with hardware_concurrency declaration. ([20d5fe5](https://github.com/jolibrain/deepdetect/commit/20d5fe5b92cab363124fc6f14293d3e57e39e9a9))
* NCNN compilation. ([089b0dc](https://github.com/jolibrain/deepdetect/commit/089b0dceb6fc14787b28c7b9e99033638eb4c057))
* NCNN input issues. ([111269a](https://github.com/jolibrain/deepdetect/commit/111269ae76c9de6ed7a2bc3fdcd391018c369454))
* NCNN service declaration. ([02df07d](https://github.com/jolibrain/deepdetect/commit/02df07d255aadbe500f2644fbc45b365a6929366))
* NCNN service model constructor. ([633bccb](https://github.com/jolibrain/deepdetect/commit/633bccbd50ad19c1bd7fbc88dca31caa09d2ebda))
* small glitch in init, was ok before, but clearer now ([bfd9fa9](https://github.com/jolibrain/deepdetect/commit/bfd9fa92955627bfbb2a43ac1626e6968beff6c3))
* throw error on NCNN extraction fail ([569800a](https://github.com/jolibrain/deepdetect/commit/569800ae9e433911fe7d82928114a2ccd4737ac5))
