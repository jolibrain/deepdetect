# DeepDetect: Open Source Deep Learning Server & API (Changelog)

## 0.11.0 (2020-11-10)


### Features

* **bench:** support for regression model benchmarking ([a385292](https://github.com/jolibrain/deepdetect/commit/a385292f708fa3a58b680a5c08f8671211b9f456))
* make python client an install package ([ec2f5e2](https://github.com/jolibrain/deepdetect/commit/ec2f5e27470371d995e640d1f6c2722d08415051))
* one protobuf to rule them all ([77912fe](https://github.com/jolibrain/deepdetect/commit/77912fed039067b6124326dfe3ac22957d2d430c))
* **api:** add versions and compile flags to /info ([67b1d99](https://github.com/jolibrain/deepdetect/commit/67b1d992d029962b879512614bea353df9b9abfb)), closes [#897](https://github.com/jolibrain/deepdetect/issues/897)
* **caffe:** add new optimizers flavors to API ([d534a16](https://github.com/jolibrain/deepdetect/commit/d534a16e56ecc7366ee109a4d4fefdf5873c5f0e))
* **ml:** tensorrt support for regression models ([77a016b](https://github.com/jolibrain/deepdetect/commit/77a016b12cea8126ea96b1173a0116817591a8e9))
* **tensorrt:** Add support for onnx image classification models ([a8b81f2](https://github.com/jolibrain/deepdetect/commit/a8b81f2f20c8d9d2ac2268d1542412dd46c9e001))
* **torch:** ranger optimizer (ie rectified ADAM + lookahead) + \ ([a3004f0](https://github.com/jolibrain/deepdetect/commit/a3004f068fe92ddddab0905c0587aaac1a129378))


### Bug Fixes

* **torch:** best model was never saved on last iteration ([6d1aa4d](https://github.com/jolibrain/deepdetect/commit/6d1aa4d7b7110455d403d60cfb36331abe4bf863))
* **torch:** clip gradient in rectified adam as stated in annex B of original paper ([1561269](https://github.com/jolibrain/deepdetect/commit/1561269ae8b83eafee16a7867764430c5fe7f27e))
* **torch:** Raise an exception if gpu is not available ([1f0887a](https://github.com/jolibrain/deepdetect/commit/1f0887aaf379d069d7418848f7f0fb59b9c400d2))
* add pytorch fatbin patch ([43a698c](https://github.com/jolibrain/deepdetect/commit/43a698cd52fd24ab4139146eec4becca618f903f))
* add tool to generate debian buster image with the workaround ([5570db4](https://github.com/jolibrain/deepdetect/commit/5570db4574626f820d759d2d4e0f8092ede1c879))
* building documentation up to date for 18.04, tensorrt and tests ([18ba916](https://github.com/jolibrain/deepdetect/commit/18ba916e2c82427f08dbbfaea792fa3fc8a91430))
* docker adds missing pytorch deps ([314160c](https://github.com/jolibrain/deepdetect/commit/314160c49aaa41f06a47bbb3d44cc6d38f6d5530))
* docker build readme link from doc ([c6682bf](https://github.com/jolibrain/deepdetect/commit/c6682bfa7eb5ec7ef8c495056cb5fe6e4b7b7eac))
* handle int64 in conversion from json to APIData ([863e697](https://github.com/jolibrain/deepdetect/commit/863e697283ab4853714bdd36146dbd963dc38c4f))
* ignore JSON conversion throw in partial chains output ([742c1c7](https://github.com/jolibrain/deepdetect/commit/742c1c7c307ff5d7ccfdce0f7ea005b2166a35c4))
* missing main in bench.py ([8b8b196](https://github.com/jolibrain/deepdetect/commit/8b8b1968c6df221cdb372b83c6db4e83e490eacd))
* proper cleanup of tensorrt models and services ([d6749d0](https://github.com/jolibrain/deepdetect/commit/d6749d0c55ab731310e3f4c1c316a1c3489083ff))
* put useful informations in case of unexpected exception ([5ab90c7](https://github.com/jolibrain/deepdetect/commit/5ab90c7c2d64b1518dc3f556e4d4701811ab09aa))
* readme table of backends, models and data formats ([f606aa8](https://github.com/jolibrain/deepdetect/commit/f606aa832fefc8ba06e78511585effd7205f111f))
* regression benchmark tool parameter ([3840218](https://github.com/jolibrain/deepdetect/commit/3840218d8035172c681d96abc8872e5922afca0e))
* tensorrt output layer lookup now throws when layer does not exist ([ba7c839](https://github.com/jolibrain/deepdetect/commit/ba7c8398bfc016b9b3224b9f6f132a2178a268d0))
* **csvts/torch:** allow to read csv timeserie directly from query ([76023db](https://github.com/jolibrain/deepdetect/commit/76023db1bfafd02e2da7848b62ebec2efb4edcbd))
* **doc:** update to neural network templates and output connector ([2916daf](https://github.com/jolibrain/deepdetect/commit/2916daf423eea2948e160ced1a3a0ee6775b037e))
* **docker:** don't share apt cache between arch build ([75dc9e9](https://github.com/jolibrain/deepdetect/commit/75dc9e98ecfae303f3272c7881004ee192086f92))
* **graph:** correctly discard dropout ([16409a6](https://github.com/jolibrain/deepdetect/commit/16409a6f0e9429f0ab5d70aa4a79e1f7e994839f))
* **stats:** measure of inference count ([b517910](https://github.com/jolibrain/deepdetect/commit/b517910fed38d56c59c43a6d082e03f7a773486d))
* **timeseries:** do not segfault if no valid files in train/test dir ([1977bba](https://github.com/jolibrain/deepdetect/commit/1977bba73cee43bbaf2e1cb1e1322cc21c0361ea))
* **torch:** add missing header needed in case of building w/o caffe backend ([2563b74](https://github.com/jolibrain/deepdetect/commit/2563b74a0128c0934a0792eb16a50cdd2ff5ecdb))
* **torch:** load weights only once ([0052a03](https://github.com/jolibrain/deepdetect/commit/0052a03027d08ea443a096c6ce0f2d351e19313d))
* **torch:** reload solver params on API device ([30fa16f](https://github.com/jolibrain/deepdetect/commit/30fa16f2e9cf27d5214db77707e3448ab23cc92a))
* tensorrt fp16 and int8 selector ([36c7488](https://github.com/jolibrain/deepdetect/commit/36c7488ee818a780c8d5aa82223c650f3b805316))
* **torch/native:** prevent loading weights before instanciating native model ([b15d767](https://github.com/jolibrain/deepdetect/commit/b15d7672240a6d4261b1cc5ec2f1b7139350eaf3))
* **torch/timeseries:** do not double read query data ([d54f60d](https://github.com/jolibrain/deepdetect/commit/d54f60df9fa453b403d393b87e591294eae29b21))

### 0.10.1 (2020-10-09)


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
