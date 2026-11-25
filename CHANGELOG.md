# DeepDetect: Open Source Deep Learning Server & API (Changelog)

### [0.30.2](https://github.com/jolibrain/deepdetect/compare/v0.30.1...v0.30.2) (2026-11-16)


### Features

* add source-scale segmentation tiling and probability fusion ([799d346](https://github.com/jolibrain/deepdetect/commit/799d346fcbbcb1a8e414e4a830f2152df71a1345))
* automated onnx export of best checkponts ([e6b7dc2](https://github.com/jolibrain/deepdetect/commit/e6b7dc2b844e5dd7d7fdddaad8cc3d8b73a0f88f))
* c++ input connector inference with pytorch workers + updated docker images ([8fcee35](https://github.com/jolibrain/deepdetect/commit/8fcee35fc9177a33c057d68ed76a05e98690f705))
* **cli:** model export and test inference ([edf2335](https://github.com/jolibrain/deepdetect/commit/edf2335c8b01b1325451740e855a8d4bfef464a0))
* **cli:** show segmentation ground truth in Visdom ([e6dbc0d](https://github.com/jolibrain/deepdetect/commit/e6dbc0d151f2a40e230996294f0b0b0818d6a661))
* **pytorch:** support external semantic segmentation workers ([9ab16e0](https://github.com/jolibrain/deepdetect/commit/9ab16e08b0b0eec454aa8bcf0ea0cf31dad3a1ec))
* run env name in title ([ba0af90](https://github.com/jolibrain/deepdetect/commit/ba0af902c3152358d9482a944830b110604b8cd3))
* save training command in checkpoint directory ([37deb88](https://github.com/jolibrain/deepdetect/commit/37deb88260796280aeb0bf535a078ec9016ad32b))
* shared checkpoints manager between native and pytorch workers ([4c718cc](https://github.com/jolibrain/deepdetect/commit/4c718ccc0060eb44c08ae0ff3eb0d095374c504b))
* **tools:** retain only the best checkpoint pair ([5199334](https://github.com/jolibrain/deepdetect/commit/5199334b55fd6c2d2ef60d38237b3f6960c41821))


### Bug Fixes

* API predict_timeout ([0e1a737](https://github.com/jolibrain/deepdetect/commit/0e1a7377a81edaac5881dee61e22d85e8fb2e4a2))
* **cli:** match segmentation validation paths to backend ([99d9977](https://github.com/jolibrain/deepdetect/commit/99d99773a4ebc6af4ec3846fea7563557bdf6ee3))
