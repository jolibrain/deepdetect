#!/usr/bin/env python3
"""Smoke test and minimal example for the embedded DeepDetect package."""

from __future__ import annotations

import argparse
import json

import deepdetect


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-repository", help="Torch model repository")
    parser.add_argument("--image", help="Image path or URI to predict")
    parser.add_argument("--service", default="python-smoke-test")
    parser.add_argument("--template", default="resnet18")
    parser.add_argument("--nclasses", type=int)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    dd = deepdetect.DeepDetect()
    print("DeepDetect build:")
    print(json.dumps(dd.build_info, indent=2, sort_keys=True))
    print("DeepDetect server:")
    print(json.dumps(dd.info(), indent=2, sort_keys=True))

    if not args.model_repository:
        print("Native import and /info smoke test passed.")
        return

    if not args.image:
        parser.error("--image is required with --model-repository")

    mllib_parameters: dict[str, object] = {
        "template": args.template,
        "gpu": args.gpu,
    }
    if args.nclasses is not None:
        mllib_parameters["nclasses"] = args.nclasses

    with dd.create_service(
        args.service,
        model={"repository": args.model_repository},
        mllib="torch",
        input_parameters={"connector": "image"},
        mllib_parameters=mllib_parameters,
        output_parameters={},
    ) as service:
        result = service.predict(
            [args.image],
            output_parameters={"best": 5},
        )
        print("Prediction:")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
