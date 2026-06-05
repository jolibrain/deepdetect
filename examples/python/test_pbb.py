import argparse
from pathlib import Path

import deepdetect


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    model_repository = Path(__file__).resolve().parent / "models" / "classifier"
    model_repository.mkdir(parents=True, exist_ok=True)

    dd = deepdetect.DeepDetect()
    print(dd.build_info)
    print(dd.info())

    try:
        with dd.create_service(
            "classifier",
            model={"repository": str(model_repository)},
            mllib="torch",
            input_parameters={"connector": "image"},
            mllib_parameters={
                "template": "resnet18",
                "nclasses": 10,
                # CPU is the default because test.log shows a local cuDNN
                # engine incompatibility. Use --gpu to test error handling.
                "gpu": args.gpu,
            },
            output_parameters={},
        ) as service:
            result = service.predict(
                ["cat.jpg"],
                input_parameters={"width": 224, "height": 224},
                output_parameters={"best": 5},
            )
            print(result)
    except deepdetect.DeepDetectError as error:
        print(
            f"DeepDetect error: status={error.status_code} "
            f"dd_code={error.dd_code} message={error.message}"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
