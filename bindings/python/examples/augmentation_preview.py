"""Synthetic preview and bounded CPU benchmark, with no DeepDetect service."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from deepdetect.augmentation import Augmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    y, x = np.mgrid[:512, :512]
    image = np.stack([x // 2, y // 2, (x + y) // 4], axis=-1).astype("uint8")
    image[np.abs(y - x) < 2] = 0
    mask = (np.abs(y - x) < 2).astype("uint8")
    policy = Augmenter(
        {
            "mirror": True,
            "rotate": True,
            "distort": {"prob": 1, "contrast": True},
            "noise": {"probabilities": {"gauss_blur": 0.1}},
        }
    )
    result = policy(image, seed=42, masks={"lines": mask})
    clean = policy.apply(image, plan=result.plan, appearance=False)
    for name, array in [
        ("original", image),
        ("geometry", clean.image),
        ("augmented", result.image),
    ]:
        Image.fromarray(array).save(args.output / (name + ".png"))
    (args.output / "plan.json").write_text(result.plan.to_json() + "\n")
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>DeepDetect augmentation</title><h1>Original / geometry / augmented</h1><div style="display:flex">'
        + "".join(
            f'<figure><figcaption>{n}</figcaption><img style="width:100%" src="{n}.png"></figure>'
            for n in ["original", "geometry", "augmented"]
        )
        + '</div><a href="plan.json">Replay plan</a>'
    )
    if args.benchmark:
        count = 48
        metrics = {}
        operations = {
            "identity_copy": lambda i: Augmenter()(image, seed=i),
            "one_call": lambda i: policy(image, seed=i),
            "replay": lambda i: policy.apply(image, plan=result.plan),
        }
        for name, fn in operations.items():
            for i in range(3):
                fn(i)
            start = time.perf_counter()
            for i in range(count):
                fn(i)
            metrics[name + "_crops_per_second"] = count / (time.perf_counter() - start)
        for workers in (1, 4):
            with ThreadPoolExecutor(workers) as executor:
                start = time.perf_counter()
                list(executor.map(operations["one_call"], range(count)))
                metrics[f"threads_{workers}_crops_per_second"] = count / (
                    time.perf_counter() - start
                )
        metrics["build"] = policy.build_info()
        (args.output / "benchmark.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
