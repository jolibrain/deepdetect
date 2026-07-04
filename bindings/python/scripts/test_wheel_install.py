from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TORCH_DEPENDENCY = "torch==2.12.1"
TORCHVISION_DEPENDENCY = "torchvision==0.27.1"
TORCH_FIXTURES = {
    "resnet50_training_torch241_small": (
        "https://www.deepdetect.com/dd/examples/torch/"
        "resnet50_training_torch241_small.tar.gz"
    ),
    "fasterrcnn_train_torch111": (
        "https://www.deepdetect.com/dd/examples/torch/"
        "fasterrcnn_train_torch111_bs2.tar.gz"
    ),
    "yolox_train_torch": (
        "https://www.deepdetect.com/dd/examples/torch/yolox_train_torch.tar.gz"
    ),
    "deeplabv3_training_torch": (
        "https://www.deepdetect.com/dd/examples/torch/deeplabv3_training_torch.tar.gz"
    ),
    "segformer_training_torch": (
        "https://www.deepdetect.com/dd/examples/torch/segformer_training_torch.tar.gz"
    ),
}


def run(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def command_output(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> str:
    print("+", " ".join(command), flush=True)
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install and smoke-test a DeepDetect Python wheel"
    )
    parser.add_argument("--wheel-dir", required=True)
    parser.add_argument("--distribution-name", required=True)
    parser.add_argument("--venv-dir", required=True)
    parser.add_argument("--torch-dependency", default=TORCH_DEPENDENCY)
    parser.add_argument("--torchvision-dependency", default=TORCHVISION_DEPENDENCY)
    parser.add_argument("--torch-index-url", default="")
    parser.add_argument("--expected-cuda", type=parse_bool, required=True)
    parser.add_argument("--pytest-tests", default="bindings/python/tests")
    parser.add_argument(
        "--torch-fixtures-dir",
        default="",
        help="directory containing or receiving Torch integration fixtures",
    )
    parser.add_argument(
        "--download-torch-fixtures",
        action="store_true",
        help="download missing Torch integration fixtures before running pytest",
    )
    parser.add_argument("--venv-python", default=sys.executable)
    return parser.parse_args()


def pip_install_command(
    python: str,
    requirements: list[str],
    *,
    index_url: str = "",
) -> list[str]:
    command = [python, "-m", "pip", "install"]
    if index_url:
        command.extend(
            ["--index-url", index_url, "--extra-index-url", "https://pypi.org/simple"]
        )
    command.extend(requirements)
    return command


def wheel_distribution_prefix(distribution_name: str) -> str:
    return distribution_name.replace("-", "_").replace(".", "_")


def select_wheel(wheel_dir: Path, distribution_name: str) -> Path:
    prefix = wheel_distribution_prefix(distribution_name)
    wheels = sorted(
        wheel_dir.glob(f"{prefix}-*.whl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not wheels:
        raise SystemExit(
            f"No wheel for {distribution_name!r} was found under {wheel_dir}"
        )
    return wheels[-1]


def torchvision_smoke_script() -> str:
    return """
import torch
import torchvision
from torchvision.ops import nms as _torchvision_nms

print(f"torch={torch.__version__} torchvision={torchvision.__version__}")
"""


def smoke_script(expected_cuda: bool) -> str:
    return f"""
import json
import importlib.metadata
import subprocess
from pathlib import Path

import deepdetect
import torch

if {expected_cuda!r} and torch.cuda.device_count() <= 0:
    raise SystemExit("GPU wheel test expected at least one visible CUDA device")

package_dir = Path(deepdetect.__file__).resolve().parent
print(f"deepdetect package dir: {{package_dir}}")
print(
    "deepdetect native files:",
    json.dumps(sorted(path.name for path in package_dir.glob("*.so*")), indent=2),
)
protobuf_libraries = sorted(package_dir.glob("libprotobuf.so*"))
for path in [
    *sorted(package_dir.glob("_native*.so")),
    package_dir / "libdeepdetect.so.0",
    *protobuf_libraries,
]:
    if path.exists():
        print(f"readelf -d {{path}}")
        subprocess.run(["readelf", "-d", str(path)], check=False)
    else:
        print(f"missing expected native file: {{path}}")

dd = deepdetect.DeepDetect()
build_info = dd.build_info
print(json.dumps(build_info, indent=2, sort_keys=True))
if bool(build_info.get("cuda")) is not {expected_cuda!r}:
    raise SystemExit(
        f"Expected deepdetect cuda={expected_cuda!r}, got {{build_info.get('cuda')!r}}"
    )
print(json.dumps(dd.info(), indent=2, sort_keys=True))
torchvision_version = importlib.metadata.version("torchvision")
print(f"torch={{torch.__version__}} torchvision={{torchvision_version}}")
"""


def installed_package_dir(python: str) -> Path:
    output = command_output(
        [
            python,
            "-c",
            (
                "from pathlib import Path; "
                "import deepdetect; "
                "print(Path(deepdetect.__file__).resolve().parent)"
            ),
        ]
    )
    return Path(output)


def prepend_env_path(env: dict[str, str], name: str, path: Path) -> None:
    value = str(path)
    current = env.get(name)
    if not current:
        env[name] = value
        return
    parts = current.split(os.pathsep)
    if value not in parts:
        env[name] = os.pathsep.join([value, *parts])


def download_torch_fixtures(fixtures_dir: Path) -> None:
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    download_dir = fixtures_dir / ".downloads"
    download_dir.mkdir(exist_ok=True)
    for dataset, url in TORCH_FIXTURES.items():
        destination = fixtures_dir / dataset
        if destination.exists():
            continue
        archive = download_dir / Path(url).name
        if not archive.exists():
            print(f"Downloading {url} to {archive}", flush=True)
            urllib.request.urlretrieve(url, archive)
        print(f"Extracting {archive} into {fixtures_dir}", flush=True)
        with tarfile.open(archive) as tar:
            tar.extractall(fixtures_dir)


def main() -> None:
    args = parse_args()
    venv_dir = (REPO_ROOT / args.venv_dir).resolve()
    wheel_dir = (REPO_ROOT / args.wheel_dir).resolve()
    tests_path = (REPO_ROOT / args.pytest_tests).resolve()
    fixtures_dir = (
        (REPO_ROOT / args.torch_fixtures_dir).resolve()
        if args.torch_fixtures_dir
        else None
    )

    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    run([args.venv_python, "-m", "venv", str(venv_dir)])

    python = str(venv_dir / "bin" / "python")
    wheel = select_wheel(wheel_dir, args.distribution_name)
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run(
        pip_install_command(
            python,
            [
                "pytest>=7",
                "numpy>=1.23",
                "nvidia-ml-py>=12",
                "Pillow>=9",
                "PyYAML>=6",
                "rich>=13.7",
                "tqdm>=4.66",
            ],
        )
    )
    run(
        pip_install_command(
            python,
            [args.torch_dependency, args.torchvision_dependency],
            index_url=args.torch_index_url,
        )
    )
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            str(wheel),
        ]
    )
    native_package_dir = installed_package_dir(python)

    copied_tests = venv_dir / "wheel-tests"
    if tests_path.is_dir():
        shutil.copytree(
            tests_path,
            copied_tests,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
    else:
        copied_tests.mkdir()
        shutil.copy2(tests_path, copied_tests / tests_path.name)

    source_scripts = REPO_ROOT / "bindings" / "python" / "scripts"
    copied_scripts = venv_dir / "scripts"
    if source_scripts.is_dir():
        shutil.copytree(
            source_scripts,
            copied_scripts,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )

    test_env = os.environ.copy()
    prepend_env_path(test_env, "LD_LIBRARY_PATH", native_package_dir)
    print(f"Using LD_LIBRARY_PATH={test_env['LD_LIBRARY_PATH']}", flush=True)
    if fixtures_dir is not None:
        if args.download_torch_fixtures:
            download_torch_fixtures(fixtures_dir)
        test_env["DEEPDETECT_TORCH_FIXTURES"] = str(fixtures_dir)
    test_env["DEEPDETECT_EXPECTED_CUDA"] = "true" if args.expected_cuda else "false"

    run(
        [python, "-m", "pytest", "--import-mode=importlib", str(copied_tests)],
        cwd=venv_dir,
        env=test_env,
    )
    # Python torchvision and bundled native libtorchvision both register C++ ops.
    # Keep their smoke checks in separate interpreters to avoid duplicate schemas.
    run([python, "-c", torchvision_smoke_script()], cwd=venv_dir, env=test_env)
    run([python, "-c", smoke_script(args.expected_cuda)], cwd=venv_dir, env=test_env)


if __name__ == "__main__":
    main()
