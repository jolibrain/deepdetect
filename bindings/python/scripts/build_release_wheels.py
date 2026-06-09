from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_WHEEL = REPO_ROOT / "bindings" / "python" / "scripts" / "build_wheel.py"
BUILD_REQUIREMENTS = ["auditwheel>=6", "scikit-build-core", "pybind11"]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CPU and GPU DeepDetect Python wheel variants"
    )
    parser.add_argument(
        "--cpu-python",
        default="",
        help="existing CPU-only Python interpreter; created automatically when omitted",
    )
    parser.add_argument(
        "--gpu-python",
        default="",
        help="existing CUDA-enabled Python interpreter; created automatically when omitted",
    )
    parser.add_argument(
        "--venv-python",
        default=sys.executable,
        help="Python interpreter used to create managed build environments",
    )
    parser.add_argument(
        "--env-root",
        default="build/python-wheel-envs",
        help="directory for managed temporary build environments",
    )
    parser.add_argument(
        "--keep-envs",
        action="store_true",
        help="keep managed build environments after the build",
    )
    parser.add_argument("--cpu-name", default="deepdetect-cpu")
    parser.add_argument("--gpu-name", default="deepdetect-gpu")
    parser.add_argument("--cpu-torch-dependency", default="torch==2.12.*")
    parser.add_argument("--gpu-torch-dependency", default="torch==2.12.*")
    parser.add_argument(
        "--cpu-torch-index-url",
        default="https://download.pytorch.org/whl/cpu",
        help="pip index URL used when installing CPU torch into a managed env",
    )
    parser.add_argument(
        "--gpu-torch-index-url",
        default="",
        help="optional pip index URL used when installing GPU torch into a managed env",
    )
    parser.add_argument("--wheel-dir", default="dist/python/release")
    parser.add_argument("--cmake", default=os.environ.get("CMAKE_COMMAND", "cmake"))
    parser.add_argument("--jobs", default=str(os.cpu_count() or 2))
    parser.add_argument("--config", default="Release")
    parser.add_argument(
        "--cuda-architectures",
        default=os.environ.get("CMAKE_CUDA_ARCHITECTURES", "86"),
    )
    parser.add_argument(
        "--cuda-compiler",
        default=os.environ.get("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc"),
    )
    parser.add_argument("--torchvision-source-dir", default="")
    parser.add_argument("--no-build-isolation", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    return parser.parse_args()


def managed_python(env_dir: Path) -> str:
    return str(env_dir / "bin" / "python")


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


def create_managed_env(
    *,
    args: argparse.Namespace,
    mode: str,
    torch_dependency: str,
    torch_index_url: str,
) -> str:
    env_dir = (REPO_ROOT / args.env_root / mode).resolve()
    if env_dir.exists():
        shutil.rmtree(env_dir)
    run([args.venv_python, "-m", "venv", str(env_dir)])
    python = managed_python(env_dir)
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run(pip_install_command(python, BUILD_REQUIREMENTS))
    run(
        pip_install_command(
            python,
            [torch_dependency],
            index_url=torch_index_url,
        )
    )
    return python


def remove_managed_env(args: argparse.Namespace, mode: str) -> None:
    env_dir = (REPO_ROOT / args.env_root / mode).resolve()
    if env_dir.exists():
        shutil.rmtree(env_dir)


def wheel_command(
    *,
    python: str,
    mode: str,
    name: str,
    torch_dependency: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python,
        str(BUILD_WHEEL),
        "--torch-mode",
        mode,
        "--distribution-name",
        name,
        "--torch-dependency",
        torch_dependency,
        "--build-dir",
        f"build/python-wheel-{mode}",
        "--sdk-prefix",
        f"build/python-wheel-{mode}/install",
        "--wheel-dir",
        str(Path(args.wheel_dir) / mode),
        "--cmake",
        args.cmake,
        "--jobs",
        args.jobs,
        "--config",
        args.config,
    ]
    if args.no_build_isolation:
        command.append("--no-build-isolation")
    if args.repair:
        command.append("--repair")
    if args.torchvision_source_dir:
        command.extend(["--torchvision-source-dir", args.torchvision_source_dir])
    if mode == "gpu":
        command.extend(
            [
                "--cuda-architectures",
                args.cuda_architectures,
                "--cuda-compiler",
                args.cuda_compiler,
            ]
        )
    return command


def build_variant(
    *,
    args: argparse.Namespace,
    mode: str,
    python: str,
    name: str,
    torch_dependency: str,
    torch_index_url: str,
) -> None:
    managed = not python
    if managed:
        python = create_managed_env(
            args=args,
            mode=mode,
            torch_dependency=torch_dependency,
            torch_index_url=torch_index_url,
        )
    try:
        run(
            wheel_command(
                python=python,
                mode=mode,
                name=name,
                torch_dependency=torch_dependency,
                args=args,
            )
        )
    finally:
        if managed and not args.keep_envs:
            remove_managed_env(args, mode)


def main() -> None:
    args = parse_args()
    if not args.skip_cpu:
        build_variant(
            args=args,
            mode="cpu",
            python=args.cpu_python,
            name=args.cpu_name,
            torch_dependency=args.cpu_torch_dependency,
            torch_index_url=args.cpu_torch_index_url,
        )
    if not args.skip_gpu:
        build_variant(
            args=args,
            mode="gpu",
            python=args.gpu_python,
            name=args.gpu_name,
            torch_dependency=args.gpu_torch_dependency,
            torch_index_url=args.gpu_torch_index_url,
        )


if __name__ == "__main__":
    main()
