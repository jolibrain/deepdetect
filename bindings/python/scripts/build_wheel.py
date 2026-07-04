from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_PROJECT = REPO_ROOT / "bindings" / "python"
TORCH_VERSION = "2.12.1"
TORCH_DEPENDENCY = "torch==2.12.1"
TORCHVISION_DEPENDENCY = "torchvision==0.27.1"

AUDITWHEEL_EXCLUDES = [
    "libc10.so",
    "libc10_cuda.so",
    "libtorch.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libtorch_cuda_linalg.so",
    "libtorch_global_deps.so",
    "libtorch_python.so",
    "libtorch_nvshmem.so",
    "libcudart.so.13",
    "libcupti.so.13",
    "libcublas.so.13",
    "libcublasLt.so.13",
    "libcufft.so.12",
    "libcufile.so.0",
    "libcurand.so.10",
    "libcusolver.so.12",
    "libcusparse.so.12",
    "libcusparseLt.so.0",
    "libnvJitLink.so.13",
    "libnvrtc.so.13",
    "libnvToolsExt.so.1",
    "libcudnn.so.9",
    "libnccl.so.2",
    "libnvshmem_host.so.3",
]


def run(command: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def torch_cmake_prefix(torch_mode: str) -> str:
    torch = importlib.import_module("torch")
    version = str(getattr(torch, "__version__", ""))
    public_version = version.split("+", 1)[0]
    if public_version != TORCH_VERSION:
        raise SystemExit(
            f"This wheel build is pinned to {TORCH_DEPENDENCY}. "
            f"The active interpreter has torch {version or 'with an unknown version'}."
        )
    torch_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    if torch_mode == "cpu" and torch_cuda:
        raise SystemExit(
            "CPU wheel builds require a CPU-only torch 2.12.1 installation. "
            f"The active interpreter has CUDA torch {version}."
        )
    if torch_mode == "gpu" and not torch_cuda:
        raise SystemExit(
            "GPU wheel builds require a CUDA-enabled torch 2.12.1 installation. "
            f"The active interpreter has torch {version}."
        )
    return str(torch.utils.cmake_prefix_path)


def site_packages_dirs() -> list[Path]:
    paths = {
        sysconfig.get_path("purelib"),
        sysconfig.get_path("platlib"),
    }
    return [Path(path) for path in paths if path]


def wheel_distribution_prefix(distribution_name: str) -> str:
    return distribution_name.replace("-", "_").replace(".", "_")


def remove_matching_wheels(directory: Path, distribution_name: str) -> None:
    prefix = wheel_distribution_prefix(distribution_name)
    for wheel in directory.glob(f"{prefix}-*.whl"):
        wheel.unlink()


def stage_protobuf_libraries(build_dir: Path, sdk_prefix: Path) -> None:
    protobuf_build_dir = build_dir / "protobuf" / "src" / "protobuf-build"
    if not protobuf_build_dir.exists():
        return

    library_dir = sdk_prefix / "lib"
    library_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "libprotobuf.so*",
        "libprotobuf-lite.so*",
        "libprotoc.so*",
        "libprotobuf*.dylib*",
        "libprotoc*.dylib*",
    ):
        for library in protobuf_build_dir.glob(pattern):
            if library.is_file() or library.is_symlink():
                shutil.copy2(library, library_dir / library.name)


def assert_bundled_runtime_libraries(wheel: Path) -> None:
    import zipfile

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    required = {
        "deepdetect/libdeepdetect.so.0",
    }
    missing = sorted(required - names)
    for pattern in (
        "deepdetect/libprotobuf.so.",
        "deepdetect/libprotobuf-lite.so.",
        "deepdetect/libprotoc.so.",
    ):
        if not any(name.startswith(pattern) for name in names):
            missing.append(f"{pattern}*")
    if missing:
        raise SystemExit(
            f"Wheel {wheel} is missing required bundled runtime libraries: "
            + ", ".join(missing)
        )


def nvidia_runtime_paths() -> tuple[Path | None, Path | None, Path | None, list[Path]]:
    cudnn_root: Path | None = None
    cudnn_include: Path | None = None
    cudnn_library: Path | None = None
    library_paths: list[Path] = []
    for site_packages in site_packages_dirs():
        nvidia_root = site_packages / "nvidia"
        if not nvidia_root.exists():
            continue
        candidate_cudnn_root = nvidia_root / "cudnn"
        candidate_cudnn_include = candidate_cudnn_root / "include"
        candidate_cudnn_library = candidate_cudnn_root / "lib" / "libcudnn.so.9"
        if cudnn_root is None and (candidate_cudnn_include / "cudnn.h").exists():
            cudnn_root = candidate_cudnn_root
            cudnn_include = candidate_cudnn_include
        if cudnn_library is None and candidate_cudnn_library.exists():
            cudnn_library = candidate_cudnn_library
        for library_dir in nvidia_root.glob("*/lib"):
            if library_dir.is_dir():
                library_paths.append(library_dir)
    deduped_library_paths = list(dict.fromkeys(library_paths))
    return cudnn_root, cudnn_include, cudnn_library, deduped_library_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Build a bundled DeepDetect Python wheel against {TORCH_DEPENDENCY}"
    )
    parser.add_argument("--build-dir", default="build/python-wheel")
    parser.add_argument("--sdk-prefix", default="build/python-wheel/install")
    parser.add_argument("--wheel-dir", default="dist/python")
    parser.add_argument(
        "--torch-mode",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="native torch mode to build against",
    )
    parser.add_argument(
        "--distribution-name",
        default="deepdetect",
        help="wheel distribution name; import package remains deepdetect",
    )
    parser.add_argument(
        "--distribution-version",
        default="",
        help="wheel distribution version; defaults to pyproject.toml",
    )
    parser.add_argument(
        "--torch-dependency",
        default=TORCH_DEPENDENCY,
        help="dependency string written to generated wheel metadata",
    )
    parser.add_argument(
        "--torchvision-dependency",
        default=TORCHVISION_DEPENDENCY,
        help="torchvision dependency string written to generated wheel metadata",
    )
    parser.add_argument("--config", default="Release")
    parser.add_argument("--jobs", default=str(os.cpu_count() or 2))
    parser.add_argument("--cmake", default=os.environ.get("CMAKE_COMMAND", "cmake"))
    parser.add_argument(
        "--cuda-architectures",
        default=os.environ.get("CMAKE_CUDA_ARCHITECTURES", "86"),
        help="CMake CUDA architecture list, for example 86 or 80;86",
    )
    parser.add_argument(
        "--cuda-compiler",
        default=os.environ.get("CMAKE_CUDA_COMPILER", "/usr/local/cuda/bin/nvcc"),
    )
    parser.add_argument("--torchvision-source-dir", default="")
    parser.add_argument("--cmake-library-path", default="")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="run auditwheel repair after building the linux wheel",
    )
    parser.add_argument("--no-repair", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-native-build", action="store_true")
    parser.add_argument("--reuse-raw-wheel", action="store_true")
    parser.add_argument("--no-build-isolation", action="store_true")
    return parser.parse_args()


def project_for_wheel(
    *,
    build_dir: Path,
    distribution_name: str,
    distribution_version: str,
    torch_dependency: str,
    torchvision_dependency: str,
) -> Path:
    if (
        distribution_name == "deepdetect"
        and not distribution_version
        and torch_dependency == TORCH_DEPENDENCY
        and torchvision_dependency == TORCHVISION_DEPENDENCY
    ):
        return PYTHON_PROJECT

    project_dir = build_dir / "python-project"
    if project_dir.exists():
        shutil.rmtree(project_dir)
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        "build",
        "dist",
        "*.egg-info",
    )
    shutil.copytree(PYTHON_PROJECT, project_dir, ignore=ignore)

    pyproject = project_dir / "pyproject.toml"
    contents = pyproject.read_text(encoding="utf-8")
    contents = contents.replace('name = "deepdetect"', f'name = "{distribution_name}"', 1)
    if distribution_version:
        contents = contents.replace(
            'version = "0.1.0"',
            f'version = "{distribution_version}"',
            1,
        )
    contents = contents.replace(
        f'"{TORCH_DEPENDENCY}"',
        f'"{torch_dependency}"',
        1,
    )
    contents = contents.replace(
        f'"{TORCHVISION_DEPENDENCY}"',
        f'"{torchvision_dependency}"',
        1,
    )
    pyproject.write_text(contents, encoding="utf-8")

    if distribution_version:
        init_file = project_dir / "deepdetect" / "__init__.py"
        init_contents = init_file.read_text(encoding="utf-8")
        init_contents = init_contents.replace(
            '__version__ = "0.1.0"',
            f'__version__ = "{distribution_version}"',
            1,
        )
        init_file.write_text(init_contents, encoding="utf-8")
    return project_dir


def main() -> None:
    args = parse_args()
    build_dir = (REPO_ROOT / args.build_dir).resolve()
    sdk_prefix = (REPO_ROOT / args.sdk_prefix).resolve()
    wheel_dir = (REPO_ROOT / args.wheel_dir).resolve()
    raw_wheel_dir = wheel_dir / "raw"
    cmake_prefix = torch_cmake_prefix(args.torch_mode)
    cudnn_root, cudnn_include, cudnn_library, nvidia_library_paths = (
        nvidia_runtime_paths()
        if args.torch_mode != "cpu"
        else (None, None, None, [])
    )

    env = os.environ.copy()
    #env.setdefault("CCACHE_DISABLE", "1")
    env["CMAKE_EXECUTABLE"] = str(Path(args.cmake).resolve())

    if not args.skip_native_build:
        configure_command = [
            args.cmake,
            "-S",
            str(REPO_ROOT),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={args.config}",
            "-DUSE_TORCH=ON",
            "-DUSE_PYTORCH_WORKER=ON",
            "-DUSE_PREBUILT_TORCH=ON",
            "-DUSE_TENSORRT=OFF",
            "-DUSE_XGBOOST=OFF",
            "-DUSE_DLIB=OFF",
            "-DUSE_NCNN=OFF",
            "-DUSE_SIMSEARCH=OFF",
            "-DUSE_HTTP_SERVER=OFF",
            "-DUSE_HTTP_SERVER_OATPP=OFF",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_TOOLS=OFF",
            "-DBUILD_PROTOBUF=ON",
            f"-DCMAKE_PREFIX_PATH={cmake_prefix}",
        ]
        if args.torch_mode == "cpu":
            configure_command.extend(
                [
                    "-DUSE_CPU_ONLY=ON",
                    "-DUSE_TORCH_CPU_ONLY=ON",
                    "-DUSE_CUDNN=OFF",
                ]
            )
        elif args.torch_mode == "gpu":
            configure_command.extend(
                [
                    "-DUSE_CPU_ONLY=OFF",
                    "-DUSE_TORCH_CPU_ONLY=OFF",
                ]
            )
        if args.torch_mode != "cpu" and args.cuda_architectures:
            configure_command.append(
                f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_architectures}"
            )
        if args.torch_mode != "cpu" and args.cuda_compiler:
            configure_command.append(f"-DCMAKE_CUDA_COMPILER={args.cuda_compiler}")
        if args.torchvision_source_dir:
            configure_command.append(
                f"-DTORCHVISION_SOURCE_DIR={Path(args.torchvision_source_dir).resolve()}"
            )
        if args.cmake_library_path:
            configure_command.append(f"-DCMAKE_LIBRARY_PATH={args.cmake_library_path}")
        elif nvidia_library_paths:
            configure_command.append(
                "-DCMAKE_LIBRARY_PATH="
                + ";".join(str(path) for path in nvidia_library_paths)
            )
        if cudnn_root is not None:
            configure_command.append(f"-DCUDNN_ROOT={cudnn_root}")
        if cudnn_include is not None:
            configure_command.append(f"-DCUDNN_INCLUDE={cudnn_include}")
            configure_command.append(f"-DCUDNN_INCLUDE_DIR={cudnn_include}")
        if cudnn_library is not None:
            configure_command.append(f"-DCUDNN_LIBRARY={cudnn_library}")

        run(configure_command, env=env)
        run([args.cmake, "--build", str(build_dir), "--parallel", args.jobs], env=env)
        run([args.cmake, "--install", str(build_dir), "--prefix", str(sdk_prefix)], env=env)
        stage_protobuf_libraries(build_dir, sdk_prefix)

    raw_wheel_dir.mkdir(parents=True, exist_ok=True)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    if not args.reuse_raw_wheel:
        remove_matching_wheels(raw_wheel_dir, args.distribution_name)
        remove_matching_wheels(wheel_dir, args.distribution_name)
        project_dir = project_for_wheel(
            build_dir=build_dir,
            distribution_name=args.distribution_name,
            distribution_version=args.distribution_version,
            torch_dependency=args.torch_dependency,
            torchvision_dependency=args.torchvision_dependency,
        )
        wheel_command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(project_dir),
            "--no-deps",
            "--wheel-dir",
            str(raw_wheel_dir),
            f"--config-settings=cmake.define.DeepDetect_DIR={sdk_prefix / 'lib' / 'cmake' / 'DeepDetect'}",
        ]
        if args.no_build_isolation:
            wheel_command.append("--no-build-isolation")
        run(wheel_command, env=env)

    wheel_prefix = wheel_distribution_prefix(args.distribution_name)
    wheels = sorted(
        raw_wheel_dir.glob(f"{wheel_prefix}-*.whl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not wheels:
        raise SystemExit(f"No deepdetect wheel was produced under {raw_wheel_dir}")
    raw_wheel = wheels[-1]

    if args.no_repair or not args.repair:
        destination = wheel_dir / raw_wheel.name
        shutil.copy2(raw_wheel, destination)
        assert_bundled_runtime_libraries(destination)
        print(destination)
        return

    auditwheel_command = [
        sys.executable,
        "-m",
        "auditwheel",
        "repair",
        "-w",
        str(wheel_dir),
    ]
    for library in AUDITWHEEL_EXCLUDES:
        auditwheel_command.extend(["--exclude", library])
    auditwheel_command.append(str(raw_wheel))
    auditwheel_env = env.copy()
    auditwheel_library_paths = [
        sdk_prefix / "lib",
        build_dir / "protobuf" / "src" / "protobuf-build",
    ]
    existing_library_path = auditwheel_env.get("LD_LIBRARY_PATH")
    if existing_library_path:
        auditwheel_library_paths.extend(
            Path(path) for path in existing_library_path.split(":") if path
        )
    auditwheel_env["LD_LIBRARY_PATH"] = ":".join(
        str(path) for path in auditwheel_library_paths
    )
    run(auditwheel_command, env=auditwheel_env)
    repaired_wheels = sorted(
        wheel_dir.glob(f"{wheel_prefix}-*.whl"),
        key=lambda path: path.stat().st_mtime,
    )
    if repaired_wheels:
        assert_bundled_runtime_libraries(repaired_wheels[-1])


if __name__ == "__main__":
    main()
