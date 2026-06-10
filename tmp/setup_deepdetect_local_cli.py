#!/usr/bin/env python3
from __future__ import annotations

import site
import sys
from pathlib import Path


LOCAL_DEEPDETECT = Path(
    "/data1/beniz/code/deepdetect/bindings/python/deepdetect"
)
PTH_NAME = "deepdetect_local_cli.pth"


def candidate_site_packages() -> list[Path]:
    paths: list[Path] = []
    for getter in (site.getsitepackages,):
        try:
            paths.extend(Path(path) for path in getter())
        except AttributeError:
            pass
    paths.append(Path(site.getusersitepackages()))
    return paths


def choose_site_packages() -> Path:
    prefix = Path(sys.prefix).resolve()
    for path in candidate_site_packages():
        resolved = path.resolve()
        if prefix in (resolved, *resolved.parents):
            return path
    return Path(site.getusersitepackages())


def main() -> int:
    if not LOCAL_DEEPDETECT.is_dir():
        print(f"local DeepDetect package not found: {LOCAL_DEEPDETECT}", file=sys.stderr)
        return 1

    site_packages = choose_site_packages()
    site_packages.mkdir(parents=True, exist_ok=True)
    pth = site_packages / PTH_NAME
    pth.write_text(
        "import deepdetect; "
        f"deepdetect.__path__.append({str(LOCAL_DEEPDETECT)!r})\n",
        encoding="utf-8",
    )

    print(f"wrote {pth}")
    print()
    print("Verify from outside the source tree:")
    print("  cd /tmp")
    print(
        "  PYTHONPATH= python3 -c "
        "\"import deepdetect, deepdetect._native, deepdetect.cli.main, pathlib; "
        "print(pathlib.Path(deepdetect.__file__).resolve())\""
    )
    print()
    print("Then run local CLI code against the installed wheel:")
    print("  cd /tmp")
    print("  PYTHONPATH= python3 -m deepdetect.cli.main --help")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
