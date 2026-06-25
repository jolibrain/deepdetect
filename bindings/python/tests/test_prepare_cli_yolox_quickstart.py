from __future__ import annotations

import importlib.util
import sys
import tarfile
from pathlib import Path

import pytest

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from deepdetect.cli import config as cli_config


SCRIPT = PYTHON_ROOT / "scripts" / "prepare_cli_yolox_quickstart.py"
SPEC = importlib.util.spec_from_file_location("prepare_cli_yolox_quickstart", SCRIPT)
assert SPEC is not None
quickstart = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(quickstart)


def write_archive(source: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname=source.name)


def prepare_archives(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    downloads = tmp_path / "downloads"
    downloads.mkdir()

    yolox = source / "yolox_train_torch"
    yolox.mkdir(parents=True)
    (yolox / "yolox-nano_cls2.pt").write_bytes(b"weights")

    fasterrcnn = source / "fasterrcnn_train_torch111"
    (fasterrcnn / "imgs").mkdir(parents=True)
    (fasterrcnn / "bboxes").mkdir()
    (fasterrcnn / "imgs" / "sample.jpg").write_bytes(b"image")
    (fasterrcnn / "bboxes" / "sample.txt").write_text(
        "1 0 0 10 10\n",
        encoding="utf-8",
    )
    list_line = (
        "../examples/torch/fasterrcnn_train_torch111/imgs/sample.jpg "
        "../examples/torch/fasterrcnn_train_torch111/bboxes/sample.txt\n"
    )
    (fasterrcnn / "train.txt").write_text(list_line, encoding="utf-8")
    (fasterrcnn / "test.txt").write_text(list_line, encoding="utf-8")

    write_archive(yolox, downloads / "yolox_train_torch.tar.gz")
    write_archive(fasterrcnn, downloads / "fasterrcnn_train_torch111_bs2.tar.gz")
    return downloads


def test_prepare_cli_yolox_quickstart_rewrites_lists_and_writes_config(tmp_path):
    downloads = prepare_archives(tmp_path)
    output = tmp_path / "quickstart"

    config_path, image_path = quickstart.prepare(
        output,
        download_dir=downloads,
        force=False,
    )

    assert config_path == output / "yolox-quickstart.yaml"
    assert image_path == (
        output / "fixtures/fasterrcnn_train_torch111/imgs/sample.jpg"
    ).resolve()

    train_list = output / "fixtures/fasterrcnn_train_torch111/train.txt"
    train_line = train_list.read_text(encoding="utf-8").strip()
    image, bbox = train_line.split()
    assert image == str(image_path)
    assert bbox == str(
        (output / "fixtures/fasterrcnn_train_torch111/bboxes/sample.txt").resolve()
    )

    values = cli_config.load_config(config_path)
    assert values["train_data"] == str(train_list.resolve())
    assert values["test_data"] == [
        str((output / "fixtures/fasterrcnn_train_torch111/test.txt").resolve())
    ]
    assert values["weights"] == str(
        (output / "fixtures/yolox_train_torch/yolox-nano_cls2.pt").resolve()
    )
    assert values["repository"] == str((output / "model-repository").resolve())
    assert values["iterations"] == 3
    assert values["test_interval"] == 3
    assert values["visdom"] is True
    assert values["visdom_results_count"] == 2


def test_prepare_cli_yolox_quickstart_requires_force_for_existing_output(tmp_path):
    downloads = prepare_archives(tmp_path)
    output = tmp_path / "quickstart"
    output.mkdir()

    with pytest.raises(FileExistsError):
        quickstart.prepare(output, download_dir=downloads, force=False)

    config_path, _ = quickstart.prepare(output, download_dir=downloads, force=True)
    assert config_path.is_file()
