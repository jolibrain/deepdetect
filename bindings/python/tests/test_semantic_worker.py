import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deepdetect.cli.profiles import get_profile
from deepdetect.cli.checks import run_training_checks, validate_segmentation_lists
from deepdetect.cli.utils import stage_model
from deepdetect.pytorch_worker.sdk import PredictionContractError, validate_prediction_result
from deepdetect.pytorch_worker.segmentation import decode_mask


def result():
    return {"output_format": "semantic_segmentation_v1", "nclasses": 2,
            "results": [{"uri": "image", "imgsize": {"width": 2, "height": 1},
                         "vals": [0, 1], "confidences": {"best": [0.9, 0.8]}}]}


def test_semantic_contract():
    value = result()
    assert validate_prediction_result(value) == value
    for key, replacement in [
        ("vals", [0]), ("vals", [0, 2]), ("vals", [0.0, 1]),
        ("imgsize", {"width": 0, "height": 1}),
        ("confidences", {"best": [float("nan"), 0.5]}),
        ("confidences", {"best": [0.2]}), ("confidences", {"9": [0.2, 0.5]}),
        ("confidences", {"best": [-0.2, 0.5]}), ("uri", ""),
    ]:
        invalid = copy.deepcopy(value)
        invalid["results"][0][key] = replacement
        with pytest.raises(PredictionContractError):
            validate_prediction_result(invalid)
    del value["output_format"]
    with pytest.raises(PredictionContractError, match="probs"):
        validate_prediction_result(value)


def test_segmentation_profile_and_ignore_mapping(tmp_path):
    profile = get_profile("external-pytorch-segmentation")
    options = profile.train_defaults()
    options.update(train_data=tmp_path / "train.txt", test_data=[tmp_path / "train.txt"])
    Image.fromarray(np.array([[0, 1, 255]], dtype=np.uint8)).save(tmp_path / "mask.png")
    options["train_data"].write_text("image.png mask.png\n")
    assert run_training_checks(profile.task, options)[-1]["checked_masks"] == 2
    with pytest.raises(ValueError, match="unknown mask"):
        validate_segmentation_lists([options["train_data"]], 2)
    assert validate_segmentation_lists(
        [options["train_data"]], 2, label_mapping={255: 1}
    )["checked_masks"] == 1
    options["service_mllib"]["entrypoint"] = "/tmp/worker.py"
    options["mllib"] = {"segmentation": {"dice_weight": 0}}
    assert profile.service_parameters(options)["mllib_parameters"]["task"] == "segmentation"
    assert profile.train_parameters(options)["mllib_parameters"]["segmentation"]["dice_weight"] == 0


def test_explicit_weights_are_distinct_from_resume(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "checkpoint-3.pt").write_bytes(b"checkpoint")
    weights = tmp_path / "mit.pt"
    weights.write_bytes(b"pretrained")
    assert stage_model(weights, repository, reuse_checkpoint=False).read_bytes() == b"pretrained"
    assert stage_model(weights, repository).read_bytes() == b"checkpoint"


def test_unmapped_colors_and_noninteger_mappings_fail():
    # Unknown colors must not silently become an ignore ID of -1.
    with pytest.raises(ValueError, match="unknown mask colors"):
        decode_mask(np.zeros((2, 2, 3), dtype=np.uint8), 2,
                    ignore_label=-1, color_mapping={"255,255,255": 1})
    with pytest.raises(ValueError, match="integer IDs"):
        decode_mask(np.zeros((2, 2), dtype=np.uint8), 2, label_mapping={0: 1.5})


@pytest.mark.skipif(not os.environ.get("DD_SEMANTIC_NATIVE"), reason="requires rebuilt wheel")
def test_native_semantic_serialization(tmp_path):
    import deepdetect

    worker = Path(__file__).parent / "fixtures" / "semantic_worker.py"
    dd = deepdetect.DeepDetect()
    with dd.create_service(
        "semantic-fixture", mllib="pytorch", model={"repository": str(tmp_path)},
        input_parameters={"connector": "image", "segmentation": True},
        mllib_parameters={"task": "segmentation", "nclasses": 2,
                          "entrypoint": str(worker), "python": sys.executable},
    ) as service:
        response = service.predict(["fixture.png"])
        prediction = response["predictions"][0]
        assert prediction["vals"] == [0, 1, 0, 1, 0, 1]
        assert prediction["imgsize"] == {"width": 3, "height": 2}
        assert prediction["confidences"]["best"] == [0.75] * 6
        assert prediction["confidences"]["0"] == [1, 0, 1, 0, 1, 0]
