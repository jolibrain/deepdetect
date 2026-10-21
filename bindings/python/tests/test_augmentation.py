"""Native tests: require the built augmentation extension, never a mock."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from PIL import Image
from deepdetect.augmentation import Augmenter, AugmentationPlan

DATA = Path(__file__).parent / "data/augmentation_legacy"


@pytest.mark.skipif(
    not os.environ.get("DD_AUGMENTATION_NATIVE"),
    reason="requires installed native wheel",
)
@pytest.mark.parametrize("runtime_first", [False, True])
def test_runtime_import_order(runtime_first):
    # Separate processes expose loader/constructor ordering errors hidden when
    # the main test process already has OpenCV or the model runtime loaded.
    operations = [
        "from deepdetect.augmentation import Augmenter; "
        "import numpy as np; "
        "Augmenter()(np.zeros((8, 8, 3), dtype=np.uint8), seed=1)",
        "from deepdetect import DeepDetect; DeepDetect()",
    ]
    if runtime_first:
        operations.reverse()
    subprocess.run([sys.executable, "-c", "; ".join(operations)], check=True)


@pytest.fixture
def image():
    y, x = np.mgrid[:64, :64]
    return np.stack([x * 3, y * 3, (x + y) * 2], axis=-1).astype("uint8")


def test_identity_ownership(image):
    image.setflags(write=False)
    result = Augmenter()(image, seed=1)
    np.testing.assert_array_equal(result.image, image)
    assert result.in_image.all() and not result.occluded.any()
    assert not np.shares_memory(result.image, image)
    result.image[:] = 0
    assert image.any()


@pytest.mark.parametrize("seed", [11, 42, 23124534])
def test_legacy_pre_refactor_fixtures(seed):
    if Augmenter.build_info()["opencv"] != "4.6.0":
        pytest.skip("binary JPEG/equalization fixture captured with OpenCV 4.6.0")
    im = np.asarray(Image.open(DATA / "input.png"))[:, :, ::-1]
    mask = np.asarray(Image.open(DATA / "mask.png"))
    policy = Augmenter(
        dict(
            mirror=True,
            rotate=True,
            crop_size=48,
            cutout=0.3,
            geometry={"prob": 0.3},
            noise={"prob": 0.4},
            distort={"prob": 0.4},
        ),
        semantics="legacy_v1",
        color_order="BGR",
    )
    result = policy(im, seed=seed, masks={"labels": mask})
    expected = np.asarray(Image.open(DATA / f"{seed}-image.png"))[:, :, ::-1]
    np.testing.assert_array_equal(result.image, expected)
    np.testing.assert_array_equal(
        result.masks["labels"], np.asarray(Image.open(DATA / f"{seed}-mask.png"))
    )
    replay = policy.apply(
        im,
        plan=AugmentationPlan.from_json(result.plan.to_json()),
        masks={"labels": mask},
    )
    np.testing.assert_array_equal(replay.image, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_geometry_and_labels(image, dtype):
    mask = (
        np.arange(4096).reshape(64, 64) % (256 if dtype == np.uint8 else 4096)
    ).astype(dtype)
    policy = Augmenter({"mirror": True, "rotate": True, "crop_size": 48})
    for seed in range(8):
        r = policy(image, seed=seed, masks={"label": mask})
        yy, xx = np.mgrid[:48, :48]
        coords = np.stack([xx.ravel(), yy.ravel(), np.ones(48 * 48)])
        original = np.linalg.inv(r.source_to_output) @ coords
        ox, oy = np.rint(original[:2]).astype(int)
        np.testing.assert_array_equal(r.masks["label"].ravel(), mask[oy, ox])
        np.testing.assert_array_equal(r.image.reshape(-1, 3), image[oy, ox])
        assert r.in_image.all()


@pytest.mark.parametrize("padding", ["constant", "mirrored", "repeat_nearest"])
def test_perspective_validity(image, padding):
    policy = Augmenter(
        {
            "geometry": {
                "prob": 1,
                "pad_mode": padding,
                "zoom_in": False,
                "zoom_factor": 0.8,
            }
        }
    )
    r = policy(image, seed=42, masks={"label": np.ones((64, 64), np.uint8)})
    assert r.in_image.any() and not r.in_image.all()
    assert set(np.unique(r.masks["label"])) <= {0, 1}
    # Validity never grants trusted support to synthetic reflected pixels.
    y, x = np.mgrid[:64, :64]
    c = np.linalg.inv(r.source_to_output) @ np.stack(
        [x.ravel(), y.ravel(), np.ones(4096)]
    )
    c = np.rint(c[:2] / c[2]).astype(int)
    expected = ((c[0] >= 0) & (c[0] < 64) & (c[1] >= 0) & (c[1] < 64)).reshape(64, 64)
    np.testing.assert_array_equal(r.in_image, expected)


def test_explicit_negative_brightness_and_legacy_range(image):
    policy = Augmenter(
        {"distort": {"prob": 1, "brightness": True, "brightness_range": [-32, -32]}}
    )
    r = policy(image, seed=2)
    np.testing.assert_array_equal(r.image, np.maximum(image.astype(int) - 32, 0))
    legacy = Augmenter({"distort": {"prob": 1}}, semantics="legacy_v1")(image, seed=2)
    values = {k: v for k, v in legacy.plan.metadata["record"]["draws"]}
    for key, value in values.items():
        if "distort_params" in key:
            assert 0 <= value <= 1


def test_rgb_bgr_parity(image):
    cfg = {
        "noise": {"prob": 1, "hist_eq": True},
        "distort": {"prob": 1, "saturation": True},
    }
    rgb = Augmenter(cfg)(image, seed=3).image
    bgr = Augmenter(cfg, color_order="BGR")(image[:, :, ::-1], seed=3).image
    np.testing.assert_array_equal(rgb, bgr[:, :, ::-1])


def test_clean_cutout_and_threaded_replay(image):
    policy = Augmenter(
        {
            "mirror": True,
            "rotate": True,
            "crop_size": 48,
            "cutout": 1,
            "noise": {"prob": 1, "saltpepper": True},
            "distort": {"prob": 1, "channel_order": True},
        }
    )
    mask = np.ones((64, 64), np.uint16) * 1000
    r = policy(image, seed=123, masks={"label": mask})
    assert r.occluded.any()
    clean = policy.apply(image, plan=r.plan, masks={"label": mask}, appearance=False)
    assert not clean.occluded.any()
    np.testing.assert_array_equal(clean.masks["label"], r.masks["label"])
    np.testing.assert_array_equal(clean.source_to_output, r.source_to_output)
    with ThreadPoolExecutor(4) as pool:
        results = list(
            pool.map(lambda _: policy.apply(image, plan=r.plan).image, range(12))
        )
    for output in results:
        np.testing.assert_array_equal(output, r.image)
    assert len(r.plan.to_json()) < 15000  # stochastic fields must stay compact


def test_sample_matches_one_call_and_rng_independence(image):
    p = Augmenter({"cutout": 1, "noise": {"prob": 1, "jpg": True}})
    plan = p.sample(list(image.shape), seed=9)
    expected = p(image, seed=9)
    np.random.seed(100)
    state = np.random.get_state()
    actual = p.apply(image, plan=plan)
    np.testing.assert_array_equal(actual.image, expected.image)
    assert plan.to_json() == expected.plan.to_json()
    after = np.random.get_state()
    np.testing.assert_equal(state, after)


def test_operation_probabilities_and_grayscale():
    p = Augmenter({"noise": {"probabilities": {"inverse": 1, "posterize": 0}}})
    r = p(np.full((8, 8), 10, np.uint8), seed=1)
    assert np.all(r.image == 245)
    assert r.plan.metadata["record"]["operations"] == ["applyNoiseInverse"]


@pytest.mark.parametrize(
    "config",
    [
        {"mystery": True},
        {"cutout": float("nan")},
        {"noise": {"prob": 0.1}},
        {"noise": {"prob": 2}},
        {"mirror": 1},
        {"crop_size": 0},
        {"distort": {"brightness_range": [2, 1]}},
        {"geometry": {"pad_mode": "unknown"}},
        {"noise": {"inverse": False, "probabilities": {"inverse": 1}}},
    ],
)
def test_invalid_config(config):
    with pytest.raises((TypeError, ValueError)):
        Augmenter(config)


def test_bad_arrays_and_plans(image):
    p = Augmenter()
    plan = p.sample(image.shape, seed=1)
    for bad in [image.astype(float), image[:, :, :2], np.empty((0, 2), np.uint8)]:
        with pytest.raises((ValueError, TypeError)):
            p(bad, seed=1)
    with pytest.raises(ValueError):
        p(image, seed=1, masks={"x": np.zeros((2, 2), np.uint8)})
    with pytest.raises(TypeError):
        p(image, seed=1, masks={"x": np.zeros((64, 64), np.int64)})
    with pytest.raises(ValueError):
        Augmenter({"crop_size": 128})(image, seed=1)
    with pytest.raises(ValueError):
        p.apply(image[:32], plan=plan)
    with pytest.raises(ValueError):
        Augmenter({"mirror": True}).apply(image, plan=plan)
    broken = plan.metadata
    broken["build"]["implementation"] = "other"
    with pytest.raises(ValueError):
        p.apply(image, plan=AugmentationPlan.from_json(json.dumps(broken)))


def test_import_does_not_initialize_runtime():
    code = 'from deepdetect.augmentation import Augmenter; import sys; Augmenter.build_info(); assert "torch" not in sys.modules; assert "deepdetect._native" not in sys.modules'
    subprocess.run([sys.executable, "-c", code], check=True)


def test_replay_rejects_truncated_random_choices(image):
    policy = Augmenter({"mirror": True})
    data = policy.sample(image.shape, seed=1).metadata
    data["record"]["draws"] = []
    with pytest.raises(ValueError, match="truncated"):
        policy.apply(image, plan=AugmentationPlan.from_json(json.dumps(data)))


def test_grayscale_colour_failure_and_non_square_rotation():
    grayscale = np.ones((8, 8), np.uint8)
    with pytest.raises(ValueError, match="colour"):
        Augmenter({"noise": {"prob": 1, "jpg": True}})(grayscale, seed=1)
    rectangular = np.ones((8, 12), np.uint8)
    with pytest.raises(ValueError, match="square"):
        Augmenter({"rotate": True})(rectangular, seed=1)
    native = Augmenter({"rotate": True}, semantics="legacy_v1")(rectangular, seed=1)
    np.testing.assert_array_equal(native.image, rectangular)


def test_disabled_colour_operation_accepts_grayscale():
    policy = Augmenter({"noise": {"probabilities": {"jpg": 0, "inverse": 1}}})
    result = policy(np.full((8, 8), 10, np.uint8), seed=4)
    assert np.all(result.image == 245)


@pytest.mark.parametrize(
    "family,operation",
    [
        ("noise", name)
        for name in (
            "decolorize",
            "gauss_blur",
            "hist_eq",
            "clahe",
            "jpg",
            "erosion",
            "posterize",
            "inverse",
            "saltpepper",
            "convert_to_hsv",
            "convert_to_lab",
        )
    ]
    + [
        ("distort", name)
        for name in ("brightness", "contrast", "saturation", "hue", "channel_order")
    ],
)
def test_each_operator_replays_without_changing_labels(image, family, operation):
    policy = Augmenter({family: {"prob": 1, operation: True}})
    labels = (image[..., 0] > 80).astype(np.uint16) * 1000
    first = policy(image, seed=42, masks={"labels": labels})
    repeated = policy.apply(image, plan=first.plan, masks={"labels": labels})
    assert first.image.dtype == np.uint8
    np.testing.assert_array_equal(first.image, repeated.image)
    np.testing.assert_array_equal(first.masks["labels"], labels)
    assert len(first.plan.metadata["record"]["operations"]) == 1


def test_legacy_zero_crop_disables_cropping(image):
    result = Augmenter({"crop_size": 0}, semantics="legacy_v1")(image, seed=1)
    np.testing.assert_array_equal(result.image, image)
