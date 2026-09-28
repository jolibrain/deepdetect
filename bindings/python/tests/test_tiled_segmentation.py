import io
from pathlib import Path

import numpy as np
from PIL import Image

from deepdetect.cli.events import EventWriter
from deepdetect.cli.tiled_segmentation import _preview_mask, run_tiled_image


class TileService:
    def __init__(self):
        self.calls = 0

    def predict(self, paths, **parameters):
        self.calls += 1
        predictions = []
        for path in paths:
            index = int(Path(path).stem.split("-")[-1])
            predictions.append(
                {
                    "uri": str(path),
                    "imgsize": {"width": 4, "height": 3},
                    "vals": [index % 2] * 12,
                    "confidences": {"best": [0.8] * 12},
                }
            )
        return {"predictions": list(reversed(predictions))}


def test_tiled_segmentation_stitches_and_writes_confidence_maps(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (6, 4), color="white").save(image)
    output = tmp_path / "output"
    service = TileService()
    writer = EventWriter(stream=io.StringIO())

    result = run_tiled_image(
        service=service,
        image_path=image,
        output_dir=output,
        predict_parameters={},
        tile_width=4,
        tile_height=3,
        overlap=1,
        batch_size=2,
        nclasses=2,
        preview_max_side=3,
        confidence_maps=True,
        writer=writer,
        warmup=1,
    )

    assert result.tiles == 4
    assert service.calls == 3
    with Image.open(output / "source_mask.png") as mask:
        assert mask.mode == "P"
        assert mask.size == (6, 4)
        np.testing.assert_array_equal(
            np.asarray(mask), np.tile([0, 0, 0, 1, 1, 1], (4, 1))
        )
    with Image.open(output / "source_overlay_preview.png") as preview:
        assert preview.size == (3, 2)
    with Image.open(output / "source_confidence.png") as confidence:
        assert confidence.size == (6, 4)
        assert np.unique(np.asarray(confidence)).tolist() == [52428]
    with Image.open(output / "source_foreground_probability.png") as foreground:
        assert np.asarray(foreground)[0].tolist() == [13107] * 3 + [52428] * 3
    assert [event["event"] for event in writer.events].count("tile_progress") == 2
    prediction = next(
        event for event in writer.events if event["event"] == "prediction"
    )
    assert prediction["class_histogram"] == {"0": 12, "1": 12}
    assert "vals" not in prediction


def test_binary_preview_keeps_a_single_foreground_pixel():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1, 1] = 1

    preview = _preview_mask(mask, (2, 2), nclasses=2)

    np.testing.assert_array_equal(np.asarray(preview), [[1, 0], [0, 0]])
