"""
Test sample to ease YOLOX debug.
Requirements:
    - jolibrain YOLOX repo & all dependencies, on branch "jit_script".
    - path of YOLOX repo in env variable "YOLOX_PATH"
    - dd build with TensorRT & tests in build/

Usage:
    YOLOX_PATH=... python3 ut_tools_torch_yolox.py
"""

import unittest
import os
import subprocess
import torch
import torchvision
import requests
import shutil

_test_dir = os.path.dirname(os.path.abspath(__file__))
_temp_dir = os.path.join(_test_dir, "temp")

yolox_path = os.environ["YOLOX_PATH"]
yolox_weights = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth"
batch_size = 2
num_classes = 81
img_size = 640


def get_detection_input(batch_size=1):
    """
    Sample input for detection models, usable for tracing or testing
    """
    return (
        torch.rand(batch_size, 3, 224, 224),
        torch.full((batch_size,), 0).long(),
        torch.Tensor([1, 1, 200, 200]).repeat((batch_size, 1)),
        torch.full((batch_size,), 1).long(),
    )


class TestYOLXExport(unittest.TestCase):
    def setUp(self):
        os.chdir(os.path.join(_test_dir, "../../tools/torch"))

    def test_yolox_export(self):
        # Download weights
        weights_fname = os.path.join(_test_dir, "yolox-m.pth")

        if not os.path.exists(weights_fname):
            print("Download weights...")
            r = requests.get(yolox_weights, allow_redirects=True)
            with open(weights_fname, "wb") as weights_file:
                weights_file.write(r.content)
            print("Done!")
        else:
            print("Reusing weights at " + weights_fname)

        # Export model
        subprocess.run(
            [
                "python3",
                "trace_yolox.py",
                "-v",
                "yolox-m",
                "--yolox_path",
                yolox_path,
                "-o",
                _temp_dir,
                "--weights",
                weights_fname,
                "--num_classes",
                str(num_classes),
                "--img_width",
                str(img_size),
                "--img_height",
                str(img_size),
            ]
        )
        model_file = os.path.join(_temp_dir, "yolox-m_cls%d.pt" % num_classes)
        self.assertTrue(os.path.exists(model_file), model_file)

        # Load and check
        # TODO

        # Export to onnx
        subprocess.run(
            [
                "python3",
                "trace_yolox.py",
                "-v",
                "yolox-m",
                "--yolox_path",
                yolox_path,
                "-o",
                _temp_dir,
                "--to_onnx",
                "--weights",
                model_file,
                "--num_classes",
                str(num_classes),
                "--batch_size",
                str(batch_size),
                "--img_width",
                str(img_size),
                "--img_height",
                str(img_size),
            ]
        )
        onnx_file = os.path.join(_temp_dir, "yolox-m.onnx")
        self.assertTrue(os.path.exists(onnx_file), onnx_file)

        # Test it against trt using the unit test
        os.chdir(os.path.join(_test_dir, "../../build/tests/"))
        print("Moved to " + os.getcwd())
        os.rename(
            "../examples/trt/yolox_onnx_trt_nowrap/yolox-s.onnx",
            "../examples/trt/yolox_onnx_trt_nowrap/yolox-s.temp",
        )
        shutil.copy(onnx_file, "../examples/trt/yolox_onnx_trt_nowrap/yolox-m.onnx")
        subprocess.run(
            ["./ut_tensorrtapi", "--gtest_filter=tensorrtapi.service_predict_bbox_onnx"]
        )

    def tearDown(self):
        print("Removing all files in %s" % _temp_dir)
        ignore = [".gitignore"]
        for f in os.listdir(_temp_dir):
            removed = os.path.join(_temp_dir, f)
            if f in ignore:
                print("Ignore %s" % removed)
            else:
                print("Remove %s" % removed)
                os.remove(removed)

        # Clean trt test dir
        os.chdir(os.path.join(_test_dir, "../../build/tests/"))
        try:
            os.rename(
                "../examples/trt/yolox_onnx_trt_nowrap/yolox-s.temp",
                "../examples/trt/yolox_onnx_trt_nowrap/yolox-s.onnx",
            )
            os.remove("../examples/trt/yolox_onnx_trt_nowrap/yolox-m.onnx")
        except:
            print("Something went wrong during trt yolox cleanup")


if __name__ == "__main__":
    unittest.main()
