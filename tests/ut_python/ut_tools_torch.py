import unittest
import os
import subprocess
import torch
import torchvision

_test_dir = os.path.dirname(__file__)
_temp_dir = os.path.join(_test_dir, "temp")

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

class TestTorchvisionExport(unittest.TestCase):

    def setUp(self):
        os.chdir(os.path.join(_test_dir, "../../tools/torch"))

    def test_resnet50_export(self):
        # Export model (not pretrained because we don't have permission for the cache)
        subprocess.run(["python3", "trace_torchvision.py", "-vp", "resnet50", "-o", _temp_dir])
        model_file = os.path.join(_temp_dir, "resnet50.pt")
        self.assertTrue(os.path.exists(model_file), model_file)

        # Export to onnx
        subprocess.run(["python3", "trace_torchvision.py", "-vp", "resnet50", "-o", _temp_dir, "--to-onnx", "--weights", model_file])
        onnx_file = os.path.join(_temp_dir, "resnet50.onnx")
        self.assertTrue(os.path.exists(onnx_file), onnx_file)

    def test_fasterrcnn_export(self):
        # Export model (not pretrained because we don't have permission for the cache)
        subprocess.run(["python3", "trace_torchvision.py", "-vp", "fasterrcnn_resnet50_fpn", "-o", _temp_dir])
        model_file = os.path.join(_temp_dir, "fasterrcnn_resnet50_fpn-cls91.pt")
        self.assertTrue(os.path.exists(model_file), model_file)

        # Test inference
        rfcnn = torch.jit.load(model_file)
        rfcnn.train()
        model_losses, model_preds = rfcnn(*get_detection_input())
        self.assertTrue("total_loss" in model_losses)
        self.assertTrue(model_losses["total_loss"] > 0)
        self.assertAlmostEqual(
                model_losses["total_loss"].item(),
                sum([model_losses[l].item() for l in model_losses if l != "total_loss"]),
                delta = 0.0001
            )

        rfcnn.eval()
        model_losses, model_preds = rfcnn(torch.rand(1, 3, 224, 224))
        self.assertTrue("boxes" in model_preds[0])

        # Export to onnx
        subprocess.run(["python3", "trace_torchvision.py", "-vp", "fasterrcnn_resnet50_fpn", "-o", _temp_dir, "--to-onnx", "--weights", model_file])
        onnx_file = os.path.join(_temp_dir, "fasterrcnn_resnet50_fpn-cls91.onnx")
        self.assertTrue(os.path.exists(onnx_file), onnx_file)

    def tearDown(self):
        print("Removing all files in %s" % _temp_dir)
        ignore=[".gitignore"]
        for f in os.listdir(_temp_dir):
            removed = os.path.join(_temp_dir, f)
            if f in ignore:
                print("Ignore %s" % removed)
            else:
                print("Remove %s" % removed)
                os.remove(removed)

if __name__ == '__main__':
    unittest.main()
