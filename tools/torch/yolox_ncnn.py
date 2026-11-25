"""Inference-only ONNX provider for upstream YOLOX-S weights and worker checkpoints."""
import torch


class DecodedHead(torch.nn.Module):
    """Return [batch, anchors, 5 + foreground classes] in resized-image pixels."""

    def __init__(self, model, height, width):
        super().__init__()
        self.model = model.eval()
        self.model.head.decode_in_inference = False
        grids, strides = [], []
        for step in self.model.head.strides:
            ys, xs = torch.meshgrid(torch.arange(height // step),
                                    torch.arange(width // step), indexing="ij")
            grid = torch.stack((xs, ys), dim=-1).reshape(1, -1, 2).float()
            grids.append(grid)
            strides.append(torch.full_like(grid[:, :, :1], step))
        self.register_buffer("grid", torch.cat(grids, dim=1))
        self.register_buffer("stride", torch.cat(strides, dim=1))

    def forward(self, image):
        head = self.model(image)
        return torch.cat(((head[..., :2] + self.grid) * self.stride,
                          head[..., 2:4].exp() * self.stride, head[..., 4:]), dim=-1)


def yolox_s(checkpoint_paths, inference_configuration):
    """Use as mllib.onnx_export.provider=yolox_ncnn:yolox_s on the export host."""
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    if len(checkpoint_paths) != 1:
        raise ValueError("YOLOX-S requires one model checkpoint")
    options = inference_configuration["service"]["mllib"]
    nclasses = int(options["nclasses"]) - 1  # DeepDetect reserves class 0
    inputs = inference_configuration.get("parameters", {}).get("input", {})
    height = int(inputs.get("height", inference_configuration.get("height", 640)))
    width = int(inputs.get("width", inference_configuration.get("width", 640)))
    if min(height, width) <= 0 or height % 32 or width % 32:
        raise ValueError("YOLOX-S export requires dimensions divisible by 32")
    model = YOLOX(YOLOPAFPN(0.33, 0.5), YOLOXHead(nclasses, 0.5))
    # The upstream experiment config uses these inference-time BatchNorm settings.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03
    checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint.get("model", checkpoint)))
    model.load_state_dict(state, strict=True)
    # PNNX cannot lower Focus's four stepped slices. A fixed stride-two
    # convolution selects exactly the same pixels in upstream concatenation order.
    stem = model.backbone.backbone.stem
    select = torch.nn.Conv2d(3, 12, kernel_size=2, stride=2, bias=False)
    with torch.no_grad():
        select.weight.zero_()
        for phase, (y, x) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            for channel in range(3):
                select.weight[3 * phase + channel, channel, y, x] = 1
    model.backbone.backbone.stem = torch.nn.Sequential(select.requires_grad_(False), stem.conv)
    return (DecodedHead(model, height, width).eval(), (torch.zeros(1, 3, height, width),),
            {"input_names": ["image"], "output_names": ["decoded"]})
