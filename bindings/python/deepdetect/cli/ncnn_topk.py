"""Lower fixed-K FP32 PNNX TopK nodes to the DeepDetect CPU layer."""

import math
import re

from .ncnn_graph import Layer, read_layers, write_layers


def keep_topk_outputs(model):
    """PNNX 20261020 removes unused ONNX TopK outputs before schema checking."""
    from onnx import shape_inference

    nodes = [n for n in model.graph.node if n.op_type == "TopK" and not n.domain]
    if not nodes:
        return False
    graph = shape_inference.infer_shapes(model).graph
    info = {v.name: v for v in (*graph.value_info, *graph.output)}
    used = {s for n in graph.node for s in n.input} | {v.name for v in graph.output}
    changed = False
    for node in nodes:
        for name in node.output:
            if name not in used:
                if name not in info:
                    raise ValueError(f"missing type or shape for TopK output {name}")
                model.graph.output.append(info[name])
                changed = True
    return changed


def lower_topk(param, intermediate, image_shape, output_width):
    layers = read_layers(param)
    if not any(n.op in ("TopK", "torch.topk") for n in layers):
        return
    ir = read_layers(intermediate)
    by_name = {n.name: n for n in ir}
    producers = {s: n for n in ir for s in n.outputs}
    consumers = {s for n in layers for s in n.inputs}
    indices = set()
    for n in layers:
        if n.op not in ("TopK", "torch.topk"):
            continue
        ref = by_name[n.name]
        attrs = dict(x.split("=", 1) for x in ref.params)
        if n.op == "TopK":
            const = producers.get(ref.inputs[1])
            expr = next((p[5:] for p in const.params if p.startswith("expr=")), "") if const else ""
            if not re.fullmatch(r"[0-9]+", expr):
                raise ValueError("NCNN TopK requires a constant positive K")
            k = int(expr)
        else:
            k = int(attrs["k"])
        shape = re.fullmatch(r"\(([^)]*)\)f32", attrs.get("#" + ref.inputs[0], ""))
        if not shape:
            raise ValueError("NCNN TopK requires a known FP32 input rank")
        sizes = [int(x) for x in shape[1].split(",") if x]
        rank = len(sizes)
        axis = int(attrs.get("axis", attrs.get("dim", -1)))
        if not -rank <= axis < rank or k <= 0:
            raise ValueError("invalid TopK axis or K")
        # Trailing-axis numbering preserves a stripped leading image batch.
        axis = axis % rank - rank
        width, stride = sizes[axis], math.prod(sizes[axis + rank + 1 :])
        if width < k or stride < 1:
            raise ValueError("NCNN TopK requires a fixed axis and valid K")
        largest = attrs.get("largest", "1") in ("1", "True")
        outputs = [s for s in n.outputs if s in consumers or s == "out0"]
        if not outputs or len(n.outputs) != 2:
            raise ValueError("unexpected TopK output layout")
        mode = 2 if len(outputs) == 2 else n.outputs.index(outputs[0])
        if mode in (1, 2):
            indices.add(n.outputs[1])
        n.op, n.inputs, n.outputs = "DeepDetectTopK", n.inputs[:1], outputs
        n.params = [
            f"0={k}",
            f"1={axis}",
            f"2={int(largest)}",
            f"3={mode}",
            f"4={width}",
            f"5={stride}",
        ]

    for n in layers:
        if indices.isdisjoint(n.inputs):
            continue
        if n.op in ("Split", "Flatten", "Reshape", "Permute", "Squeeze", "ExpandDims"):
            indices.update(n.outputs)
        elif n.op == "Tensor.to" and "dtype=torch.float" in by_name[n.name].params:
            n.op, n.params = "DeepDetectIndicesToFloat", []
        else:
            raise ValueError(f"NCNN lowering for TopK int32 indices consumed by {n.op} is required")
    if "out0" in indices:
        raise ValueError("cast TopK indices to FP32 for the service output")

    # Drop only the now-unreferenced, weightless K expressions and their splits.
    live = {"out0"}
    kept = []
    for n in reversed(layers):
        if not live.intersection(n.outputs) and n.op in ("Split", "pnnx.Expression"):
            continue
        kept.append(n)
        live.update(n.inputs)
    layers = list(reversed(kept))
    if output_width is not None:
        # Some PNNX graphs retain the leading singleton batch as a physical
        # dimension. The NCNN image connector always feeds a three-dimensional Mat.
        wrapper = intermediate.with_name(
            intermediate.name.replace(".pnnx.param", "_ncnn.py")
        ).read_text()
        match = re.search(r'ex.input\("in0", ncnn.Mat\(in0.numpy\(\), batch_index=(\d+)\)', wrapper)
        if not match or match[1] not in ("0", "233"):
            raise ValueError("unrecognized PNNX image batch layout")
        if match[1] == "233":
            image = next(n for n in layers if n.op == "Input")
            image.outputs = ["dd_image"]
            _, c, h, w = image_shape
            layers.insert(
                layers.index(image) + 1,
                Layer(
                    "Reshape",
                    "dd_input_shape",
                    ["dd_image"],
                    ["in0"],
                    [f"0={w}", f"1={h}", f"11={c}", "2=1"],
                ),
            )
        result = next(n for n in layers if "out0" in n.outputs)
        result.outputs[result.outputs.index("out0")] = "dd_result"
        layers.append(
            Layer(
                "Reshape" if output_width else "Flatten",
                "dd_output_shape",
                ["dd_result"],
                ["out0"],
                [f"0={output_width}", "1=-1"] if output_width else [],
            )
        )
    write_layers(param, layers)
