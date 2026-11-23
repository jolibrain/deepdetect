"""Assemble a single NCNN graph for the fixed-size RT-DETRv4-S deployment head."""

import shutil

from .ncnn_graph import Layer, read_layers, write_layers


def convert_rtdetrv4(graph, binary, shape, classes, stage, convert_part):
    import onnx
    from onnx import TensorProto, helper, numpy_helper, shape_inference
    from onnx.utils import Extractor

    if graph.suffix != ".onnx" or shape != [1, 3, 640, 640] or classes < 2:
        raise ValueError("RT-DETRv4-S requires batch-one 640x640 RGB ONNX and foreground classes")
    model = shape_inference.infer_shapes(onnx.load(graph))
    nodes = {n.name: n for n in model.graph.node}
    dec = "/network/decoder/"
    inner = dec + "decoder/"
    foreground = classes - 1
    # These boundaries are part of the deployment provider's ONNX contract.
    try:
        memory = nodes[dec + "Mul"].input[1]
        features = nodes[dec + "enc_output/norm/LayerNormalization"].output[0]
        scores = nodes[dec + "TopK"].input[0]
        anchor_gather = nodes[dec + "GatherElements"]
        selected_anchors = anchor_gather.output[0]
        selected_features = nodes[dec + "GatherElements_1"].output[0]
        lqe = nodes[inner + "lqe_layers.2/TopK"]
        logits = nodes[inner + "dec_score_head.2/Add"].output[0]
        boxes = nodes["/post/GatherElements"].input[0]
        final = nodes["/post/TopK"]
        anchors = next(t for t in model.graph.initializer if t.name == anchor_gather.input[0])
        class_bias = next(
            t
            for t in model.graph.initializer
            if t.name == nodes[inner + "dec_score_head.2/Add"].input[0]
        )
    except (KeyError, StopIteration, IndexError) as exc:
        raise ValueError("expected the RT-DETRv4-S deployment provider graph") from exc
    if list(anchors.dims) != [1, 8400, 4] or list(class_bias.dims) != [foreground]:
        raise ValueError("RT-DETRv4 anchor or foreground class count mismatch")
    constants = {
        n.output[0]: numpy_helper.to_array(n.attribute[0].t)
        for n in model.graph.node
        if n.op_type == "Constant" and n.attribute[0].name == "value"
    }
    # The last join replaces the provider's one-based query/class decoding.
    # Reject another wrapper even if it happens to expose six columns.
    try:
        join = nodes["/Concat"]
        score_column = nodes["/Unsqueeze_1"]
        label_column = nodes["/Unsqueeze"]
        label_offset = nodes["/Add"]
        if (
            join.op_type != "Concat"
            or list(join.output) != [model.graph.output[0].name]
            or list(join.input)
            != [
                label_column.output[0],
                score_column.output[0],
                nodes["/post/GatherElements"].output[0],
            ]
            or score_column.input[0] != final.output[0]
            or label_column.input[0] != nodes["/Cast"].output[0]
            or nodes["/Cast"].input[0] != label_offset.output[0]
            or label_offset.input[0] != nodes["/post/Sub_2"].output[0]
            or int(constants[label_offset.input[1]].item()) != 1
            or int(constants[nodes["/post/Div"].input[1]].item()) != foreground
        ):
            raise ValueError("unexpected RT-DETRv4 detection row or class numbering")
    except (KeyError, IndexError) as exc:
        raise ValueError("use the one-based RT-DETRv4 deployment provider") from exc
    for node, k in [(nodes[dec + "TopK"], 300), (lqe, 4), (final, 300)]:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (
            node.op_type != "TopK"
            or list(constants.get(node.input[1], [])) != [k]
            or attrs.get("axis", -1) != -1
            or not attrs.get("largest", 1)
        ):
            raise ValueError("unexpected RT-DETRv4 ranking contract")
    shapes = {
        features: [1, 8400, 256],
        scores: [1, 8400],
        lqe.input[0]: [1, 300, 4, 33],
        final.input[0]: [1, 300 * foreground],
        memory: [1, 8400, 256],
        selected_anchors: [1, 300, 4],
        selected_features: [1, 300, 256],
        logits: [1, 300, foreground],
        lqe.output[0]: [1, 300, 4, 4],
        boxes: [1, 300, 4],
    }
    for v in model.graph.value_info:
        if v.name in shapes:
            v.CopyFrom(helper.make_tensor_value_info(v.name, TensorProto.FLOAT, shapes[v.name]))
    # The fixed encoder emits three feature maps at strides 8, 16 and 32.
    # Freeze only their shape queries crossing the encoder/decoder boundary.
    encoder_shapes = [
        n
        for n in model.graph.node
        if n.op_type == "Shape"
        and n.name.startswith(dec)
        and n.input[0].startswith("/network/encoder/")
    ]
    maps = list(dict.fromkeys(n.input[0] for n in encoder_shapes))
    if len(maps) != 3:
        raise ValueError("expected the three RT-DETRv4-S feature levels")
    for n in encoder_shapes:
        side = (80, 40, 20)[maps.index(n.input[0])]
        if n.attribute:
            raise ValueError("unexpected sliced feature shape")
        value = helper.make_tensor("shape", TensorProto.INT64, [4], [1, 256, side, side])
        n.CopyFrom(helper.make_node("Constant", [], list(n.output), name=n.name, value=value))
    # The LQE MLP receives four values plus their mean for each box side.
    lqe_shape = "dd_lqe_statistics_shape"
    model.graph.initializer.append(
        helper.make_tensor(lqe_shape, TensorProto.INT64, [3], [1, 300, 20])
    )
    nodes[inner + "lqe_layers.2/Reshape_1"].input[1] = lqe_shape
    extractor = Extractor(model)
    layers = []
    weights = stage / "model.ncnn.bin"

    def part(name, inputs, outputs, input_shapes, native_inputs, native_outputs):
        work = graph.parent / name
        work.mkdir()
        path = work / "part.onnx"
        section_model = extractor.extract_model(inputs, outputs)
        onnx.checker.check_model(section_model)
        onnx.save(section_model, path)
        convert_part(path, binary, input_shapes, work)
        section = read_layers(work / "model.ncnn.param")
        if name == "core":
            # PNNX loses dim=-1 in the weighted sum over 12 sample points.
            # Verify the source operation and shape before repairing the width axis.
            sums = [n for n in read_layers(work / "part.pnnx.param") if n.op == "torch.sum"]
            if len(sums) != 3 or any(
                "dim=-1" not in n.params or f"#{n.inputs[0]}=(8,32,300,12)f32" not in n.params
                for n in sums
            ):
                raise ValueError("unexpected RT-DETRv4 attention reductions")
            reductions = [n for n in section if n.op == "Reduction" and "0=0" in n.params]
            if len(reductions) != 3:
                raise ValueError("expected three NCNN attention sums")
            for n in reductions:
                if "-23303=0" in n.params:
                    n.params[n.params.index("-23303=0")] = "-23303=1,2"
                elif "-23303=1,2" not in n.params:
                    raise ValueError("unexpected NCNN attention sum axis")
        names = {f"in{i}": s for i, s in enumerate(native_inputs)}
        names.update({f"out{i}": s for i, s in enumerate(native_outputs)})
        for n in section:
            if n.op == "Input":
                continue
            n.name = name + "_" + n.name
            n.inputs = [names.get(s, name + "_" + s) for s in n.inputs]
            n.outputs = [names.get(s, name + "_" + s) for s in n.outputs]
            layers.append(n)
        with (work / "model.ncnn.bin").open("rb") as src, weights.open("ab") as dest:
            shutil.copyfileobj(src, dest)

    def add(op, name, inputs, outputs, *params):
        layers.append(Layer(op, name, inputs.split(), outputs.split(), list(params)))

    add("Input", "image_input", "", "image")
    part("encoder", [model.graph.input[0].name], [memory], [shape], ["image"], ["memory"])
    add("Split", "reuse_memory", "memory", "proposal_input core_input")
    part(
        "proposals",
        [memory],
        [scores, features],
        [[1, 8400, 256]],
        ["proposal_input"],
        ["proposal_scores", "features"],
    )
    add("DeepDetectTopK", "proposal_topk", "proposal_scores", "selected", "0=300", "3=1")
    add("Split", "reuse_indices", "selected", "anchor_ids feature_ids")
    add("MemoryData", "anchors", "", "anchors", "0=4", "1=8400")
    with weights.open("ab") as stream:
        stream.write(numpy_helper.to_array(anchors).astype("<f4").tobytes())
    add("DeepDetectGatherRows", "gather_anchors", "anchors anchor_ids", "selected_anchors")
    add("DeepDetectGatherRows", "gather_features", "features feature_ids", "selected_features")
    add("Reshape", "memory_batch", "core_input", "core_memory", "0=256", "1=8400", "2=1")
    part(
        "core",
        [memory, selected_anchors, selected_features],
        [lqe.input[0], logits, boxes],
        [[1, 8400, 256], [1, 300, 4], [1, 300, 256]],
        ["core_memory", "selected_anchors", "selected_features"],
        ["corner_probs", "class_logits", "corners"],
    )
    add("DeepDetectTopK", "quality_topk", "corner_probs", "best", "0=4", "3=0")
    add("Reshape", "quality_batch", "best", "quality_input", "0=4", "1=4", "11=300", "2=1")
    add(
        "Reshape",
        "logits_batch",
        "class_logits",
        "quality_logits",
        f"0={foreground}",
        "1=300",
        "2=1",
    )
    part(
        "quality",
        [lqe.output[0], logits],
        [final.input[0]],
        [[1, 300, 4, 4], [1, 300, foreground]],
        ["quality_input", "quality_logits"],
        ["confidences"],
    )
    add("Flatten", "flatten_scores", "confidences", "all_scores")
    add("DeepDetectTopK", "final_topk", "all_scores", "scores ids", "0=300", "3=2")
    add("Reshape", "box_rows", "corners", "boxes", "0=4", "1=300")
    add("DeepDetectDetectionRows", "detections", "scores ids boxes", "rows", f"0={foreground}")
    write_layers(stage / "model.ncnn.param", layers)
