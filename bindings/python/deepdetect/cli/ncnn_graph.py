"""Read and assemble PNNX text graphs while preserving layer and weight order."""
from dataclasses import dataclass


@dataclass
class Layer:
    op: str
    name: str
    inputs: list[str]
    outputs: list[str]
    params: list[str]

    @classmethod
    def parse(cls, line):
        op, name, ni, no, *fields = line.split()
        i, j = int(ni), int(ni) + int(no)
        return cls(op, name, fields[:i], fields[i:j], fields[j:])

    def line(self):
        header = [self.op, self.name, str(len(self.inputs)), str(len(self.outputs))]
        return " ".join([*header, *self.inputs, *self.outputs, *self.params])


def read_layers(path):
    magic, counts, *lines = path.read_text().splitlines()
    if magic.strip() != "7767517" or int(counts.split()[0]) != len(lines):
        raise ValueError(f"invalid PNNX/NCNN graph: {path.name}")
    return [Layer.parse(line) for line in lines]


def write_layers(path, layers):
    blobs = {s for layer in layers for s in layer.outputs}
    path.write_text(
        f"7767517\n{len(layers)} {len(blobs)}\n" + "".join(layer.line() + "\n" for layer in layers)
    )
