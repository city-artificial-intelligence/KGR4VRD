"""The two registered relation heads. Both return logits including the auxiliary column."""

from torch import nn


def build_model(name: str, input_dimension: int, predicate_count: int) -> nn.Module:
    if name == "geometry_mlp":
        widths = [
            input_dimension,
            2 * input_dimension,
            2 * input_dimension,
            2 * input_dimension,
            input_dimension,
            predicate_count + 1,
        ]
    elif name == "visual_mlp_v1":
        widths = [input_dimension, 512, 256, predicate_count + 1]
    else:
        raise ValueError(f"Unsupported model: {name}")
    layers: list[nn.Module] = []
    for index, (source, target) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(source, target))
        if index < len(widths) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)
