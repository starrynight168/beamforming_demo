from __future__ import annotations


def layer_name(index: int) -> str:
    return f"fc{int(index)}"


def bias_range_name(layer: str) -> str:
    return f"{layer}.bias"


def bias_dac_name(layer: str) -> str:
    return f"{layer}.bias_dac"


def input_dac_name(layer: str) -> str:
    return f"{layer}.input_dac"


def output_preactivation_name(layer: str) -> str:
    return f"{layer}.output.preactivation"


def output_controls_name(layer: str) -> str:
    return f"{layer}.output.controls"
