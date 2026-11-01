import numpy as np


def _sequence(value):
    if isinstance(value, list):
        return np.asarray(value)
    if value["encoding"] == "arithmetic_sequence":
        return value["start"] + np.arange(value["count"]) * value["step"]
    if value["encoding"] == "symmetric_interleaved_arithmetic_sequence":
        magnitudes = value["magnitude_start"] + np.arange(value["pair_count"]) * value["magnitude_step"]
        return np.ravel(np.column_stack((-magnitudes, magnitudes))).tolist() + [0.0]
    raise ValueError(f"Unsupported angle encoding: {value['encoding']}")


def target_label(config, domain, algorithm):
    generation = config["generation"]
    entry = generation["training_targets"]["domain_targets"][domain][algorithm]
    source_domain = entry.get("source_domain", domain)
    name = config.get("algorithm_labels", {}).get(algorithm, algorithm.upper())
    if entry.get("kind") == "image_reference":
        return f"{name} ({source_domain.upper()})"
    angles = entry.get("angles")
    if angles is None:
        return f"{name} ({domain.upper()})"
    source = generation["input_iq"]
    if "indices" in angles:
        indices = _sequence(angles["indices"])
    elif angles["selection"] == "all":
        indices = np.arange(len(_sequence(source["all_steering_angles_rad"])))
    elif angles["selection"] == "input":
        indices = _sequence(source["selected_angle_indices"])
    else:
        raise ValueError("Target angles must declare indices or select input/all")
    if len(indices) == 1:
        angle = np.rad2deg(_sequence(source["all_steering_angles_rad"])[int(indices[0])])
        return f"{name} ({domain.upper()}, 1 angle, {angle:g}°)"
    return f"{name} ({domain.upper()}, {len(indices)} angles compounded)"
