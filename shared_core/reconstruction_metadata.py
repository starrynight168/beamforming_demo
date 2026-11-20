"""Contract for the reconstruction metadata embedded in HDF5 files."""

from shared_core.data import embedded_config


RECONSTRUCTION_METADATA_VERSION = 3


def require_reconstruction_config(config):
    if not isinstance(config, dict) or config.get("schema_version") != RECONSTRUCTION_METADATA_VERSION:
        raise ValueError(
            f"H5 重建元数据必须使用 schema_version={RECONSTRUCTION_METADATA_VERSION}，请重新生成"
        )
    generation = config.get("generation")
    snapshot = generation.get("reconstruction_config") if isinstance(generation, dict) else None
    required = {"reconstruction", "algorithm_params", "outputs", "storage", "compression"}
    if not isinstance(snapshot, dict) or any(
        not isinstance(snapshot.get(key), dict) for key in required
    ):
        raise ValueError("H5 缺少完整的 generation.reconstruction_config")
    return snapshot


def target_reconstruction(config, signal_domain, algorithm=None):
    from algorithms.shared_core.common import INTERP_CHOICES

    require_reconstruction_config(config)
    generation = config["generation"]
    if algorithm is None:
        ground_truth = generation.get("ground_truth")
        algorithm = ground_truth.get("algorithm") if isinstance(ground_truth, dict) else None
    targets = generation.get("training_targets")
    domains = targets.get("domain_targets") if isinstance(targets, dict) else None
    entries = domains.get(signal_domain) if isinstance(domains, dict) else None
    entry = entries.get(algorithm) if isinstance(entries, dict) else None
    if (
        not isinstance(entry, dict)
        or not isinstance(entry.get("parameters"), dict)
        or not isinstance(entry.get("interpolation"), str)
        or entry["interpolation"] not in INTERP_CHOICES
    ):
        raise ValueError(f"H5 缺少 {signal_domain}/{algorithm} 的有效重建参数或插值")
    return entry


def read_target_reconstruction(hf, signal_domain, algorithm=None):
    return target_reconstruction(embedded_config(hf), signal_domain, algorithm)
