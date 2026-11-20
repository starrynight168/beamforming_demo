import h5py
import numpy as np
import yaml

from evaluation.context import picmus_reference_path


def write_picmus_sample(
    root, mode="contrast_speckle", source="simulation", has_gt=True, domain="iq"
):
    path = root / "sample.h5"
    phantom_path = (
        str(picmus_reference_path(mode, source))
        if source in {"simulation", "experiments"}
        else ""
    )
    with h5py.File(path, "w") as handle:
        handle["h5_schema_version"] = 3
        if domain == "iq":
            handle["iq_i"] = np.zeros((1, 1, 1, 1))
        elif domain == "rf":
            handle["rf"] = np.zeros((1, 1, 1, 1))
        elif domain == "analytic_rf":
            handle["ar_i"] = np.zeros((1, 1, 1, 1))
        else:
            raise ValueError(f"unsupported test domain: {domain}")
        handle["x_grid"] = np.linspace(-0.025, 0.025, 201)
        handle["z_grid"] = np.linspace(0.005, 0.055, 201)
        if has_gt:
            handle["mv_iq_db"] = np.zeros((1, 1, 201, 201))
        handle["config_yaml"] = yaml.safe_dump(
            {
                "generation": {
                    "storage": {
                        "inputs": {
                            domain: {
                                "dtype": "float32",
                                "normalization": "none",
                            }
                        }
                    },
                    "ground_truth": {
                        "algorithm": "mv",
                        "signal_domain": domain,
                        "dynamic_range_db": 60.0,
                    },
                    "training_targets": {
                        "domain_targets": {
                            domain: {"mv": {"db": "/mv_iq_db"}}
                        }
                    },
                },
                "source_samples": [
                    {
                        "id": "renamed_sample",
                        "phantom_mode": mode,
                        "phantom_source": source,
                        "phantom_path": phantom_path,
                    }
                ],
            }
        )
    return path


def update_sample_metadata(path, **changes):
    with h5py.File(path, "r+") as handle:
        config = yaml.safe_load(handle["config_yaml"][()])
        config["source_samples"][0].update(changes)
        del handle["config_yaml"]
        handle["config_yaml"] = yaml.safe_dump(config)
