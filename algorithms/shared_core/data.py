"""Read explicitly described channel representations and linear teachers."""

import h5py
import numpy as np
import yaml
from scipy.signal import firwin

from algorithms.shared_core.signals import rf_to_analytic, rf_to_iq


def _require(hf, fields, message):
    missing = [field for field in fields if field not in hf]
    if missing:
        raise KeyError(f"{message}; missing H5 fields: {missing}")


def _metadata(hf):
    _require(hf, ("config_yaml",), "H5 requires config_yaml to describe its processing")
    config = yaml.safe_load(hf["config_yaml"][()])
    if not isinstance(config, dict) or not isinstance(config.get("generation"), dict):
        raise ValueError("config_yaml must describe generation as a mapping")
    return config


def _frame(hf, field, sample_idx):
    dataset = hf[field]
    if dataset.ndim != 4 or min(dataset.shape) == 0:
        raise ValueError(f"{field} must have nonempty [sample,angle,time,channel] dimensions")
    if not 0 <= sample_idx < dataset.shape[0]:
        raise IndexError(f"sample_idx={sample_idx} is outside {field} sample count {dataset.shape[0]}")
    values = dataset[sample_idx]
    if np.iscomplexobj(values):
        raise ValueError(f"{field} must store real components")
    return values.astype(np.float32)


def _sampling(hf, values, sample_idx, *, fs_key, t0_key, length_key, angles_key):
    _require(hf, (fs_key, t0_key, length_key, angles_key), "Channel sampling metadata is incomplete")
    fs = float(hf[fs_key][()])
    t0 = np.asarray(hf[t0_key][sample_idx], dtype=np.float64)
    angles = np.asarray(hf[angles_key][:], dtype=np.float64)
    raw_length = hf[length_key][sample_idx]
    length = int(raw_length)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"{fs_key} must be finite and positive")
    if t0.shape != (values.shape[0],) or angles.shape != t0.shape:
        raise ValueError(f"{t0_key}/{angles_key} must match the channel angle dimension")
    if not np.isfinite(t0).all() or not np.isfinite(angles).all():
        raise ValueError("Channel time origins and angles must be finite")
    if raw_length != length or not 2 <= length <= values.shape[1]:
        raise ValueError(f"{length_key} must be an integer within the saved time dimension")
    values = values[:, :length]
    if not np.isfinite(values).all():
        raise ValueError("Valid channel samples must be finite")
    return values, fs, t0, angles


def _scale(hf, sample_idx, required=False):
    if required and "all_scale_ref" not in hf:
        raise KeyError("Normalized inputs and teacher targets require H5 all_scale_ref")
    scale = float(hf["all_scale_ref"][sample_idx]) if "all_scale_ref" in hf else 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("all_scale_ref must be finite and positive")
    return scale


def input_is_normalized(hf, domain):
    inputs = _metadata(hf)["generation"].get("storage", {}).get("inputs")
    if not isinstance(inputs, dict):
        if domain == "analytic_rf":
            raise ValueError("Analytic RF cache requires explicit generation.storage.inputs.analytic_rf.normalize")
        return domain == "iq"
    if domain not in inputs or type(inputs[domain].get("normalize")) is not bool:
        raise ValueError(f"generation.storage.inputs.{domain}.normalize must be explicit")
    return inputs[domain]["normalize"]


def _rf_frame(hf, sample_idx):
    _require(hf, ("rf",), "RF input was not saved; select rf in storage.inputs")
    return _sampling(
        hf, _frame(hf, "rf", sample_idx), sample_idx,
        fs_key="rf_fs", t0_key="rf_t0", length_key="rf_length", angles_key="rf_angles",
    )


def _ddc_settings(hf, source_fs):
    config = _metadata(hf)
    settings = config["generation"].get("input_iq")
    if not isinstance(settings, dict):
        raise ValueError("generation.input_iq is required to derive IQ from RF")
    ddc = settings.get("ddc")
    if not isinstance(ddc, dict) or (
        ddc.get("phase_sign_q") != -1
        or ddc.get("mixing_gain", 1.0) != 1.0
        or ddc.get("lowpass") != "firwin_64_then_filtfilt"
        or ddc.get("resampling") != "scipy_resample_poly_kaiser_12"
    ):
        raise ValueError("Unsupported generation.input_iq.ddc definition")
    factor = settings.get("decimation_factor")
    if type(factor) is not int or factor < 1:
        raise ValueError("input_iq.decimation_factor must be a positive integer")
    fs = settings.get("source_sampling_frequency_hz")
    if fs is None or not np.isfinite(fs) or fs <= 0 or not np.isclose(fs, source_fs, rtol=1e-6):
        raise ValueError("input_iq source sampling frequency must match rf_fs")
    acquisition = config.get("acquisition")
    if not isinstance(acquisition, dict):
        raise ValueError("config_yaml.acquisition must describe the acquisition settings")
    fc = acquisition.get("fc_hz")
    if fc is None or not np.isfinite(fc) or not 0 < fc < fs / 2:
        raise ValueError("config_yaml.acquisition.fc_hz must describe the RF demodulation frequency")
    taps = ddc.get("num_taps", 64)
    cutoff = ddc.get("cutoff_hz", 0.8 * fc)
    if type(taps) is not int or taps < 2 or not np.isfinite(cutoff) or not 0 < cutoff < fs / 2:
        raise ValueError("DDC FIR length/cutoff must describe a valid lowpass filter")
    normalization = settings.get("normalization")
    if not isinstance(normalization, dict) or normalization.get("mode") not in ("complex_rms", "rms", "env_std"):
        raise ValueError("input_iq.normalization.mode must be complex_rms or env_std")
    epsilon = normalization.get("epsilon")
    if epsilon is None or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("input_iq.normalization.epsilon must be finite and positive")
    packed_fs = settings.get("packed_sampling_frequency_hz")
    if packed_fs is None or not np.isclose(packed_fs, fs / factor, rtol=1e-6):
        raise ValueError("input_iq packed sampling frequency does not match decimation_factor")
    return fs, fc, firwin(taps, cutoff / (fs / 2)), factor, normalization


def load_input(h5_path, sample_idx, domain="iq"):
    if not isinstance(sample_idx, (int, np.integer)) or isinstance(sample_idx, bool):
        raise ValueError("sample_idx must be an integer")
    if domain not in ("rf", "analytic_rf", "iq"):
        raise ValueError("domain must be rf, analytic_rf or iq")
    with h5py.File(h5_path, "r") as hf:
        cache_names = (("all_multi_I", "all_multi_Q") if domain == "iq" else
                       ("ar_i", "ar_q") if domain == "analytic_rf" else ())
        if any(field in hf for field in cache_names):
            _require(hf, cache_names, f"{domain.upper()} cache is incomplete; check storage.inputs")
            real = _frame(hf, cache_names[0], sample_idx)
            imag = _frame(hf, cache_names[1], sample_idx)
            if real.shape != imag.shape:
                raise ValueError("Cached complex components must have matching dimensions")
            keys = (dict(fs_key="fs", t0_key="time_start_vector", length_key="valid_time_samples", angles_key="angles")
                    if domain == "iq" else dict(fs_key="ar_fs", t0_key="ar_t0", length_key="ar_len", angles_key="ar_angles"))
            real, fs, t0, angles = _sampling(hf, real, sample_idx, **keys)
            imag = imag[:, :real.shape[1]]
            if not np.isfinite(imag).all():
                raise ValueError("Valid complex samples must be finite")
            normalized = input_is_normalized(hf, domain)
            scale = _scale(hf, sample_idx, required=domain == "iq" or normalized)
            if not normalized:
                real, imag = real / scale, imag / scale
        else:
            real, fs, t0, angles = _rf_frame(hf, sample_idx)
            normalized_rf = input_is_normalized(hf, "rf")
            scale = _scale(hf, sample_idx, required=normalized_rf)
            if normalized_rf:
                real *= scale
            if domain == "iq":
                fs, fc, coefficients, factor, normalization = _ddc_settings(hf, fs)
                real, imag = rf_to_iq(real, fs, fc, t0, coefficients, factor)
                fs /= factor
                if "all_scale_ref" not in hf:
                    epsilon = normalization["epsilon"]
                    if normalization["mode"] in ("complex_rms", "rms"):
                        scale = float(np.sqrt(np.mean(real**2 + imag**2) + epsilon))
                    else:
                        scale = float(np.std(np.hypot(real, imag)) + epsilon)
                    if not np.isfinite(scale) or scale <= 0:
                        raise ValueError("Derived IQ normalization scale must be finite and positive")
            elif domain == "analytic_rf":
                real, imag = rf_to_analytic(real)
            else:
                imag = np.zeros_like(real)
            real, imag = real / scale, imag / scale
    return {"domain": domain, "real": real, "imag": imag, "fs": fs,
            "t0": t0, "angles": angles, "scale": scale}


def load_target(h5_path, sample_idx, algorithm="mv", domain="iq", angles="input"):
    if not isinstance(sample_idx, (int, np.integer)) or isinstance(sample_idx, bool):
        raise ValueError("sample_idx must be an integer")
    fields = {"iq": "complex_iq_targets", "rf": "rf_targets",
              "analytic_rf": "analytic_rf_targets"}
    if domain not in fields:
        raise ValueError("domain must be rf, analytic_rf or iq")
    with h5py.File(h5_path, "r") as hf:
        targets = _metadata(hf)["generation"].get("training_targets", {})
        if not isinstance(targets, dict):
            raise ValueError("generation.training_targets must be a mapping")
        map_name = fields[domain]
        option = f"{domain}.{algorithm}"
        mapping = targets.get(map_name, {})
        if not isinstance(mapping, dict):
            raise ValueError(f"training_targets.{map_name} must be a mapping")
        if algorithm not in mapping:
            raise KeyError(f"No {algorithm}/{domain} teacher; select signal in outputs.{option}")
        angle_spec = targets.get("output_angles", {}).get(algorithm)
        if not isinstance(angle_spec, dict) or "selection" not in angle_spec:
            raise ValueError(f"Teacher {algorithm} requires explicit output_angles metadata")
        if angle_spec["selection"] != angles:
            raise ValueError(f"Teacher {algorithm} uses angles={angle_spec['selection']!r}, requested {angles!r}")
        entry = mapping[algorithm]
        if domain == "iq":
            paths = (entry["i"], entry["q"])
        elif domain == "analytic_rf":
            paths = (entry["real"], entry["imag"])
        else:
            paths = (entry,)
        _require(hf, (*paths, "x_grid", "z_grid"), f"Teacher fields are incomplete; check outputs.{option}")
        grid = {name: np.asarray(hf[name][:], dtype=np.float32) for name in ("x_grid", "z_grid")}
        for name, values in grid.items():
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or not np.all(np.diff(values) > 0):
                raise ValueError(f"{name} must contain finite increasing coordinates")
        expected_shape = (grid["z_grid"].size, grid["x_grid"].size)
        components = []
        for path in paths:
            dataset = hf[path]
            if dataset.ndim != 4 or dataset.shape[1:] != (1, *expected_shape):
                raise ValueError(f"Teacher {path} must have [N,1,H,W] dimensions matching its grid")
            if not 0 <= sample_idx < dataset.shape[0]:
                raise IndexError(f"sample_idx={sample_idx} is outside teacher {path} sample count")
            value = dataset[sample_idx, 0]
            if np.iscomplexobj(value) or not np.isfinite(value).all():
                raise ValueError(f"Teacher {path} must contain finite real components")
            components.append(value.astype(np.float32))
        real = components[0]
        imag = components[1] if len(components) == 2 else np.zeros_like(real)
        scale = _scale(hf, sample_idx, required=True)
    return {"algorithm": algorithm, "domain": domain, "real": real, "imag": imag,
            "grid": grid, "scale": scale, "angles": angles}
