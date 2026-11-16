import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.shared_core import common
from tools.shared.run_output import ToolOutput


SCENE_IDS = (
    "simulation_contrast_speckle",
    "simulation_resolution_distortion",
    "experiments_contrast_speckle",
    "experiments_resolution_distortion",
)
METHODS = ("max", "rms", "std", "complex_whitening")


def load_scenes(config_path):
    path = config_path.expanduser()
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    scenes = config.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("config.yaml must contain a scenes list")
    by_id = {str(scene.get("id")): scene for scene in scenes if isinstance(scene, dict)}
    missing = [scene_id for scene_id in SCENE_IDS if scene_id not in by_id]
    if missing:
        raise ValueError(f"config.yaml is missing standard scenes: {missing}")
    selected = []
    for scene_id in SCENE_IDS:
        scene = by_id[scene_id]
        data_path = Path(scene["h5_path"]).expanduser()
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        data_path = data_path.resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"{scene_id}: H5 file not found: {data_path}")
        selected.append({
            "id": scene_id,
            "h5_path": data_path,
            "sample_idx": int(scene.get("sample_idx", 0)),
        })
    return selected


def _sample_vectors(sample, pixels, seed, f_number, device):
    x = torch.as_tensor(sample.x_grid, dtype=torch.float32, device=device)
    z = torch.as_tensor(sample.z_grid, dtype=torch.float32, device=device)
    total = x.numel() * z.numel()
    if pixels > total:
        raise ValueError(f"pixels={pixels} exceeds image pixels={total}")
    selected = np.sort(np.random.default_rng(seed).choice(total, pixels, replace=False))
    flat = torch.as_tensor(selected, dtype=torch.long, device=device)
    x_idx = flat % x.numel()
    z_idx = flat // x.numel()
    xp, depth = x[x_idx], z[z_idx]
    channels = sample.n_elem
    channel_pos = (torch.arange(channels, device=device, dtype=torch.float32) - (channels - 1) / 2) * sample.pitch
    receive = torch.sqrt(depth[:, None].square() + (xp[:, None] - channel_pos[None, :]).square())
    center = torch.round(xp / sample.pitch + (channels - 1) / 2).long().clamp(0, channels - 1)
    aperture = torch.floor(depth / (f_number * sample.pitch)).long() + 1
    if bool(((aperture < 1) | (aperture > channels)).any()):
        raise ValueError("Dynamic aperture exceeds the physical channel count; adjust f-number")
    start = (center - aperture // 2).clamp(min=0)
    start = torch.minimum(start, channels - aperture)
    channel_idx = torch.arange(channels, device=device)[None, :]
    active = (channel_idx >= start[:, None]) & (channel_idx < (start + aperture)[:, None])
    result_i = torch.zeros((sample.angles.size, pixels, channels), dtype=torch.float32, device=device)
    result_q = torch.zeros_like(result_i)
    with torch.no_grad():
        for angle_idx, angle in enumerate(sample.angles):
            tx_dist = depth * float(np.cos(angle)) + xp * float(np.sin(angle))
            tof = (tx_dist[:, None] + receive) / sample.c
            positions = (tof - float(sample.t0_vec[angle_idx])) * sample.fs
            valid = (positions >= 0) & (positions <= sample.i_data.shape[1] - 1)
            positions = positions.clamp(0, sample.i_data.shape[1] - 1)
            idx0 = positions.floor().long().clamp(0, sample.i_data.shape[1] - 2)
            frac = (positions - idx0.float()).clamp(0, 1)
            idx1 = idx0 + 1
            i_frame = torch.as_tensor(sample.i_data[angle_idx], dtype=torch.float32, device=device)
            q_frame = torch.as_tensor(sample.q_data[angle_idx], dtype=torch.float32, device=device)
            i_val = i_frame[idx0, channel_idx] * (1 - frac) + i_frame[idx1, channel_idx] * frac
            q_val = q_frame[idx0, channel_idx] * (1 - frac) + q_frame[idx1, channel_idx] * frac
            if sample.signal_domain == "iq":
                phase = 2 * torch.pi * sample.fc * (tof - tx_dist[:, None] / sample.c)
                cos_p, sin_p = torch.cos(phase), torch.sin(phase)
                i_val, q_val = i_val * cos_p - q_val * sin_p, i_val * sin_p + q_val * cos_p
            result_i[angle_idx] = i_val * valid
            result_q[angle_idx] = q_val * valid
    return result_i, result_q, active


def _normalize(i_data, q_data, mask, method, epsilon):
    weights = mask.to(i_data.dtype).unsqueeze(0).expand_as(i_data)
    i_data, q_data = i_data * weights, q_data * weights
    count = weights.sum(-1, keepdim=True).clamp_min(1)
    regularized = torch.zeros_like(count, dtype=torch.bool)
    if method == "max":
        scale = torch.hypot(i_data, q_data).amax(-1, keepdim=True).clamp_min(1e-8)
        ni, nq = i_data / scale, q_data / scale
        restore = lambda a, b: (a * scale * weights, b * scale * weights)
    elif method in ("rms", "std"):
        if method == "rms":
            mean = torch.zeros_like(count)
            scale = (((i_data.square() + q_data.square()) * weights).sum(-1, keepdim=True) / count).sqrt().clamp_min(1e-8)
            ni, nq = i_data / scale, q_data / scale
        else:
            mean = ((i_data + q_data) * weights).sum(-1, keepdim=True) / (2 * count)
            variance = (((i_data - mean).square() + (q_data - mean).square()) * weights).sum(-1, keepdim=True) / (2 * count)
            scale = variance.sqrt().clamp_min(1e-8)
            ni, nq = (i_data - mean) * weights / scale, (q_data - mean) * weights / scale
        restore = lambda a, b: ((a * scale + mean) * weights, (b * scale + mean) * weights)
    elif method == "complex_whitening":
        mean_i = i_data.sum(-1, keepdim=True) / count
        mean_q = q_data.sum(-1, keepdim=True) / count
        di, dq = (i_data - mean_i) * weights, (q_data - mean_q) * weights
        cii = di.square().sum(-1) / count.squeeze(-1)
        cqq = dq.square().sum(-1) / count.squeeze(-1)
        ciq = (di * dq).sum(-1) / count.squeeze(-1)
        covariance = torch.stack((cii, ciq, ciq, cqq), dim=-1).reshape(*cii.shape, 2, 2)
        eigenvalues, vectors = torch.linalg.eigh(covariance)
        floor = (eigenvalues[..., -1:] * epsilon).clamp_min(1e-12)
        regularized = eigenvalues[..., :1] < floor
        eigenvalues = torch.maximum(eigenvalues, floor)
        inverse = (vectors * eigenvalues.rsqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
        forward = (vectors * eigenvalues.sqrt().unsqueeze(-2)) @ vectors.transpose(-1, -2)
        ni = (inverse[..., 0, 0, None] * di + inverse[..., 0, 1, None] * dq) * weights
        nq = (inverse[..., 1, 0, None] * di + inverse[..., 1, 1, None] * dq) * weights
        restore = lambda a, b: (
            (forward[..., 0, 0, None] * a + forward[..., 0, 1, None] * b + mean_i) * weights,
            (forward[..., 1, 0, None] * a + forward[..., 1, 1, None] * b + mean_q) * weights,
        )
    else:
        raise ValueError(f"Unknown normalization: {method}")
    active = mask.unsqueeze(0).expand_as(i_data)
    n_count = active.sum(-1, keepdim=True).clamp_min(1)
    w = active.to(i_data.dtype)
    mean_i = (ni * w).sum(-1, keepdim=True) / n_count
    mean_q = (nq * w).sum(-1, keepdim=True) / n_count
    di, dq = (ni - mean_i) * w, (nq - mean_q) * w
    denom = (di.square().sum(-1) * dq.square().sum(-1)).sqrt()
    valid_corr = denom > 1e-20
    corr = ((di * dq).sum(-1) / denom.clamp_min(1e-20)).abs()
    ri, rq = restore(ni, nq)
    fi, fq = ni.to(torch.float16).float(), nq.to(torch.float16).float()
    finite = bool(torch.isfinite(fi).all() and torch.isfinite(fq).all())
    energy = float((i_data.double().square() + q_data.double().square()).sum().item())
    err32 = float(((ri - i_data).double().square() + (rq - q_data).double().square()).sum().item())
    err16 = None
    if finite:
        hi, hq = restore(fi, fq)
        err16 = float(((hi - i_data).double().square() + (hq - q_data).double().square()).sum().item())
    values = torch.cat((ni[active], nq[active])).abs()
    underflow = int((((ni != 0) & (fi == 0)) | ((nq != 0) & (fq == 0)))[active].sum().item())
    return {
        "method": method,
        "vectors": int(mask.shape[0] * i_data.shape[0]),
        "components": int(values.numel()),
        "input_energy": energy,
        "roundtrip_f32_error_energy": err32,
        "roundtrip_f16_error_energy": err16,
        "f16_finite": finite,
        "f16_underflow_pairs": underflow,
        "normalized_component_max": float(values.max().item()),
        "normalized_component_p99": float(torch.quantile(values, 0.99).item()),
        "fraction_abs_gt_1": float((values > 1).float().mean().item()),
        "fraction_abs_gt_2": float((values > 2).float().mean().item()),
        "mean_abs_iq_correlation": float(corr[valid_corr].mean().item()) if valid_corr.any() else None,
        "regularized_vectors": int(regularized.sum().item()),
    }


def summarize(records):
    groups = [(scene_id, [row for row in records if row["scene_id"] == scene_id]) for scene_id in SCENE_IDS]
    groups.append(("all_scenes", records))
    summary = []
    for scene_id, group in groups:
        for method in METHODS:
            rows = [row for row in group if row["method"] == method]
            energy = sum(row["input_energy"] for row in rows)
            components = sum(row["components"] for row in rows)
            if not rows or energy <= 0 or components <= 0:
                raise ValueError(f"Selected scene group {scene_id} contains no signal energy")
            weighted = lambda key: sum(row[key] * row["components"] for row in rows) / components
            correlations = [row["mean_abs_iq_correlation"] for row in rows if row["mean_abs_iq_correlation"] is not None]
            finite = all(row["f16_finite"] for row in rows)
            summary.append({
                "scene_id": scene_id,
                "method": method,
                "frames": len(rows),
                "vectors": sum(row["vectors"] for row in rows),
                "components": components,
                "roundtrip_f32_nrmse_percent": 100 * np.sqrt(sum(row["roundtrip_f32_error_energy"] for row in rows) / energy),
                "roundtrip_f16_nrmse_percent": 100 * np.sqrt(sum(row["roundtrip_f16_error_energy"] for row in rows) / energy) if finite else None,
                "f16_finite": finite,
                "f16_underflow_pairs": sum(row["f16_underflow_pairs"] for row in rows),
                "normalized_component_max": max(row["normalized_component_max"] for row in rows),
                "mean_frame_p99_abs_component": float(np.mean([row["normalized_component_p99"] for row in rows])),
                "fraction_abs_gt_1": weighted("fraction_abs_gt_1"),
                "fraction_abs_gt_2": weighted("fraction_abs_gt_2"),
                "mean_frame_abs_iq_correlation": float(np.mean(correlations)) if correlations else None,
                "regularized_vectors": sum(row["regularized_vectors"] for row in rows),
            })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare complex input normalizations on the four standard scenes.")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--signal-domain", choices=("iq", "analytic_rf", "rf"), default="iq")
    parser.add_argument("--pixels", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--f-number", type=float, default=1.5)
    parser.add_argument("--whitening-epsilon", type=float, default=1e-6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.pixels < 1 or not np.isfinite(args.f_number) or args.f_number <= 0 or not 0 < args.whitening_epsilon < 1:
        parser.error("pixels and f-number must be positive; whitening-epsilon must be in (0,1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    scenes = load_scenes(args.config)
    records = []
    for scene_idx, scene in enumerate(scenes):
        sample = common.load_from_h5(scene["h5_path"], scene["sample_idx"], signal_domain=args.signal_domain)
        i_data, q_data, mask = _sample_vectors(sample, args.pixels, args.seed + scene_idx, args.f_number, device)
        for method in METHODS:
            row = _normalize(i_data, q_data, mask, method, args.whitening_epsilon)
            row.update({"scene_id": scene["id"], "file": str(scene["h5_path"]), "sample_index": scene["sample_idx"]})
            records.append(row)
        print(f"{scene['id']}: done", flush=True)
    summary = summarize(records)
    output = ToolOutput("compare_input_normalization", args.output_dir)
    output.config({
        "tool": "compare_input_normalization",
        "scope": "four standard scenes; aligned dynamic-aperture samples; no trained model or image-quality measurement",
        "signal_domain": args.signal_domain,
        "pixels_per_scene": args.pixels,
        "seed": args.seed,
        "f_number": args.f_number,
        "whitening_epsilon": args.whitening_epsilon,
        "device": str(device),
        "methods": METHODS,
        "definitions": {
            "max": "divide each pixel aperture by its maximum complex-channel magnitude",
            "rms": "divide I/Q by the active-channel complex RMS",
            "std": "jointly center and scale the active I/Q components",
            "complex_whitening": "whiten the centered 2x2 I/Q covariance with an eigenvalue floor",
        },
    })
    output.json("metrics", "report.json", {"summary": summary, "scenes": scenes, "records": records})
    output.csv("metrics", "summary.csv", summary)
    output.csv("metrics", "scene_metrics.csv", records)
    print(f"Results: {output.root.resolve()}")


if __name__ == "__main__":
    main()
