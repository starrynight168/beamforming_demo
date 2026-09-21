from __future__ import annotations

import csv
import json
import os
import weakref
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .beamforming import (
    beamform_iq_with_tx_phase,
    build_dynamic_interpolation_cache,
    dynamic_interpolation_cache_bytes,
    normalize_db,
    predict_aperture_weights,
)
from .config import normalize_angle_selection
from .config import runtime as _runtime

write_log = _runtime.write_log
FULL_CHECK_SCHEMA = "mban-h5-full-check-v1"
_VERIFIED_FULL_CHECKS: set[tuple[str, int, int]] = set()


def resolve_angle_indices(angles: np.ndarray | torch.Tensor, selection: object) -> np.ndarray:
    values = np.asarray(angles, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("angles 必须是非空有限数组")
    selection = normalize_angle_selection(selection)
    if selection == "all":
        return np.arange(values.size, dtype=np.int64)
    if selection == "center":
        return np.asarray([int(np.argmin(np.abs(values)))], dtype=np.int64)
    if isinstance(selection, int):
        if selection > values.size:
            raise ValueError(f"请求角度数超过 H5 中的角度数: requested={selection}, total={values.size}")
        if selection == 1:
            return np.asarray([int(np.argmin(np.abs(values)))], dtype=np.int64)
        order = np.argsort(values)
        return order[np.linspace(0, values.size - 1, selection, dtype=int)].astype(np.int64)
    indices = np.asarray(selection, dtype=np.int64)
    if np.any(indices >= values.size):
        raise IndexError(f"angle_selection 索引超出 H5 范围 [0, {values.size - 1}]")
    return indices


def _amp_enabled(device: torch.device) -> bool:
    if device.type != "cuda" or os.environ.get("MBAN_DISABLE_AMP") == "1":
        return False
    return _runtime.args.mode == "qat" or os.environ.get("MBAN_FORCE_AMP") == "1"


def decode_h5_text(value) -> str:
    return value.decode("utf-8").strip("\x00") if isinstance(value, (bytes, np.bytes_)) else str(value)


def _portable_basename(path: str | os.PathLike[str]) -> str:
    return str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def require_full_check_certificate(h5_path: str | os.PathLike[str]) -> None:
    source_path = Path(h5_path).resolve()
    source = source_path.stat()
    cache_key = (str(source_path), source.st_size, source.st_mtime_ns)
    if cache_key in _VERIFIED_FULL_CHECKS:
        return
    certificate_path = Path(f"{source_path}.check.json")
    checker_path = Path(__file__).resolve().parents[2] / "data" / "checks" / "check_data.py"
    command = f'python "{checker_path}" --full "{source_path}"'
    try:
        with certificate_path.open("r", encoding="utf-8") as stream:
            certificate = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"H5缺少有效的全量检查凭证: {certificate_path}\n请先运行: {command}") from error
    certificate_source = certificate.get("source", {})
    checked_datasets = certificate.get("datasets", {})
    valid = (
        certificate.get("schema") == FULL_CHECK_SCHEMA
        and certificate.get("status") == "pass"
        and certificate_source.get("name") == source_path.name
        and certificate_source.get("size") == source.st_size
        and certificate_source.get("mtime_ns") == source.st_mtime_ns
        and {"all_multi_I", "all_multi_Q"} <= set(checked_datasets)
    )
    if not valid:
        raise RuntimeError(f"H5全量检查凭证失败或已过期: {certificate_path}\n请重新运行: {command}")
    _VERIFIED_FULL_CHECKS.add(cache_key)


def read_training_target_maps(h5) -> dict:
    if "config_yaml" not in h5:
        raise KeyError("H5缺少config_yaml，无法解析通用训练目标")
    metadata = yaml.safe_load(decode_h5_text(h5["config_yaml"][()])) or {}
    if not isinstance(metadata, dict):
        raise ValueError("H5 config_yaml 必须是 YAML 映射")
    generation = metadata.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("H5 config_yaml.generation 必须是映射")
    ground_truth = generation.get("ground_truth", {})
    if not isinstance(ground_truth, dict):
        raise ValueError("H5 config_yaml.generation.ground_truth 必须是映射")
    targets = generation.get("training_targets", ground_truth.get("training_targets", {}))
    if not isinstance(targets, dict) or not targets:
        raise KeyError("config_yaml缺少generation.training_targets")
    return targets


def h5_key(path_value: str) -> str:
    return str(path_value).lstrip("/")


def resolve_algorithm_targets(h5, algorithm: str) -> tuple[str, str, str | None, str]:
    targets = read_training_target_maps(h5)
    iq_targets = targets.get("complex_iq_targets", {})
    available = sorted(iq_targets)
    if algorithm not in iq_targets:
        raise KeyError(f"H5没有算法{algorithm!r}的复数IQ目标；可用算法: {available}")
    iq_entry = iq_targets[algorithm]
    i_key, q_key = h5_key(iq_entry["i"]), h5_key(iq_entry["q"])
    image_path = targets.get("image_targets", {}).get(algorithm)
    image_key = h5_key(image_path) if image_path else None
    norm_path = targets.get("norm_references", {}).get(algorithm)
    norm_key = h5_key(norm_path) if norm_path else "all_scale_ref"
    missing = [key for key in (i_key, q_key, norm_key) if key not in h5]
    if image_key is not None and image_key not in h5:
        missing.append(image_key)
    if missing:
        raise KeyError(f"算法{algorithm!r}的H5目标字段缺失: {missing}")
    return i_key, q_key, image_key, norm_key


def _read_h5_labels(h5_path: str, n_samples: int) -> tuple[list[str], list[str]]:
    with h5py.File(h5_path, "r") as hf:
        if "acquisition_id" in hf:
            acquisition_ids = [decode_h5_text(v) for v in hf["acquisition_id"][:]]
        else:
            acquisition_ids = [str(i) for i in range(n_samples)]
        if "body_region" in hf:
            body_regions = [decode_h5_text(v) for v in hf["body_region"][:]]
        else:
            body_regions = [""] * n_samples
    if len(acquisition_ids) != n_samples or len(body_regions) != n_samples:
        raise ValueError(
            f"H5标签长度与样本数不一致: {h5_path}, "
            f"samples={n_samples}, acquisition_id={len(acquisition_ids)}, body_region={len(body_regions)}"
        )
    return acquisition_ids, body_regions


def _validate_acquisition_split(
    h5_paths: list[str],
    split: dict[str, dict[str, np.ndarray]],
) -> None:
    for path in h5_paths:
        with h5py.File(path, "r") as hf:
            n_samples = int(hf["all_multi_I"].shape[0])
        acquisition_ids, _ = _read_h5_labels(path, n_samples)
        groups = {name: {acquisition_ids[int(index)] for index in indices} for name, indices in split[path].items()}
        overlap = (
            (groups["train"] & groups["val"]) | (groups["train"] & groups["test"]) | (groups["val"] & groups["test"])
        )
        if overlap:
            raise ValueError(f"acquisition_id 跨 train/val/test 泄漏: {path}, 示例={sorted(overlap)[:5]}")


def load_or_create_mixed_split(
    h5_paths: list[str],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    *,
    split_file: str | os.PathLike[str] | None = None,
    seed: int | None = None,
    make_new_split: bool | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Independently split every H5, then combine corresponding partitions."""
    if not h5_paths:
        raise ValueError("h5_paths 不能为空")
    ratios = np.asarray([train_ratio, validation_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0) or train_ratio <= 0 or validation_ratio <= 0 or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("train_ratio和validation_ratio必须>0，test_ratio必须>=0，三者之和必须为1")

    def counts_for_size(n_samples: int) -> tuple[int, int, int]:
        n_train = int(round(n_samples * train_ratio))
        n_val = int(round(n_samples * validation_ratio))
        n_test = n_samples - n_train - n_val
        if n_train < 1 or n_val < 1 or n_test < 0 or (test_ratio > 0 and n_test < 1):
            raise ValueError(f"样本数{n_samples}不足以按比例{ratios.tolist()}划分")
        return n_train, n_val, n_test

    h5_paths = [os.path.abspath(path) for path in h5_paths]
    split_seed = int(_runtime.args.seed if seed is None else seed)
    reuse_existing = not bool(_runtime.args.make_new_split if make_new_split is None else make_new_split)
    for path in h5_paths:
        require_full_check_certificate(path)
    path_by_name: dict[str, str] = {}
    for path in h5_paths:
        name = _portable_basename(path)
        if name in path_by_name:
            raise ValueError(f"多个H5文件同名，固定划分无法唯一匹配: {name}")
        path_by_name[name] = path
    ratio_tag = f"{train_ratio:g}-{validation_ratio:g}-{test_ratio:g}".replace(".", "p")
    split_csv = os.path.abspath(
        split_file
        or _runtime.args.split_file
        or os.path.join(_runtime.args.output_directory, f"mixed_split_seed{split_seed}_{ratio_tag}.csv")
    )
    split_names = ("train", "val", "test")

    if os.path.exists(split_csv) and reuse_existing:
        with open(split_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_columns = {"split", "original_index", "acquisition_id", "body_region", "seed", "source_h5"}
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(f"固定划分缺少列 {sorted(missing_columns)}: {split_csv}")
            rows = list(reader)
        try:
            stored_seeds = {int(row["seed"]) for row in rows}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"固定划分包含无效 seed: {split_csv}") from exc
        if stored_seeds != {split_seed}:
            raise ValueError(
                f"固定划分 seed={sorted(stored_seeds)} 与当前 seed={split_seed} 不一致: {split_csv}。"
                "请将make_new_split设为true重新生成。"
            )
        out = {path: {name: [] for name in split_names} for path in h5_paths}
        source_rows = {path: [] for path in h5_paths}
        for row in rows:
            source_name = row.get("source_h5", "").strip()
            if not source_name:
                raise ValueError(f"固定划分缺少 source_h5: {split_csv}")
            src = os.path.abspath(source_name)
            if src not in out:
                src = path_by_name.get(_portable_basename(source_name), src)
            if src not in out:
                raise ValueError(f"固定划分包含未知 source_h5={source_name!r}: {split_csv}")
            if row["split"] not in split_names:
                raise ValueError(f"固定划分包含未知 split={row['split']!r}: {split_csv}")
            out[src][row["split"]].append(int(row["original_index"]))
            source_rows[src].append(row)
        split = {
            path: {name: np.array(sorted(values), dtype=np.int64) for name, values in parts.items()}
            for path, parts in out.items()
        }
        for path in h5_paths:
            if len(split[path]["train"]) == 0 or len(split[path]["val"]) == 0:
                raise ValueError(f"混合划分文件缺少 train/val: {path} ({split_csv})")
            with h5py.File(path, "r") as hf:
                expected = counts_for_size(int(hf["all_multi_I"].shape[0]))
            actual = tuple(len(split[path][name]) for name in split_names)
            if actual != expected:
                raise ValueError(
                    f"固定划分与当前比例不一致: {path}, CSV={actual}, 当前要求={expected}。"
                    "请将make_new_split设为true重新生成。"
                )
            all_indices = np.concatenate([split[path][name] for name in split_names])
            if not np.array_equal(np.sort(all_indices), np.arange(sum(expected), dtype=np.int64)):
                raise ValueError(f"固定划分必须无重复、无遗漏且覆盖 [0, {sum(expected) - 1}]: {path} ({split_csv})")
            acquisition_ids, body_regions = _read_h5_labels(path, sum(expected))
            for row in source_rows[path]:
                index = int(row["original_index"])
                if row["acquisition_id"] != acquisition_ids[index] or row["body_region"] != body_regions[index]:
                    raise ValueError(
                        f"固定划分与当前H5样本标签不一致: {path}, index={index}。请将make_new_split设为true重新生成。"
                    )
        summary = []
        for path in h5_paths:
            summary.append(
                f"{os.path.basename(path)} train={len(split[path]['train'])}, "
                f"val={len(split[path]['val'])}, test={len(split[path]['test'])}"
            )
        write_log(f"[SPLIT] 复用 {split_csv} | " + " | ".join(summary), level="SUCCESS")
        _validate_acquisition_split(h5_paths, split)
        return split

    os.makedirs(os.path.dirname(split_csv) or ".", exist_ok=True)
    split: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    for file_idx, h5_path in enumerate(h5_paths):
        with h5py.File(h5_path, "r") as hf:
            n_samples = int(hf["all_multi_I"].shape[0])
        n_train, n_val, _ = counts_for_size(n_samples)
        rng = np.random.default_rng(split_seed + file_idx)
        indices = rng.permutation(n_samples)
        parts = {
            "train": np.sort(indices[:n_train]),
            "val": np.sort(indices[n_train : n_train + n_val]),
            "test": np.sort(indices[n_train + n_val :]),
        }
        split[h5_path] = parts
        acquisition_ids, body_regions = _read_h5_labels(h5_path, n_samples)
        for name in split_names:
            for idx in parts[name]:
                rows.append(
                    {
                        "split": name,
                        "original_index": int(idx),
                        "acquisition_id": acquisition_ids[idx],
                        "body_region": body_regions[idx],
                        "seed": split_seed,
                        "source_h5": h5_path,
                    }
                )

    _validate_acquisition_split(h5_paths, split)
    temporary_csv = f"{split_csv}.{os.getpid()}.tmp"
    try:
        with open(temporary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["split", "original_index", "acquisition_id", "body_region", "seed", "source_h5"],
            )
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_csv, split_csv)
    finally:
        if os.path.exists(temporary_csv):
            os.unlink(temporary_csv)

    summary = []
    for path in h5_paths:
        summary.append(
            f"{os.path.basename(path)} train={len(split[path]['train'])}, "
            f"val={len(split[path]['val'])}, test={len(split[path]['test'])}"
        )
    write_log(f"[SPLIT] 已生成 {split_csv} | " + " | ".join(summary), level="SUCCESS")
    return split


# =====================================================================
# Loss / GCF
# =====================================================================


class UltrasoundImageDataset(Dataset):
    """按H5元数据选择任意具有复数IQ字段的波束合成目标。"""

    @staticmethod
    def _read_selected(ds, selected_indices: np.ndarray, full_samples: int) -> np.ndarray:
        # Large HDF5 fancy-index reads can be much slower than one contiguous read.
        if len(selected_indices) / max(full_samples, 1) >= 0.5:
            return ds[:][selected_indices]
        return ds[selected_indices]

    def __init__(
        self,
        h5_path: str,
        name: str = "Data",
        angle_indices: list[int] | None = None,
        target_algorithm: str = "mv",
        sample_indices: np.ndarray | list[int] | None = None,
    ) -> None:
        super().__init__()
        require_full_check_certificate(h5_path)
        self.h5_path = h5_path
        self.target_algorithm = target_algorithm

        with h5py.File(self.h5_path, "r") as hf:
            iq_shape = hf["all_multi_I"].shape
            if len(iq_shape) != 4 or hf["all_multi_Q"].shape != iq_shape:
                raise ValueError(f"{name} 的 all_multi_I/all_multi_Q 必须是形状一致的 [N,A,T,C]")
            full_samples = hf["all_multi_I"].shape[0]
            if full_samples < 1:
                raise ValueError(f"{name} 的 H5 不含样本")
            if sample_indices is None:
                selected_indices = np.arange(full_samples, dtype=np.int64)
            else:
                selected_indices = np.sort(np.asarray(sample_indices, dtype=np.int64))
                if selected_indices.ndim != 1 or len(selected_indices) == 0:
                    raise ValueError(f"{name} 的 sample_indices 不能为空的一维索引")
                if selected_indices.min() < 0 or selected_indices.max() >= full_samples:
                    raise IndexError(f"{name} 的 sample_indices 超出 H5 范围 [0, {full_samples - 1}]")
                if np.unique(selected_indices).size != selected_indices.size:
                    raise ValueError(f"{name} 的 sample_indices 不能包含重复索引")
            self.sample_indices = selected_indices
            self.num_samples = len(selected_indices)
            self.geometry_cache_ids = torch.full((self.num_samples,), -1, dtype=torch.long)
            if "acquisition_id" in hf:
                self.acquisition_ids = [decode_h5_text(v) for v in hf["acquisition_id"][selected_indices]]
            else:
                self.acquisition_ids = []

            if angle_indices is None:
                self.all_multi_I = torch.from_numpy(
                    self._read_selected(hf["all_multi_I"], selected_indices, full_samples)
                ).float()
                self.all_multi_Q = torch.from_numpy(
                    self._read_selected(hf["all_multi_Q"], selected_indices, full_samples)
                ).float()
                self.time_start = torch.from_numpy(
                    self._read_selected(hf["time_start_vector"], selected_indices, full_samples)
                ).float()
                self.angles = torch.from_numpy(hf["angles"][:]).float()
            else:
                angle_indices = np.asarray(angle_indices, dtype=np.int64)
                available_angles = int(hf["all_multi_I"].shape[1])
                if (
                    angle_indices.ndim != 1
                    or angle_indices.size == 0
                    or np.unique(angle_indices).size != angle_indices.size
                    or angle_indices.min() < 0
                    or angle_indices.max() >= available_angles
                ):
                    raise ValueError(f"{name} 的 angle_indices 必须是 [0, {available_angles - 1}] 内无重复的一维索引")
                # h5py 不支持同时对样本维和角度维使用 fancy indexing；
                # 先从 H5 按样本读取，再在 NumPy 内存数组中选角度。
                self.all_multi_I = torch.from_numpy(
                    self._read_selected(hf["all_multi_I"], selected_indices, full_samples)[:, angle_indices, :, :]
                ).float()
                self.all_multi_Q = torch.from_numpy(
                    self._read_selected(hf["all_multi_Q"], selected_indices, full_samples)[:, angle_indices, :, :]
                ).float()
                self.time_start = torch.from_numpy(
                    self._read_selected(hf["time_start_vector"], selected_indices, full_samples)
                )[:, angle_indices].float()
                self.angles = torch.from_numpy(hf["angles"][:])[angle_indices].float()

            if "valid_time_samples" in hf:
                valid_time_values = np.asarray(
                    self._read_selected(hf["valid_time_samples"], selected_indices, full_samples)
                )
                if not np.all(np.isfinite(valid_time_values)) or not np.all(
                    valid_time_values == np.floor(valid_time_values)
                ):
                    raise ValueError(f"{name} 的 valid_time_samples 必须是有限整数")
                self.valid_time_samples = torch.from_numpy(valid_time_values.astype(np.int64, copy=False))
            else:
                self.valid_time_samples = torch.full(
                    (self.num_samples,),
                    self.all_multi_I.shape[2],
                    dtype=torch.long,
                )
            if self.valid_time_samples.ndim != 1 or self.valid_time_samples.numel() != self.num_samples:
                raise ValueError(f"{name} 的 valid_time_samples 必须为长度 {self.num_samples} 的一维数组")
            if torch.any(self.valid_time_samples < 2) or torch.any(self.valid_time_samples > self.all_multi_I.shape[2]):
                raise ValueError(f"{name} 的 valid_time_samples 必须在 [2, {self.all_multi_I.shape[2]}] 内")

            target_maps = read_training_target_maps(hf)
            self._display_gt_keys = {
                algorithm: h5_key(path)
                for algorithm, path in target_maps.get("image_targets", {}).items()
                if h5_key(path) in hf
            }
            self._display_gt_cache: dict[tuple[str, int], np.ndarray] = {}
            iq_i_key, iq_q_key, image_key, norm_ref_key = resolve_algorithm_targets(hf, target_algorithm)
            self._iq_target_keys = f"{iq_i_key.split('/')[-1]}/{iq_q_key.split('/')[-1]}"
            target_shape = hf[iq_i_key].shape
            if len(target_shape) != 4 or target_shape[0] != full_samples or target_shape[1] != 1:
                raise ValueError(f"{name} 的训练目标必须为 [N,1,H,W]")
            self.gt_height = int(hf[iq_i_key].shape[2])
            self.gt_width = int(hf[iq_i_key].shape[3])
            self.target_display = (
                None
                if image_key is None
                else torch.from_numpy(self._read_selected(hf[image_key], selected_indices, full_samples)).float()
            )
            self.all_norm_ref = torch.from_numpy(
                self._read_selected(hf[norm_ref_key], selected_indices, full_samples)
            ).float()
            self.target_i = torch.from_numpy(self._read_selected(hf[iq_i_key], selected_indices, full_samples)).float()
            self.target_q = torch.from_numpy(self._read_selected(hf[iq_q_key], selected_indices, full_samples)).float()
            if (
                self.target_i.shape != self.target_q.shape
                or self.target_i.ndim != 4
                or self.target_i.shape[0] != self.num_samples
                or self.target_i.shape[1] != 1
            ):
                raise ValueError(f"{name} 的训练目标 I/Q 必须是形状一致的 [N,1,H,W]")
            if self.all_norm_ref.shape[0] != self.num_samples:
                raise ValueError(f"{name} 的 norm reference 样本数不匹配")
            if self.target_display is not None and self.target_display.shape != self.target_i.shape:
                raise ValueError(f"{name} 的显示目标形状必须匹配 IQ 目标")
            # 物理常量
            self.fs = float(hf["fs"][()])
            self.c = float(hf["c"][()])
            self.fc = float(hf["fc"][()])  # pack 里必填，不再 fallback
            self.pitch = float(hf["pitch"][()])  # pack 里必填
            if "num_channels" in hf:
                self.num_channels = int(np.asarray(hf["num_channels"][()]).item())
            else:
                self.num_channels = self.all_multi_I.shape[-1]  # 兜底
            self.z_grid = torch.from_numpy(hf["z_grid"][:]).float()
            self.x_grid = torch.from_numpy(hf["x_grid"][:]).float()

        if self.all_multi_I.shape != self.all_multi_Q.shape:
            raise ValueError(f"{name} 的输入 IQ 形状非法")
        if self.time_start.shape != self.all_multi_I.shape[:2] or not torch.isfinite(self.time_start).all():
            raise ValueError(f"{name} 的 time_start_vector 必须为有限 [N,A]")
        if (
            self.angles.ndim != 1
            or self.angles.numel() != self.all_multi_I.shape[1]
            or not torch.isfinite(self.angles).all()
        ):
            raise ValueError(f"{name} 的 angles 必须匹配 IQ 角度维")
        for grid_name, grid, expected_size in (
            ("z_grid", self.z_grid, self.target_i.shape[2]),
            ("x_grid", self.x_grid, self.target_i.shape[3]),
        ):
            if (
                grid.ndim != 1
                or grid.numel() != expected_size
                or not torch.isfinite(grid).all()
                or not torch.all(torch.diff(grid) > 0)
            ):
                raise ValueError(f"{name} 的 {grid_name} 必须为与目标匹配的有限严格递增一维网格")
        if self.num_channels != self.all_multi_I.shape[-1] or not all(
            np.isfinite(value) and value > 0 for value in (self.fs, self.c, self.fc, self.pitch)
        ):
            raise ValueError(f"{name} 的物理常量或通道数非法")

    def __len__(self) -> int:
        return self.num_samples

    def get_display_gt(self, mode: str, idx: int) -> np.ndarray:
        if mode == self.target_algorithm and self.target_display is not None:
            return self.target_display[idx, 0].numpy()
        if mode not in self._display_gt_keys:
            if mode == self.target_algorithm:
                i = self.target_i[idx, 0].numpy()
                q = self.target_q[idx, 0].numpy()
                envelope = np.sqrt(i * i + q * q)
                db = 20.0 * np.log10(envelope / max(float(envelope.max()), 1.0e-12) + 1.0e-12)
                return np.clip((db + _runtime.DR) / _runtime.DR, 0.0, 1.0).astype(np.float32)
            raise KeyError(f"H5没有算法{mode!r}的显示域目标；可用: {sorted(self._display_gt_keys)}")
        cache_key = (mode, int(idx))
        if cache_key not in self._display_gt_cache:
            raw_idx = int(self.sample_indices[idx])
            with h5py.File(self.h5_path, "r") as hf:
                self._display_gt_cache[cache_key] = np.asarray(
                    hf[self._display_gt_keys[mode]][raw_idx, 0], dtype=np.float32
                )
        return self._display_gt_cache[cache_key]

    def __getitem__(self, idx: int):
        norm_ref = self.all_norm_ref[idx]
        if norm_ref.dim() > 1:
            norm_ref = norm_ref.mean()
        target_db = self.target_i[idx, 0] if self.target_display is None else self.target_display[idx, 0]
        return (
            self.all_multi_I[idx],
            self.all_multi_Q[idx],
            target_db,
            self.target_i[idx, 0],
            self.target_q[idx, 0],
            self.time_start[idx],
            norm_ref,
            self.valid_time_samples[idx],
            self.geometry_cache_ids[idx],
        )


def find_visualization_sample_index(dataset: UltrasoundImageDataset) -> int:
    """Return the requested frame's index inside the validation subset."""
    requested_id = _runtime.args.visualization_acquisition.strip()
    if not requested_id:
        return 0
    try:
        idx = dataset.acquisition_ids.index(requested_id)
    except ValueError as exc:
        raise ValueError(
            f"验证集不包含 --visualization_acquisition_id={requested_id!r}；"
            "请保留现有 split CSV，或改用验证集中的 acquisition_id。"
        ) from exc
    write_log(
        f"[VIS] 使用 {requested_id}（验证集局部索引={idx}，原始索引={dataset.sample_indices[idx]}）",
        level="SUCCESS",
    )
    return idx


def pick_visualization_dataset(val_datasets: list[UltrasoundImageDataset]) -> tuple[UltrasoundImageDataset, int]:
    requested_id = _runtime.args.visualization_acquisition.strip()
    if requested_id:
        for dataset in val_datasets:
            if requested_id in dataset.acquisition_ids:
                return dataset, find_visualization_sample_index(dataset)
        write_log(
            f"验证集不包含 --visualization_acquisition_id={requested_id!r}，改用第一个验证样本画图",
            level="WARNING",
        )
    return val_datasets[0], 0


# =====================================================================
# GPU 切片引擎
# =====================================================================


def extract_windows_on_gpu(
    rf_I_pad,
    rf_Q_pad,
    t_starts,
    img_idx,
    pixel_idx,
    fs,
    offsets,
    angle_idx,
    max_t_len,
    valid_time_samples,
    pitch,
    x_grid,
    z_grid,
    angles_rad,
):
    P = len(pixel_idx)
    T = len(offsets)
    device = rf_I_pad.device
    if rf_I_pad.shape != rf_Q_pad.shape or rf_I_pad.ndim != 4:
        raise ValueError(
            "IQ 插值输入必须是形状一致的 [batch, angle, time, channel] 张量: "
            f"I={tuple(rf_I_pad.shape)}, Q={tuple(rf_Q_pad.shape)}"
        )
    batch_size, available_angles, available_time, N = rf_I_pad.shape
    if available_time < 2:
        raise ValueError(f"IQ 时间维至少需要 2 个样本，实际为 {available_time}")
    if max_t_len != available_time:
        raise ValueError(f"传入 max_t_len={max_t_len} 与 IQ 实际时间维 {available_time} 不一致")
    valid_time_samples = torch.as_tensor(valid_time_samples, device=device).reshape(-1)
    if valid_time_samples.numel() != batch_size:
        raise ValueError(f"valid_time_samples 长度 {valid_time_samples.numel()} 与批大小 {batch_size} 不一致")
    if torch.any(valid_time_samples < 2) or torch.any(valid_time_samples > available_time):
        raise ValueError(f"valid_time_samples 必须在 [2, {available_time}] 内")
    pixel_count = int(z_grid.numel() * x_grid.numel())
    if t_starts.ndim != 2 or t_starts.shape[0] != batch_size or t_starts.shape[1] < available_angles:
        raise ValueError(
            "time_start_vector 应匹配 IQ 的 [batch, angle] 维度: "
            f"t0={tuple(t_starts.shape)}, IQ={tuple(rf_I_pad.shape)}"
        )
    if angles_rad.numel() < available_angles:
        raise ValueError(f"angles 长度 {angles_rad.numel()} 小于 IQ 角度维 {available_angles}")
    safe_img_idx = img_idx.long()
    safe_angle_idx = angle_idx.long()
    safe_pixel_idx = pixel_idx.long()
    for name, values, upper in (
        ("img_idx", safe_img_idx, batch_size),
        ("angle_idx", safe_angle_idx, available_angles),
        ("pixel_idx", safe_pixel_idx, pixel_count),
    ):
        if values.numel() and (torch.any(values < 0) or torch.any(values >= upper)):
            raise IndexError(f"{name} 必须在 [0, {upper - 1}] 内")

    x_count = int(x_grid.numel())
    z_idx = (safe_pixel_idx // x_count).long()
    x_idx = (safe_pixel_idx % x_count).long()
    depth = z_grid[z_idx]
    lateral_pixel = x_grid[x_idx]
    lateral_channel = (torch.arange(N, device=device).float() - (N - 1) / 2.0) * pitch

    dx = lateral_pixel.unsqueeze(1) - lateral_channel.unsqueeze(0)
    receive_dist = torch.sqrt(depth.unsqueeze(1) ** 2 + dx**2)

    A = int(safe_angle_idx.numel())
    theta = angles_rad[safe_angle_idx]
    tx_dist = depth.unsqueeze(1) * torch.cos(theta).unsqueeze(0) + lateral_pixel.unsqueeze(1) * torch.sin(
        theta
    ).unsqueeze(0)
    total_dist = tx_dist.unsqueeze(2) + receive_dist.unsqueeze(1)
    tx_tof = tx_dist / _runtime.c_global
    total_tof = total_dist / _runtime.c_global

    ts = t_starts[safe_img_idx]
    if ts.dim() == 1:
        ts = ts.unsqueeze(1)

    center_pos = (total_tof - ts.unsqueeze(2)) * fs
    t_pos = center_pos.unsqueeze(-1) + offsets.float().view(1, 1, 1, -1)
    sample_time_limit = valid_time_samples[safe_img_idx].view(P, 1, 1, 1)
    valid = (t_pos >= 0.0) & (t_pos < sample_time_limit - 1)
    t0 = torch.floor(t_pos).clamp(0, available_time - 2).long()
    # 再次封顶；即使上游时间轴或 offsets 异常，也绝不把 t1 用作越界索引。
    t1 = (t0 + 1).clamp_max(available_time - 1)
    frac = (t_pos - t0.float()).clamp(0.0, 1.0)

    c_ind = torch.arange(N, device=device).view(1, 1, N, 1)
    img_ind = safe_img_idx.view(-1, 1, 1, 1)
    a_ind = safe_angle_idx[:A].view(1, -1, 1, 1)

    one_minus_frac = 1.0 - frac
    extracted_I = rf_I_pad[img_ind, a_ind, t0, c_ind] * one_minus_frac + rf_I_pad[img_ind, a_ind, t1, c_ind] * frac
    extracted_Q = rf_Q_pad[img_ind, a_ind, t0, c_ind] * one_minus_frac + rf_Q_pad[img_ind, a_ind, t1, c_ind] * frac
    valid = valid.to(extracted_I.dtype)
    extracted_I = extracted_I * valid
    extracted_Q = extracted_Q * valid
    return extracted_I, extracted_Q, total_tof, tx_tof


def compute_aperture_mask(pixel_idx, depth_grid, x_grid, pitch, return_geometry: bool = False):
    F = _runtime.args.f_number
    z_idx_local = (pixel_idx // _runtime.IMG_W).long()
    x_idx_local = (pixel_idx % _runtime.IMG_W).long()
    d = depth_grid[z_idx_local]
    xp = x_grid[x_idx_local]
    center_ch = torch.round(xp / pitch + (_runtime.NUM_CHANNELS - 1) / 2.0).long().clamp(0, _runtime.NUM_CHANNELS - 1)
    k = torch.floor(d / (F * pitch)).long() + 1
    if bool(((k < 1) | (k > _runtime.NUM_CHANNELS)).any().item()):
        raise ValueError("动态孔径超出物理通道范围；请检查成像深度、F-number 和阵元间距")
    start = torch.clamp(center_ch - k // 2, min=0)
    start = torch.minimum(start, _runtime.NUM_CHANNELS - k)
    ch = torch.arange(_runtime.NUM_CHANNELS, device=pixel_idx.device).view(1, -1)
    mask = ((ch >= start.unsqueeze(1)) & (ch < (start + k).unsqueeze(1))).float()
    return (mask, start, k) if return_geometry else mask


def derive_network_channels(
    depth_grid: torch.Tensor,
    pitch: float,
    f_number: float,
    physical_channels: int,
    expected: int | None = None,
) -> tuple[int, torch.Tensor]:
    if pitch <= 0.0 or f_number <= 0.0 or physical_channels < 1:
        raise ValueError("pitch、f_number 和 physical_channels 必须为正数")
    depths = depth_grid.reshape(-1).float()
    if depths.numel() == 0 or not torch.isfinite(depths).all():
        raise ValueError("depth_grid 必须包含有限值")
    aperture_sizes = torch.floor(depths / (f_number * pitch)).long() + 1
    if int(aperture_sizes.min().item()) < 1:
        raise ValueError("depth_grid 计算得到的动态孔径必须至少为 1")
    maximum = int(aperture_sizes.max().item())
    if maximum > physical_channels:
        raise ValueError(
            f"几何计算得到最大动态孔径 {maximum}，超过物理通道数 {physical_channels}；"
            "请检查成像深度、F-number 和阵元间距"
        )
    if expected is not None and expected > 0 and maximum != expected:
        raise ValueError(
            f"几何计算得到 network_channels={maximum}，但当前模型要求 {expected}；"
            "请检查成像深度、F-number 和阵元间距"
        )
    return maximum, aperture_sizes


def build_fixed_geometry_cache(
    depth_grid: torch.Tensor,
    x_grid: torch.Tensor,
    angles_rad: torch.Tensor,
    angle_idx: torch.Tensor,
    pitch: float,
    fs: float,
    fc: float,
    t_start: torch.Tensor,
    max_t_len: int,
    dynamic_network_channels: int | None = None,
    dynamic_control_count: int | None = None,
) -> dict[str, torch.Tensor] | None:
    device = depth_grid.device
    pixel_count = int(depth_grid.numel() * x_grid.numel())
    angle_count = int(angle_idx.numel())
    channel_count = int(_runtime.NUM_CHANNELS)
    estimated_bytes = pixel_count * (angle_count * (channel_count * 21 + 4))
    if _runtime.args.dynamic_aperture:
        estimated_bytes += pixel_count * channel_count + pixel_count * 16
        if (
            dynamic_network_channels is not None
            and dynamic_control_count is not None
            and dynamic_control_count < dynamic_network_channels
        ):
            estimated_bytes += dynamic_interpolation_cache_bytes(
                dynamic_network_channels,
                dynamic_control_count,
                _runtime.args.output_interpolation,
            )
    if device.type != "cuda":
        return None
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if estimated_bytes > int(free_bytes * 0.60):
        write_log(
            f"固定几何缓存预计 {estimated_bytes / 2**30:.2f} GiB，超过当前空闲显存的60%，自动回退实时计算",
            level="WARNING",
        )
        return None

    cache = {
        "t0": torch.empty((pixel_count, angle_count, channel_count), dtype=torch.int64, device=device),
        "frac": torch.empty((pixel_count, angle_count, channel_count), dtype=torch.float32, device=device),
        "valid": torch.empty((pixel_count, angle_count, channel_count), dtype=torch.bool, device=device),
        "cos_rx": torch.empty((pixel_count, angle_count, channel_count), dtype=torch.float32, device=device),
        "sin_rx": torch.empty((pixel_count, angle_count, channel_count), dtype=torch.float32, device=device),
        "tx_tof": torch.empty((pixel_count, angle_count), dtype=torch.float32, device=device),
        "angle_index": torch.arange(angle_count, device=device),
        "channel_index": torch.arange(channel_count, device=device),
    }
    if _runtime.args.dynamic_aperture:
        cache["mask"] = torch.empty((pixel_count, channel_count), dtype=torch.bool, device=device)
        cache["aperture_start"] = torch.empty(pixel_count, dtype=torch.int64, device=device)
        cache["aperture_size"] = torch.empty(pixel_count, dtype=torch.int64, device=device)

    lateral_channel = (
        torch.arange(channel_count, device=device, dtype=torch.float32) - (channel_count - 1) / 2.0
    ) * pitch
    theta = angles_rad[angle_idx.long()]
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    t_start = t_start.to(device=device, dtype=torch.float32).reshape(1, angle_count, 1)
    chunk_size = min(32768, pixel_count)
    with torch.no_grad():
        for start in range(0, pixel_count, chunk_size):
            end = min(start + chunk_size, pixel_count)
            pixel_idx = torch.arange(start, end, device=device)
            z_idx = pixel_idx // int(x_grid.numel())
            x_idx = pixel_idx % int(x_grid.numel())
            depth = depth_grid[z_idx]
            lateral = x_grid[x_idx]
            receive_dist = torch.sqrt(depth[:, None].square() + (lateral[:, None] - lateral_channel[None, :]).square())
            tx_dist = depth[:, None] * cos_theta[None, :] + lateral[:, None] * sin_theta[None, :]
            total_dist = tx_dist[:, :, None] + receive_dist[:, None, :]
            tx_tof = tx_dist / _runtime.c_global
            total_tof = total_dist / _runtime.c_global
            center_pos = (total_tof - t_start) * fs
            valid = (center_pos >= 0.0) & (center_pos < max_t_len - 1)
            t0_float = torch.floor(center_pos).clamp(0, max_t_len - 2)
            phase = 2.0 * np.pi * fc * (total_tof - tx_tof[:, :, None])
            cache["t0"][start:end] = t0_float.to(torch.int64)
            cache["frac"][start:end] = (center_pos - t0_float).clamp(0.0, 1.0)
            cache["valid"][start:end] = valid
            cache["cos_rx"][start:end] = torch.cos(phase)
            cache["sin_rx"][start:end] = torch.sin(phase)
            cache["tx_tof"][start:end] = tx_tof
            if _runtime.args.dynamic_aperture:
                mask, aperture_start, aperture_size = compute_aperture_mask(
                    pixel_idx,
                    depth_grid,
                    x_grid,
                    pitch,
                    return_geometry=True,
                )
                cache["mask"][start:end] = mask.bool()
                cache["aperture_start"][start:end] = aperture_start
                cache["aperture_size"][start:end] = aperture_size
        if (
            _runtime.args.dynamic_aperture
            and dynamic_network_channels is not None
            and dynamic_control_count is not None
            and dynamic_control_count < dynamic_network_channels
        ):
            cache.update(
                build_dynamic_interpolation_cache(
                    dynamic_network_channels,
                    dynamic_control_count,
                    _runtime.args.output_interpolation,
                    device,
                )
            )
    return cache


def extract_aligned_iq_cached(
    rf_i: torch.Tensor,
    rf_q: torch.Tensor,
    img_idx: torch.Tensor,
    pixel_idx: torch.Tensor,
    geometry_cache: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t0 = geometry_cache["t0"][pixel_idx]
    t1 = t0 + 1
    frac = geometry_cache["frac"][pixel_idx]
    valid = geometry_cache["valid"][pixel_idx]
    _, a, n = t0.shape
    img_ind = img_idx.view(-1, 1, 1)
    angle_ind = geometry_cache["angle_index"].view(1, a, 1)
    channel_ind = geometry_cache["channel_index"].view(1, 1, n)
    one_minus_frac = 1.0 - frac
    i_sample = (
        rf_i[img_ind, angle_ind, t0, channel_ind] * one_minus_frac
        + rf_i[img_ind, angle_ind, t1, channel_ind] * frac
    )
    q_sample = (
        rf_q[img_ind, angle_ind, t0, channel_ind] * one_minus_frac
        + rf_q[img_ind, angle_ind, t1, channel_ind] * frac
    )
    valid = valid.to(i_sample.dtype)
    i_sample = i_sample * valid
    q_sample = q_sample * valid
    cos_rx = geometry_cache["cos_rx"][pixel_idx]
    sin_rx = geometry_cache["sin_rx"][pixel_idx]
    return (
        i_sample * cos_rx - q_sample * sin_rx,
        i_sample * sin_rx + q_sample * cos_rx,
        geometry_cache["tx_tof"][pixel_idx],
    )


# =====================================================================
# 单角度 DAS 调试波束形成器
# =====================================================================
def compute_single_angle_das(dataset, sample_idx, device, pitch, x_grid, depth_grid, angles_rad, angle_idx, offsets):
    rf_img_I = dataset.all_multi_I[sample_idx].unsqueeze(0).to(device)
    rf_img_Q = dataset.all_multi_Q[sample_idx].unsqueeze(0).to(device)
    t_starts = dataset.time_start[sample_idx].unsqueeze(0).to(device)
    norm_ref_sample = dataset.all_norm_ref[sample_idx].to(device)
    if norm_ref_sample.numel() == 1:
        norm_ref_flat = norm_ref_sample.expand(_runtime.NUM_PIXELS)
    else:
        norm_ref_flat = norm_ref_sample.view(-1)

    max_t_len = rf_img_I.shape[2]
    valid_time_samples = dataset.valid_time_samples[sample_idx : sample_idx + 1].to(device)
    preds = []
    batch_pixels = max(int(_runtime.args.optimization_batch_pixels), 1)

    with torch.no_grad():
        for i in range(0, _runtime.NUM_PIXELS, batch_pixels):
            end = min(i + batch_pixels, _runtime.NUM_PIXELS)
            pixel_idx = torch.arange(i, end, device=device)
            img_idx = torch.zeros(end - i, dtype=torch.long, device=device)
            ext_I, ext_Q, tof, tx_tof = extract_windows_on_gpu(
                rf_img_I,
                rf_img_Q,
                t_starts,
                img_idx,
                pixel_idx,
                dataset.fs,
                offsets,
                angle_idx,
                max_t_len,
                valid_time_samples,
                pitch,
                x_grid,
                depth_grid,
                angles_rad,
            )
            center_t = ext_I.shape[-1] // 2
            I_c = ext_I[:, :, :, center_t]
            Q_c = ext_Q[:, :, :, center_t]
            phase = 2.0 * np.pi * dataset.fc * (tof - tx_tof.unsqueeze(2))
            cos_p, sin_p = torch.cos(phase), torch.sin(phase)
            I_a = I_c * cos_p - Q_c * sin_p
            Q_a = I_c * sin_p + Q_c * cos_p

            if _runtime.args.dynamic_aperture:
                mask = compute_aperture_mask(pixel_idx, depth_grid, x_grid, pitch).unsqueeze(1)
            else:
                mask = torch.ones_like(I_a)
            denom = mask.sum(dim=2).clamp(min=1.0)
            i_angle = torch.sum(I_a * mask, dim=2) / denom
            q_angle = torch.sum(Q_a * mask, dim=2) / denom
            tx_phase = 2.0 * np.pi * dataset.fc * tx_tof
            cos_tx, sin_tx = torch.cos(tx_phase), torch.sin(tx_phase)
            I_out = torch.mean(i_angle * cos_tx - q_angle * sin_tx, dim=1)
            Q_out = torch.mean(i_angle * sin_tx + q_angle * cos_tx, dim=1)
            env = torch.sqrt(I_out**2 + Q_out**2)

            batch_norm_ref = norm_ref_flat[pixel_idx]
            if _runtime.args.use_tgc:
                z_l = (pixel_idx // _runtime.IMG_W).long()
                d_l = depth_grid[z_l]
                tgc_v = 10.0 ** (0.5 * (dataset.fc / 1e6) * (d_l * 100.0) * 2.0 / 20.0)
                env_p = env * tgc_v / (batch_norm_ref + 1e-12)
            else:
                env_p = env / (batch_norm_ref + 1e-12)
            pred_dB = normalize_db(env_p)
            preds.append(pred_dB.cpu().numpy())
            del ext_I, ext_Q

    return np.concatenate(preds).reshape(_runtime.IMG_H, _runtime.IMG_W)


# =====================================================================
# 重构图（Single-Angle DAS / Multi-Angle DAS-GT / MV-GT / MBAN）
# =====================================================================
def save_reconstruction_image(
    model,
    dataset,
    sample_idx,
    epoch,
    device,
    angle_idx,
    offsets,
    fs,
    pitch,
    x_grid,
    depth_grid,
    angles_rad,
    geometry_caches: dict[int, dict[str, torch.Tensor] | None] | None = None,
):
    if getattr(model, "hardware_qat", None) is not None:
        model.hardware_qat.disable_observer()
        if model.hardware_qat.config.noise_enabled:
            model.hardware_qat.begin_noise_realization()
    model.eval()
    baseline_cache = getattr(save_reconstruction_image, "_baseline_cache", None)
    if baseline_cache is None:
        baseline_cache = weakref.WeakKeyDictionary()
        save_reconstruction_image._baseline_cache = baseline_cache
    dataset_cache = baseline_cache.setdefault(dataset, {})
    cache_key = (
        int(sample_idx),
        _runtime.args.target_algorithm,
        bool(_runtime.args.use_tgc),
        bool(_runtime.args.dynamic_aperture),
        float(_runtime.args.f_number),
    )
    if cache_key not in dataset_cache:
        das_img = compute_single_angle_das(
            dataset,
            sample_idx,
            device,
            pitch,
            x_grid,
            depth_grid,
            angles_rad,
            angle_idx,
            offsets,
        )
        target_gt_img = dataset.get_display_gt(_runtime.args.target_algorithm, sample_idx)
        das_gt_img = dataset.get_display_gt("das", sample_idx) if "das" in dataset._display_gt_keys else das_img
        dataset_cache[cache_key] = (das_img, das_gt_img, target_gt_img)
    else:
        das_img, das_gt_img, target_gt_img = dataset_cache[cache_key]

    rf_img_I = dataset.all_multi_I[sample_idx].unsqueeze(0).to(device)
    rf_img_Q = dataset.all_multi_Q[sample_idx].unsqueeze(0).to(device)
    t_starts = dataset.time_start[sample_idx].unsqueeze(0).to(device)
    norm_ref_sample = dataset.all_norm_ref[sample_idx].to(device)
    if norm_ref_sample.numel() == 1:
        norm_ref_flat = norm_ref_sample.expand(_runtime.NUM_PIXELS)
    else:
        norm_ref_flat = norm_ref_sample.view(-1)

    max_t_len = rf_img_I.shape[2]
    valid_time_samples = dataset.valid_time_samples[sample_idx : sample_idx + 1].to(device)
    INFER_BATCH_SIZE = _runtime.args.optimization_batch_pixels
    cache_id = int(dataset.geometry_cache_ids[sample_idx])
    geometry_cache = None if not geometry_caches else geometry_caches.get(cache_id)

    mban_preds = []
    with torch.no_grad():
        for i in range(0, _runtime.NUM_PIXELS, INFER_BATCH_SIZE):
            end = min(i + INFER_BATCH_SIZE, _runtime.NUM_PIXELS)
            size = end - i
            pixel_idx = torch.arange(i, end, device=device)
            img_idx = torch.zeros(size, dtype=torch.long, device=device)
            if geometry_cache is None:
                ext_I, ext_Q, tof, tx_tof = extract_windows_on_gpu(
                    rf_img_I,
                    rf_img_Q,
                    t_starts,
                    img_idx,
                    pixel_idx,
                    fs,
                    offsets,
                    angle_idx,
                    max_t_len,
                    valid_time_samples,
                    pitch,
                    x_grid,
                    depth_grid,
                    angles_rad,
                )
                center_t = ext_I.shape[-1] // 2
                I_c = ext_I[:, :, :, center_t]
                Q_c = ext_Q[:, :, :, center_t]
                phase = 2.0 * np.pi * _runtime.fc_global * (tof - tx_tof.unsqueeze(2))
                cos_p, sin_p = torch.cos(phase), torch.sin(phase)
                I_a = I_c * cos_p - Q_c * sin_p
                Q_a = I_c * sin_p + Q_c * cos_p
            else:
                I_a, Q_a, tx_tof = extract_aligned_iq_cached(
                    rf_img_I,
                    rf_img_Q,
                    img_idx,
                    pixel_idx,
                    geometry_cache,
                )
            with torch.amp.autocast("cuda", enabled=_amp_enabled(device)):
                mask = aperture_start = aperture_size = None
                if _runtime.args.dynamic_aperture:
                    if geometry_cache is not None and "mask" in geometry_cache:
                        mask = geometry_cache["mask"][pixel_idx]
                        aperture_start = geometry_cache["aperture_start"][pixel_idx]
                        aperture_size = geometry_cache["aperture_size"][pixel_idx]
                    else:
                        mask, aperture_start, aperture_size = compute_aperture_mask(
                            pixel_idx,
                            depth_grid,
                            x_grid,
                            pitch,
                            return_geometry=True,
                        )
                I_use, Q_use, weights, controls, input_scale, input_mean, _ = predict_aperture_weights(
                    model,
                I_a,
                Q_a,
                mask,
                aperture_start,
                    aperture_size,
                    depth_index=pixel_idx // _runtime.IMG_W,
                    depth_count=_runtime.IMG_H,
                    dynamic_cache=geometry_cache,
                )

                I_out, Q_out = beamform_iq_with_tx_phase(
                    weights,
                    I_use,
                    Q_use,
                    tx_tof,
                    input_scale,
                    input_mean,
                    controls=controls,
                    aperture_start=aperture_start,
                    aperture_size=aperture_size,
                    network_channels=model.network_channels,
                )
                env = torch.sqrt(I_out**2 + Q_out**2)
                batch_norm_ref = norm_ref_flat[pixel_idx]
                if _runtime.args.use_tgc:
                    z_l = (pixel_idx // _runtime.IMG_W).long()
                    d_l = depth_grid[z_l]
                    tgc_v = 10.0 ** (0.5 * (_runtime.fc_global / 1e6) * (d_l * 100.0) * 2.0 / 20.0)
                    env_p = env * tgc_v / (batch_norm_ref + 1e-12)
                else:
                    env_p = env / (batch_norm_ref + 1e-12)
                pred_dB = normalize_db(env_p)
            mban_preds.append(pred_dB.cpu().numpy())
            if geometry_cache is None:
                del ext_I, ext_Q

    mban_img = np.concatenate(mban_preds).reshape(_runtime.IMG_H, _runtime.IMG_W)
    panels = [
        (f"MBAN Ep {epoch}", mban_img),
        (f"{_runtime.args.target_algorithm.upper()}-GT", target_gt_img),
        ("Single-Angle DAS", das_img),
        ("DAS-GT (Multi-Angle)", das_gt_img),
    ]
    plt.figure(figsize=(20, 6))
    for idx, (title, img) in enumerate(panels, start=1):
        plt.subplot(1, 4, idx)
        plt.title(title, fontsize=12)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(_runtime.args.output_directory, f"val_reconstruct_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    write_log(f"[IMAGE] epoch {epoch} 对比图已保存: {save_path}", level="SUCCESS")
    return mban_img
