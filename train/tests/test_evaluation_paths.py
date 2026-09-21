"""Regression tests for shared evaluation, hardware, and beamforming paths."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_DIR))

from mban_core import beamforming
from mban_core.config import MODEL_SEMANTICS_VERSION, load_config, load_nonideal_profile, runtime
from mban_core.data import derive_network_channels, resolve_angle_indices
from mban_core.hardware import QATConfig, resolve_layer_bias_implementation, resolve_nonideal_profile_raw
from mban_core.hardware_backend import HardwareQAT, MemristorLinear
from mban_core import model as model_module
from mban_core.model import MBAN
from mban_core.naming import output_controls_name, output_preactivation_name
from mban_core.training_support import (
    _validate_checkpoint_semantics,
    count_nonfinite_gradients,
    count_nonfinite_parameters,
)
from eval_core.config import (
    load_model_requests,
    parse_model_value,
    split_execution_overrides,
)
from eval_core.config import ModelSpec, resolve_execution_implementation, validate_deployment_checkpoint
from eval_core.hardware import fold_checkpoint, read_hidden_signal_references
from eval_core.hardware_trace import HardwareTrace, TraceCollector
from eval_core.inference import WeightPathCollector, require_explicit_weight_evaluation
from eval_core.inference import bootstrap_mean_ci, read_test_indices, select_test_indices


class DeploymentFoldTests(unittest.TestCase):
    @staticmethod
    def _checkpoint(
        normalization: str, bias_implementation: str, weight_transform: str = "none"
    ) -> dict[str, object]:
        torch.manual_seed(94041)
        array_bias = bias_implementation == "array"
        input_width = 5 if array_bias else 4
        state: dict[str, torch.Tensor] = {
            "fc1.weight": torch.randn(3, input_width),
            "fc2.weight": torch.randn(2, 3),
        }
        if not array_bias:
            state["fc1.bias"] = torch.randn(3)
            state["fc2.bias"] = torch.randn(2)
        model_config: dict[str, object] = {
            "total_fc_layers": 2,
            "hidden_layers": 1,
            "normalization": normalization,
            "normalization_layers": ["fc1"] if normalization != "none" else [],
            "centering": "none",
            "centering_layers": [],
            "weight_transform": weight_transform,
            "weight_transform_layers": ["fc1", "fc2"] if weight_transform == "ws" else [],
            "running_stat_epsilon": 1.0e-5,
            "bias_layers": ["fc1", "fc2"],
            "bias_implementation": bias_implementation,
        }
        deployment = {key: value.clone() for key, value in state.items()}
        if weight_transform == "ws":
            for layer in ("fc1", "fc2"):
                weight = state[f"{layer}.weight"]
                mean = weight.mean(dim=1, keepdim=True)
                variance = (weight - mean).square().mean(dim=1, keepdim=True)
                deployment[f"{layer}.weight"] = (weight - mean) * torch.rsqrt(variance + 1.0e-5)
        if normalization != "none":
            running_mean = torch.tensor([0.2, -0.3, 0.4])
            running_var = torch.tensor([0.7, 1.3, 2.1])
            norm_state = {
                "running_standardizers.fc1.running_mean": running_mean,
                "running_standardizers.fc1.running_var": running_var,
            }
            scale = torch.rsqrt(running_var + 1.0e-5)
            if normalization == "batch_renorm":
                affine_weight = torch.tensor([1.2, 0.8, -0.6])
                affine_bias = torch.tensor([-0.1, 0.2, 0.3])
                norm_state.update(
                    {
                        "running_standardizers.fc1.weight": affine_weight,
                        "running_standardizers.fc1.bias": affine_bias,
                    }
                )
                scale = scale * affine_weight
            else:
                affine_bias = torch.zeros(3)
            if array_bias:
                weight = deployment["fc1.weight"]
                deployment["fc1.weight"] = torch.cat(
                    ((weight[:, :-1] * scale[:, None]), ((weight[:, -1] - running_mean) * scale + affine_bias).unsqueeze(1)),
                    dim=1,
                )
            else:
                deployment["fc1.weight"] = deployment["fc1.weight"] * scale[:, None]
                deployment["fc1.bias"] = (state["fc1.bias"] - running_mean) * scale + affine_bias
            state.update(norm_state)
        return {
            "model_semantics_version": MODEL_SEMANTICS_VERSION,
            "model_config": model_config,
            "evaluation_config": {
                "input_normalization": "std",
                "control_normalization": "linf",
            },
            "model_state_dict": state,
            "deployment_linear_state": deployment,
            "hardware_qat_state": {
                "activation_ranges": {"fc1.activation": 1.0},
                "weight_scales": {"fc1": [1.0]},
                "realization_state": {"seed": 17},
            },
        }

    def test_fold_checks_all_normalization_and_bias_branches(self) -> None:
        for normalization in ("running_zscore", "batch_renorm"):
            for bias_implementation in ("ordinary", "array"):
                with self.subTest(normalization=normalization, bias_implementation=bias_implementation):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        source = root / "source.pth"
                        destination = root / "folded.pth"
                        torch.save(self._checkpoint(normalization, bias_implementation), source)
                        fold_checkpoint(source, destination)
                        folded = torch.load(destination, map_location="cpu", weights_only=False)
                        self.assertTrue(folded["deployment_folded"])
                        self.assertEqual(folded["model_config"]["normalization"], "none")
                        self.assertEqual(folded["model_config"]["normalization_layers"], [])
                        self.assertEqual(
                            folded["deployment_runtime_transforms"],
                            {
                                "input_normalization": "std",
                                "control_normalization": "linf",
                                "input_normalization_folded": False,
                                "control_normalization_folded": False,
                            },
                        )
                        self.assertTrue(folded["deployment_fold_parity"]["passed"])
                        self.assertIn("activation_ranges", folded["hardware_qat_state"])
                        self.assertFalse(any(key.startswith("running_standardizers.") for key in folded["model_state_dict"]))
                        self.assertTrue(destination.with_name(destination.name + ".fold_parity.json").is_file())
                        validate_deployment_checkpoint(folded, destination)

    def test_fold_rejects_incomplete_deployment_state(self) -> None:
        checkpoint = self._checkpoint("batch_renorm", "ordinary")
        del checkpoint["deployment_linear_state"]["fc2.weight"]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            torch.save(checkpoint, source)
            with self.assertRaisesRegex(ValueError, "deployment_linear_state.*不完整"):
                fold_checkpoint(source, source.with_name("folded.pth"))

    def test_fold_checks_noop_and_weight_transform_branch(self) -> None:
        for bias_implementation in ("ordinary", "array"):
            with self.subTest(bias_implementation=bias_implementation):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = root / "source.pth"
                    destination = root / "folded.pth"
                    torch.save(self._checkpoint("none", bias_implementation, "ws"), source)
                    fold_checkpoint(source, destination)
                    folded = torch.load(destination, map_location="cpu", weights_only=False)
                    self.assertTrue(folded["deployment_folded"])
                    self.assertEqual(folded["model_config"]["weight_transform"], "none")
                    self.assertTrue(folded["deployment_fold_parity"]["passed"])

    def test_fold_rejects_dynamic_or_sample_centering_paths(self) -> None:
        for normalization, centering in (("l1", "none"), ("l2", "none"), ("batch_renorm", "sample_mean")):
            with self.subTest(normalization=normalization, centering=centering):
                checkpoint = self._checkpoint(normalization, "ordinary")
                checkpoint["model_config"]["centering"] = centering
                checkpoint["model_config"]["centering_layers"] = ["fc1"] if centering != "none" else []
                with tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / "source.pth"
                    torch.save(checkpoint, source)
                    with self.assertRaises(ValueError):
                        fold_checkpoint(source, source.with_name("folded.pth"))

    def test_deployment_validation_rejects_unfolded_checkpoint(self) -> None:
        checkpoint = self._checkpoint("batch_renorm", "ordinary")
        with self.assertRaisesRegex(ValueError, "不是部署态"):
            validate_deployment_checkpoint(checkpoint, Path("source.pth"))

    def test_none_checkpoint_is_deployable_without_explicit_fold(self) -> None:
        checkpoint = self._checkpoint("none", "ordinary")
        checkpoint["deployment_folded"] = True
        validate_deployment_checkpoint(checkpoint, Path("source.pth"))

    def test_runtime_normalizations_are_validated_and_not_folded(self) -> None:
        checkpoint = self._checkpoint("none", "ordinary")
        checkpoint["evaluation_config"]["input_normalization"] = "invalid"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pth"
            torch.save(checkpoint, source)
            with self.assertRaisesRegex(ValueError, "input_normalization"):
                fold_checkpoint(source, source.with_name("folded.pth"))


class HardwarePathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_args = runtime.args
        runtime.args = SimpleNamespace(
            control_normalization="linf",
            control_bits=4,
            control_adc_full_scale=1.0,
            control_adc_range="full_scale",
            beamforming_implementation="explicit",
            output_weights="real",
            output_domain="nonnegative",
            unity_constraint="hard",
            unity_scope="global",
            angle_selection="center",
            angle_reduction="sum",
            output_interpolation="linear",
            interpolation_bits=32,
            input_normalization="none",
            projection_controls=0,
            loss={
                "error_function": "mse",
                "charbonnier_epsilon": 1.0e-3,
                "envelope_epsilon": 1.0e-6,
                "terms": {
                    "iq": {"weight": 1.0},
                    "envelope": {"weight": 0.1},
                    "unity": {"weight": 0.0},
                    "d1": {"domain": "shape", "weight": 0.0},
                    "d2": {"domain": "shape", "weight": 0.0},
                    "lrange": {"limit": 8.0, "weight": 0.0},
                },
            },
        )
        self.previous_fc_global = runtime.fc_global
        runtime.fc_global = 1.0

    def tearDown(self) -> None:
        runtime.args = self.previous_args
        runtime.fc_global = self.previous_fc_global

    def test_invalid_reference_uses_active_uniform_fallback(self) -> None:
        class InvalidReference:
            def control_reference(self, maximum: torch.Tensor, name: str) -> torch.Tensor:
                return torch.full_like(maximum, -1.0)

        controls = torch.tensor([[[0.2, 0.4, 0.0, 0.0], [0.0, 0.0, 0.3, 0.6]]])
        active_mask = torch.tensor([[True, True, False, False]])
        normalized = beamforming.normalize_controls_for_adc(
            controls,
            active_mask,
            InvalidReference(),
            output_controls_name("fc3"),
        )
        expected = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]])
        torch.testing.assert_close(normalized, expected)
        self.assertTrue(torch.isfinite(normalized).all())

    def test_previous_checkpoint_semantics_are_rejected(self) -> None:
        checkpoint = {"model_semantics_version": MODEL_SEMANTICS_VERSION - 1}
        with self.assertRaisesRegex(ValueError, "model_semantics_version"):
            _validate_checkpoint_semantics(checkpoint, "legacy.pth")

    def test_calibration_rejects_legacy_hidden_signal_reference(self) -> None:
        payload = {
            "definition": {"inter_layer": ["analog"]},
            "shared_reference": {"activation_references": {"fc1": 1.0}},
        }
        with self.assertRaisesRegex(ValueError, "hidden_signal_references"):
            read_hidden_signal_references(payload)

    def test_wizard_does_not_silently_mutate_semantic_fields(self) -> None:
        from copy import deepcopy
        from mban_wizard_cn import sync_dependent_fields

        config = {
            "training": {"resume": "all", "initial_checkpoint": "legacy.pth"},
            "backbone": {
                "hidden_layers": 2,
                "layer_activations": {"fc1": "relu", "fc3": "relu"},
                "centering": "none",
                "centering_layers": ["fc3"],
                "normalization": "none",
                "normalization_layers": ["fc3"],
                "weight_transform": "none",
                "weight_transform_layers": ["fc3"],
                "branch_mode": "single",
            },
            "output_head": {"output_controls": 8, "projection_controls": 2},
        }
        original = deepcopy(config)

        sync_dependent_fields(config)

        self.assertEqual(config, original)

    def test_config_loader_accepts_string_paths(self) -> None:
        config_path = TRAIN_DIR / "config.yaml"
        hardware_path = config_path.with_name("hardware_eval.yaml")
        self.assertEqual(QATConfig().inter_layer, "none")
        loaded = load_config(str(config_path), hardware_config_path=str(hardware_path))
        self.assertEqual(loaded.normalization, "none")
        self.assertEqual(loaded.control_normalization, "none")
        self.assertEqual(loaded.angle_selection, "center")
        self.assertEqual(loaded.angle_reduction, "sum")
        self.assertEqual(loaded.bias_bits, 4)
        self.assertFalse(hasattr(loaded, "beamforming_implementation"))
        self.assertIsNone(loaded.expected_network_channels)
        one_control = load_config(
            str(config_path),
            hardware_config_path=str(hardware_path),
            overrides=["output_controls=1"],
        )
        self.assertEqual(one_control.output_controls, 1)
        self.assertEqual(
            [stage["profile"] for stage in loaded.qat_schedule["stages"]],
            ["ideal", "common"],
        )
        overridden = load_config(
            str(config_path),
            hardware_config_path=str(hardware_path),
            profile_name="ideal",
        )
        self.assertEqual(
            [stage["profile"] for stage in overridden.qat_schedule["stages"]],
            ["ideal", "ideal"],
        )
        bias_bits = load_config(
            str(config_path),
            hardware_config_path=str(hardware_path),
            overrides=["bias_bits=8"],
        )
        self.assertEqual(bias_bits.bias_bits, 8)
        multi_angle = load_config(
            str(config_path),
            hardware_config_path=str(hardware_path),
            overrides=["angle_selection=all", "angle_reduction=mean", "unity_scope=per_angle"],
        )
        self.assertEqual(multi_angle.angle_selection, "all")
        self.assertEqual(multi_angle.angle_reduction, "mean")
        self.assertEqual(multi_angle.unity_scope, "per_angle")
        indexed_angles = load_config(
            str(config_path),
            hardware_config_path=str(hardware_path),
            overrides=["angle_selection=[0,2,5]"],
        )
        self.assertEqual(indexed_angles.angle_selection, [0, 2, 5])

        for field, mode, expected in (
            ("centering_layers", "centering=sample_mean", ["fc1", "fc2"]),
            ("normalization_layers", "normalization=l2", ["fc1", "fc2"]),
            ("weight_transform_layers", "weight_transform=ws", ["fc1", "fc2"]),
        ):
            normalized = load_config(
                str(config_path),
                hardware_config_path=str(hardware_path),
                overrides=[mode, f"{field}=[fc2,fc1]"],
            )
            self.assertEqual(getattr(normalized, field), expected)
            with self.assertRaisesRegex(ValueError, f"{field} 不能包含重复层名"):
                load_config(
                    str(config_path),
                    hardware_config_path=str(hardware_path),
                    overrides=[mode, f"{field}=[fc1,fc1]"],
                )
        for interpolation in ("linear", "nearest", "cubic"):
            for domain, unity in (
                ("nonnegative", "hard"),
                ("nonnegative", "free"),
                ("signed", "hard"),
                ("signed", "free"),
            ):
                combination = load_config(
                    str(config_path),
                    hardware_config_path=str(hardware_path),
                    overrides=[
                        f"output_interpolation={interpolation}",
                        f"output_domain={domain}",
                        f"unity_constraint={unity}",
                        "control_adc_range=full_scale",
                        "output_controls=0",
                        "projection_controls=2",
                    ],
                )
                self.assertEqual(combination.output_interpolation, interpolation)
                self.assertEqual(combination.output_domain, domain)
                self.assertEqual(combination.unity_constraint, unity)
        for obsolete_range in ("unit_interval", "symmetric_unit"):
            with self.assertRaisesRegex(ValueError, "control_adc_range 必须是 observer 或 full_scale"):
                load_config(
                    str(config_path),
                    hardware_config_path=str(hardware_path),
                    overrides=[f"control_adc_range={obsolete_range}"],
                )

        with self.assertRaisesRegex(ValueError, "output_controls 与 projection_controls"):
            load_config(
                str(config_path),
                hardware_config_path=str(hardware_path),
                overrides=[
                    "output_controls=8",
                    "projection_controls=2",
                ],
            )

        with self.assertRaisesRegex(ValueError, "--set 未知配置字段"):
            load_config(
                str(config_path),
                hardware_config_path=str(hardware_path),
                overrides=[
                    "output_controls=0",
                    "projection_controls=2",
                    "beamforming_implementation=factorized",
                ],
            )
        with self.assertRaisesRegex(ValueError, "投影控制点数量"):
            load_config(
                str(config_path),
                hardware_config_path=str(hardware_path),
                overrides=["projection_controls=-1"],
            )

    def test_angle_selection_resolves_center_count_all_and_indices(self) -> None:
        angles = torch.tensor([-0.2, 0.1, 0.0, 0.3, -0.1])
        self.assertEqual(resolve_angle_indices(angles, "center").tolist(), [2])
        self.assertEqual(resolve_angle_indices(angles, "all").tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(resolve_angle_indices(angles, 3).tolist(), [0, 2, 3])
        self.assertEqual(resolve_angle_indices(angles, [4, 1]).tolist(), [4, 1])

    def test_ordinary_bias_path_follows_inter_layer(self) -> None:
        config_path = TRAIN_DIR / "config.yaml"
        hardware_path = config_path.with_name("hardware_eval.yaml")
        for inter_layer in ("none", "analog", "digital"):
            config = load_config(
                str(config_path),
                hardware_config_path=str(hardware_path),
                overrides=[f"inter_layer={inter_layer}", "bias_implementation=ordinary"],
            )
            self.assertEqual(config.bias_implementation, "ordinary")
            self.assertEqual(resolve_layer_bias_implementation(config.bias_implementation, inter_layer), inter_layer)
        array_config = load_config(
            str(config_path),
            hardware_config_path=str(hardware_path),
            overrides=["inter_layer=digital", "bias_implementation=array"],
        )
        self.assertEqual(array_config.bias_implementation, "array")

    def test_bias_layers_control_linear_bias_parameters(self) -> None:
        previous_runtime_args = model_module._runtime_args
        model_module.set_runtime_args(
            SimpleNamespace(
                qat_enabled=False,
                output_weights="real",
                unity_constraint="none",
                output_domain="raw",
            )
        )
        try:
            no_bias = MBAN(
                num_channels=2,
                hidden_width=4,
                hidden_layers=1,
                output_controls=1,
                input_tile_features=2,
                bias_layers=[],
            )
            partial_bias = MBAN(
                num_channels=2,
                hidden_width=4,
                hidden_layers=1,
                output_controls=1,
                bias_layers=["fc2"],
            )
            self.assertIsNone(no_bias.fc1.bias)
            self.assertIsNone(no_bias.fc2.bias)
            self.assertEqual(no_bias(torch.ones(3, 2), torch.ones(3, 2)).shape, (3, 1))
            self.assertIsNone(partial_bias.fc1.bias)
            self.assertIsNotNone(partial_bias.fc2.bias)
        finally:
            model_module.set_runtime_args(previous_runtime_args)

    def test_array_bias_preserves_ordinary_initialization_stream(self) -> None:
        previous_runtime_args = model_module._runtime_args
        model_module.set_runtime_args(
            SimpleNamespace(
                qat_enabled=False,
                output_weights="real",
                unity_constraint="hard",
                output_domain="nonnegative",
            )
        )
        try:
            torch.manual_seed(42)
            ordinary = MBAN(
                num_channels=8,
                hidden_width=4,
                hidden_layers=2,
                output_controls=3,
                bias_layers=["fc1", "fc2", "fc3"],
                bias_implementation="ordinary",
            ).eval()
            ordinary_next_random = torch.rand(8)
            torch.manual_seed(42)
            array = MBAN(
                num_channels=8,
                hidden_width=4,
                hidden_layers=2,
                output_controls=3,
                bias_layers=["fc1", "fc2", "fc3"],
                bias_implementation="array",
            ).eval()
            array_next_random = torch.rand(8)
            for name in ("fc1", "fc2", "fc3"):
                ordinary_layer = getattr(ordinary, name)
                array_layer = getattr(array, name)
                torch.testing.assert_close(ordinary_layer.weight, array_layer.weight[:, :-1])
                torch.testing.assert_close(ordinary_layer.bias, array_layer.weight[:, -1])
            torch.testing.assert_close(ordinary_next_random, array_next_random)
            rf_i = torch.randn(16, 8)
            rf_q = torch.randn_like(rf_i)
            torch.testing.assert_close(ordinary(rf_i, rf_q), array(rf_i, rf_q), atol=3.0e-6, rtol=1.0e-6)
        finally:
            model_module.set_runtime_args(previous_runtime_args)

    def test_array_output_bias_is_zero_for_free_signed_output(self) -> None:
        previous_runtime_args = model_module._runtime_args
        model_module.set_runtime_args(
            SimpleNamespace(
                qat_enabled=False,
                output_weights="real",
                unity_constraint="free",
                output_domain="signed",
            )
        )
        try:
            array = MBAN(
                num_channels=2,
                hidden_width=4,
                hidden_layers=1,
                output_controls=1,
                bias_layers=["fc1", "fc2"],
                bias_implementation="array",
            )
            self.assertTrue(torch.equal(array.fc2.weight[:, -1], torch.zeros_like(array.fc2.weight[:, -1])))
        finally:
            model_module.set_runtime_args(previous_runtime_args)

    def test_factorized_weight_evaluation_is_rejected(self) -> None:
        factorized = ModelSpec(
            "F",
            Path("f.pth"),
            object(),
            {"beamforming_implementation": "factorized"},
            object(),
            "fp32",
        )
        explicit = ModelSpec(
            "E",
            Path("e.pth"),
            object(),
            {"beamforming_implementation": "explicit"},
            object(),
            "fp32",
        )
        with self.assertRaisesRegex(ValueError, "F/factorized"):
            require_explicit_weight_evaluation([factorized], "权重统计")
        require_explicit_weight_evaluation([explicit], "权重统计")

    def test_weight_path_collector_preserves_indices_and_final_normalization(self) -> None:
        collector = WeightPathCollector(sample_limit=32)
        raw = torch.tensor([[[0.0, 10.0, 20.0]]])
        controls = torch.tensor([[[0.0, 1.0, 2.0]]])
        weights = torch.tensor([[[1.0, 2.0, 1.0]]])

        collector.add(raw, controls, weights, None)

        rows = {(row["stage"], row["component"], row["index"]): row for row in collector.rows()}
        self.assertEqual(rows["raw_control", "real", 1]["mean"], 10.0)
        self.assertEqual(rows["adc_control", "real", 2]["mean"], 2.0)
        self.assertAlmostEqual(rows["physical_weight", "real", 0]["mean"], 0.25)
        self.assertAlmostEqual(rows["physical_weight", "real", 1]["mean"], 0.5)

    def test_execution_implementation_can_override_checkpoint_metadata(self) -> None:
        stored = {"beamforming_implementation": "explicit"}
        self.assertEqual(
            resolve_execution_implementation(stored, SimpleNamespace(beamforming_implementation="factorized")),
            "factorized",
        )
        self.assertEqual(resolve_execution_implementation(stored, SimpleNamespace()), "explicit")
        config_overrides, implementation = split_execution_overrides(
            ["output_interpolation=nearest", "beamforming_implementation=factorized"]
        )
        self.assertEqual(config_overrides, ["output_interpolation=nearest"])
        self.assertEqual(implementation, "factorized")

    def test_model_config_requires_display_name_and_model_type(self) -> None:
        self.assertEqual(parse_model_value("QAT=q.pth", "qat").model_type, "qat")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.yaml"
            path.write_text(
                "models:\n"
                "  - display_name: FP32\n"
                "    checkpoint: fp32/best_val.pth\n"
                "    model_type: fp32\n"
                "  - display_name: PTQ4\n"
                "    checkpoint: fp32/best_val.pth\n"
                "    model_type: ptq\n"
                "    overrides:\n"
                "      control_bits: 4\n",
                encoding="utf-8",
            )
            requests = load_model_requests(path)
            self.assertEqual(requests[0].model_type, "fp32")
            self.assertEqual(requests[1].model_type, "ptq")
            self.assertEqual(requests[1].overrides, ("control_bits=4",))

    def test_fixed_test_selection_is_deterministic_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h5_path = root / "dataset.h5"
            split_path = root / "split.csv"
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("all_multi_I", shape=(6, 1, 2, 1), dtype="float32")
            with split_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["split", "original_index", "source_h5"])
                writer.writeheader()
                writer.writerows(
                    (
                        {"split": "test", "original_index": 4, "source_h5": str(h5_path)},
                        {"split": "test", "original_index": 1, "source_h5": str(h5_path)},
                        {"split": "test", "original_index": 3, "source_h5": str(h5_path)},
                    )
                )
            indices = read_test_indices(split_path, h5_path)
            self.assertEqual(indices, [1, 3, 4])
            self.assertEqual(select_test_indices(indices, 2), [1, 4])
            self.assertEqual(select_test_indices(indices, 3), indices)
            with self.assertRaises(ValueError):
                select_test_indices(indices, 4)
            with split_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["split", "original_index", "source_h5"])
                writer.writeheader()
                writer.writerows(
                    (
                        {"split": "test", "original_index": 1, "source_h5": str(h5_path)},
                        {"split": "test", "original_index": 1, "source_h5": str(h5_path)},
                    )
                )
            with self.assertRaises(ValueError):
                read_test_indices(split_path, h5_path)

    def test_standard_baselines_evaluate_their_own_images(self) -> None:
        from eval_core.diagnostics import _evaluate_standard_baselines
        from eval_core.inference import SceneData

        data = SceneData(
            Path("scene.h5"),
            0,
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            1.0,
            1.0,
            1.0,
            1.0,
            np.zeros((4, 4), dtype=np.float32),
        )
        das = np.zeros((4, 4), dtype=np.float32)
        mv = np.full((4, 4), -60.0, dtype=np.float32)
        with patch("eval_core.diagnostics.standard_baselines", return_value=(das, mv)):
            (das_db, das_env), (mv_db, mv_env) = _evaluate_standard_baselines(data, Path("."), 1.5)
        self.assertAlmostEqual(das_db, 1.0)
        self.assertAlmostEqual(das_env, 1.0)
        self.assertLess(mv_db, das_db)
        self.assertLess(mv_env, das_env)

    def test_bootstrap_rejects_nonpositive_sample_count(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_mean_ci(np.array([1.0, 2.0]), 0, samples=0)

    def test_hardware_scenarios_use_common_baseline(self) -> None:
        config_path = TRAIN_DIR / "config.yaml"
        hardware_path = config_path.with_name("hardware_eval.yaml")
        common = load_nonideal_profile(str(config_path), "common", str(hardware_path))
        self.assertIn("write_error_std=0.05", common)
        self.assertFalse(any(item.startswith("inter_layer=") for item in common))
        analog_config = load_config(
            str(config_path),
            hardware_config_path=str(hardware_path),
            profile_name="common",
            overrides=["inter_layer=analog"],
        )
        digital_config = load_config(
            str(config_path),
            hardware_config_path=str(hardware_path),
            profile_name="common",
            overrides=["inter_layer=digital"],
        )
        none_config = load_config(
            str(config_path),
            hardware_config_path=str(hardware_path),
            profile_name="common",
            overrides=["inter_layer=none"],
        )
        self.assertEqual(analog_config.inter_layer, "analog")
        self.assertEqual(digital_config.inter_layer, "digital")
        self.assertEqual(none_config.inter_layer, "none")
        with self.assertRaises(ValueError):
            load_nonideal_profile(str(config_path), "analog", str(hardware_path))

    def test_common_profile_selects_inter_layer_and_bias_pressure(self) -> None:
        profile_config = {
            "profiles": {
                "ideal": {"parameters": {}},
                "common": {
                    "parameters": {
                        "activation_threshold": 0.4,
                        "adc_gain_error": 0.3,
                        "bias_noise_std": 0.2,
                        "static_mismatch_std": {
                            "activation_threshold": 0.1,
                            "adc_gain_error": 0.2,
                            "bias_programming_offset": 0.3,
                        },
                    }
                },
            }
        }

        def values(inter_layer: str, bias_implementation: str) -> dict[str, str]:
            return {
                key: raw
                for item in resolve_nonideal_profile_raw(
                    profile_config,
                    "common",
                    inter_layer,
                    bias_implementation,
                )
                for key, _, raw in (item.partition("="),)
            }

        analog = values("analog", "ordinary")
        self.assertEqual(analog["activation_threshold"], "0.4")
        self.assertEqual(analog["adc_gain_error"], "0.3")
        self.assertEqual(analog["bias_noise_std"], "0.2")
        self.assertIn("activation_threshold", analog["static_mismatch_std"])
        self.assertIn("adc_gain_error", analog["static_mismatch_std"])

        digital = values("digital", "ordinary")
        self.assertEqual(digital["activation_threshold"], "0.0")
        self.assertEqual(digital["adc_gain_error"], "0.3")
        self.assertEqual(digital["bias_noise_std"], "0.0")
        self.assertNotIn("bias_programming_offset", digital["static_mismatch_std"])

        none = values("none", "ordinary")
        self.assertEqual(none["activation_threshold"], "0.0")
        self.assertEqual(none["adc_gain_error"], "0.3")
        self.assertEqual(none["bias_noise_std"], "0.0")
        self.assertNotIn("bias_programming_offset", none["static_mismatch_std"])

        array = values("analog", "array")
        self.assertEqual(array["bias_noise_std"], "0.0")
        self.assertNotIn("bias_programming_offset", array["static_mismatch_std"])

    def test_digital_bias_ignores_analog_bias_nonidealities(self) -> None:
        hardware = HardwareQAT(
            QATConfig(
                enabled=True,
                bias_programming_gain_error=0.2,
                bias_programming_offset=0.3,
                bias_noise_std=0.4,
                bias_channel_gain_mismatch_std=0.1,
                noise_enabled=True,
                static_mismatch_std={"bias_programming_offset": 0.5},
            )
        )
        bias = torch.tensor([0.2, -0.4, 0.8])
        reference = HardwareQAT(QATConfig(enabled=True)).bias(bias, "fc1", "digital")
        torch.testing.assert_close(hardware.bias(bias, "fc1", "digital"), reference)

    def test_array_constant_line_uses_independent_full_scale_dac(self) -> None:
        hardware = HardwareQAT(QATConfig(enabled=True, weight_bits=32, input_bits=4))
        layer = MemristorLinear(3, 2, False, hardware, "fc1", "array")
        signal_calls = []
        bias_calls = []

        def capture_signal(value, name, bits):
            signal_calls.append((value.detach().clone(), name, bits))
            return value

        def capture_bias(value, name, bits, unsigned, value_max=None):
            bias_calls.append((value.detach().clone(), name, bits, unsigned, value_max.detach().clone()))
            return value

        with patch.object(hardware, "dac", side_effect=capture_signal), patch.object(
            hardware, "digital_dac", side_effect=capture_bias
        ):
            layer(torch.tensor([[0.2, -0.3, 1.0]]))
            layer(torch.tensor([[0.2, -0.3, 1.0]]), quantize_input=False)
            layer.forward_tiled(
                torch.tensor([[0.2, -0.3, 1.0]]),
                tile=2,
                effective_weight=layer.weight,
            )

        self.assertEqual(len(signal_calls), 2)
        self.assertEqual(len(bias_calls), 3)
        for value, name, bits in signal_calls:
            torch.testing.assert_close(value, torch.tensor([[0.2, -0.3]]))
            self.assertEqual(name, "fc1.input_dac")
            self.assertEqual(bits, 4)
        for value, name, bits, unsigned, value_max in bias_calls:
            torch.testing.assert_close(value, torch.tensor([[1.0]]))
            self.assertEqual(name, "fc1.bias_dac")
            self.assertEqual(bits, 4)
            self.assertTrue(unsigned)
            torch.testing.assert_close(value_max, torch.tensor(1.0))

    def test_all_bias_interlayer_paths_smoke(self) -> None:
        previous_runtime_args = model_module._runtime_args
        model_module.set_runtime_args(runtime.args)
        try:
            runtime.args.output_domain = "nonnegative"
            runtime.args.unity_constraint = "free"
            for bias_implementation in ("ordinary", "array"):
                for inter_layer in ("none", "analog", "digital"):
                    for hidden_layers in (2, 4, 6):
                        layers = [f"fc{index}" for index in range(1, hidden_layers + 2)]
                        activation = {
                            name: {"type": "relu"}
                            for name in layers[:-1]
                        }
                        model = MBAN(
                            num_channels=4,
                            hidden_width=3,
                            hidden_layers=hidden_layers,
                            output_controls=4,
                            activation=activation,
                            bias_layers=layers,
                            bias_implementation=bias_implementation,
                            hardware_config=QATConfig(
                                enabled=True,
                                inter_layer=inter_layer,
                                bias_implementation=bias_implementation,
                                observer_enabled=True,
                            ),
                            hardware_enabled=True,
                        ).eval()
                        output = model(torch.randn(2, 4), torch.randn(2, 4))
                        if inter_layer == "digital":
                            output = model.prepare_output_controls(output)
                        output = model.quantize_controls(output)
                        self.assertEqual(output.shape, (2, 4), (bias_implementation, inter_layer, hidden_layers))
                        self.assertTrue(torch.isfinite(output).all(), (bias_implementation, inter_layer, hidden_layers))
                        if inter_layer == "digital":
                            self.assertIn(
                                model.output_preactivation_range_name,
                                model.hardware_qat.activation_ranges,
                            )
                            self.assertNotIn(model.output_control_range_name, model.hardware_qat.activation_ranges)
                        else:
                            self.assertIn(model.output_control_range_name, model.hardware_qat.activation_ranges)
        finally:
            model_module.set_runtime_args(previous_runtime_args)

    def test_digital_output_bias_is_after_signed_adc(self) -> None:
        previous_runtime_args = model_module._runtime_args
        model_module.set_runtime_args(runtime.args)
        try:
            runtime.args.output_domain = "nonnegative"
            runtime.args.unity_constraint = "free"
            model = MBAN(
                num_channels=2,
                hidden_width=2,
                hidden_layers=1,
                output_controls=2,
                bias_layers=["fc2"],
                bias_implementation="ordinary",
                hardware_config=QATConfig(
                    enabled=True,
                    inter_layer="digital",
                    control_bits=4,
                    bias_bits=4,
                    observer_enabled=True,
                ),
                hardware_enabled=True,
            )
            with torch.no_grad():
                model.fc2.bias.copy_(torch.tensor([0.4, 0.0]))
            prepared = model.prepare_output_controls(torch.tensor([[-0.3, 0.1]]))
            torch.testing.assert_close(prepared, torch.tensor([[1.0 / 7.0, 1.0 / 7.0]]))
            torch.testing.assert_close(model.quantize_controls(prepared), prepared)
        finally:
            model_module.set_runtime_args(previous_runtime_args)

    def test_array_digital_output_uses_unified_unsigned_codes(self) -> None:
        previous_runtime_args = model_module._runtime_args
        model_module.set_runtime_args(runtime.args)
        try:
            runtime.args.output_domain = "nonnegative"
            runtime.args.unity_constraint = "free"
            model = MBAN(
                num_channels=2,
                hidden_width=2,
                hidden_layers=1,
                output_controls=2,
                bias_layers=["fc2"],
                bias_implementation="array",
                hardware_config=QATConfig(
                    enabled=True,
                    inter_layer="digital",
                    control_bits=4,
                    observer_enabled=False,
                ),
                hardware_enabled=True,
            )
            model.hardware_qat.activation_ranges[model.output_preactivation_range_name] = torch.tensor(1.0)
            prepared = model.prepare_output_controls(torch.tensor([[0.2, 0.05]]))
            quantized = model.quantize_controls(prepared)
            torch.testing.assert_close(prepared, torch.tensor([[1.0 / 7.0, 0.0]]))
            torch.testing.assert_close(quantized, prepared)
        finally:
            model_module.set_runtime_args(previous_runtime_args)

    def test_digital_output_keeps_signed_adc_scale(self) -> None:
        previous_runtime_args = model_module._runtime_args
        model_module.set_runtime_args(runtime.args)
        try:
            runtime.args.output_domain = "nonnegative"
            runtime.args.unity_constraint = "free"
            model = MBAN(
                num_channels=2,
                hidden_width=2,
                hidden_layers=1,
                output_controls=2,
                bias_layers=["fc2"],
                bias_implementation="array",
                hardware_config=QATConfig(enabled=True, inter_layer="digital", control_bits=4),
                hardware_enabled=True,
            )
            model.hardware_qat.activation_ranges[model.output_preactivation_range_name] = torch.tensor(1.0)
            prepared = model.prepare_output_controls(torch.tensor([[0.2, 0.05]]))
            torch.testing.assert_close(model.quantize_controls(prepared), prepared)
        finally:
            model_module.set_runtime_args(previous_runtime_args)

    def test_signed_digital_output_skips_relu_and_second_quantization(self) -> None:
        previous_runtime_args = model_module._runtime_args
        model_module.set_runtime_args(runtime.args)
        try:
            runtime.args.output_domain = "signed"
            runtime.args.unity_constraint = "free"
            model = MBAN(
                num_channels=2,
                hidden_width=2,
                hidden_layers=1,
                output_controls=2,
                bias_layers=["fc2"],
                bias_implementation="array",
                hardware_config=QATConfig(enabled=True, inter_layer="digital", control_bits=4),
                hardware_enabled=True,
            )
            model.hardware_qat.activation_ranges[model.output_preactivation_range_name] = torch.tensor(1.0)
            prepared = model.prepare_output_controls(torch.tensor([[-0.2, 0.2]]))
            torch.testing.assert_close(prepared, torch.tensor([[-1.0 / 7.0, 1.0 / 7.0]]))
            torch.testing.assert_close(model.quantize_controls(prepared), prepared)
        finally:
            model_module.set_runtime_args(previous_runtime_args)

    def test_array_bias_uses_frozen_bias_range_before_mapping(self) -> None:
        hardware = HardwareQAT(
            QATConfig(enabled=True, weight_bits=4, bias_bits=2, input_bits=32, noise_enabled=False)
        )
        hardware.activation_ranges["fc1.bias"] = torch.tensor(1.0)
        hardware.activation_ranges["fc1.ordinary.bias"] = torch.tensor(1.0)
        hardware.disable_observer()
        layer = MemristorLinear(3, 2, False, hardware, "fc1", "array")
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[0.2, -0.1, 2.0], [0.1, 0.2, -2.0]]))

        layer(torch.tensor([[0.2, -0.3, 1.0]]), quantize_input=False)

        self.assertAlmostEqual(float(hardware.weight_scales["fc1"].max()), 1.0, places=6)
        torch.testing.assert_close(
            hardware.bias(torch.tensor([0.8]), "fc1", "array"),
            torch.tensor([6.0 / 7.0]),
        )
        torch.testing.assert_close(
            hardware.bias(torch.tensor([0.8]), "fc1.ordinary", "none"),
            torch.tensor([1.0]),
        )

    def test_ir_drop_path_keeps_finite_float32_gradient(self) -> None:
        hardware = HardwareQAT(
            QATConfig(
                enabled=True,
                weight_bits=4,
                input_bits=32,
                ir_drop_enabled=True,
                ir_drop_tile_rows=2,
                ir_drop_tile_cols=2,
            )
        )
        inputs = torch.randn(3, 4, requires_grad=True)
        weights = torch.randn(2, 4, requires_grad=True)
        state = hardware.map_weight(weights, "fc1")
        output = hardware.crossbar_mvm(inputs, state.effective_weight, conductance_state=state, name="fc1")
        output.sum().backward()

        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.isfinite(weights.grad).all())

    def test_output_digital_bias_uses_output_control_range(self) -> None:
        hardware = HardwareQAT(QATConfig(enabled=True, bias_bits=4))
        control_name = output_preactivation_name("fc3")
        hardware.activation_ranges[control_name] = torch.tensor(1.0)
        bias = torch.tensor([0.125, -0.125, 1.5])
        quantized = hardware.bias(
            bias,
            "fc3",
            "digital",
            quantization_range=control_name,
        )
        expected = torch.tensor([1.0 / 7.0, -1.0 / 7.0, 1.0])
        torch.testing.assert_close(quantized, expected)
        torch.testing.assert_close(hardware.activation_ranges[control_name], torch.tensor(1.0))

    def test_adc_saturation_keeps_qat_gradient(self) -> None:
        hardware = HardwareQAT(QATConfig(enabled=True, adc_nonlinearity=0.0, qat_clip_outside_grad=0.1))
        value = torch.tensor([0.4, 1.01, 1.5, 2.0], requires_grad=True)
        quantized = hardware.adc_unsigned(value, output_controls_name("fc3"), 4, value_max=1.0)
        torch.testing.assert_close(quantized, torch.tensor([6.0 / 15.0, 1.0, 1.0, 1.0]))
        quantized.sum().backward()
        torch.testing.assert_close(value.grad, torch.tensor([1.0, 0.1 / 1.01, 0.1 / 1.5, 0.05]))

    def test_zero_clip_outside_gradient_restores_hard_clamp(self) -> None:
        hardware = HardwareQAT(QATConfig(enabled=True, qat_clip_outside_grad=0.0))
        value = torch.tensor([0.5, 2.0], requires_grad=True)
        hardware.adc_unsigned(value, output_controls_name("fc3"), 4, value_max=1.0).sum().backward()
        torch.testing.assert_close(value.grad, torch.tensor([1.0, 0.0]))

    def test_nonfinite_stats_count_gradients_and_parameters(self) -> None:
        finite = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        nonfinite = torch.nn.Parameter(torch.tensor([1.0, float("nan")]))
        finite.grad = torch.tensor([0.0, float("inf")])
        nonfinite.grad = torch.tensor([0.0, 1.0])

        gradient_count, gradient_total = count_nonfinite_gradients([finite, nonfinite])
        parameter_count, parameter_total = count_nonfinite_parameters([finite, nonfinite])

        self.assertEqual(int(gradient_count.item()), 1)
        self.assertEqual(gradient_total, 4)
        self.assertEqual(int(parameter_count.item()), 1)
        self.assertEqual(parameter_total, 4)

    def test_none_bias_keeps_quantization_without_analog_nonidealities(self) -> None:
        hardware = HardwareQAT(
            QATConfig(
                enabled=True,
                bias_bits=4,
                bias_programming_gain_error=0.2,
                bias_programming_offset=0.3,
                bias_noise_std=0.4,
                bias_channel_gain_mismatch_std=0.1,
                noise_enabled=True,
            )
        )
        bias = torch.tensor([0.2, -0.4, 0.8])
        reference = HardwareQAT(QATConfig(enabled=True, bias_bits=4)).bias(bias, "fc1.bias", "none")
        torch.testing.assert_close(hardware.bias(bias, "fc1.bias", "none"), reference)

    def test_single_analog_activation_uses_gain_and_threshold(self) -> None:
        hardware = HardwareQAT(
            QATConfig(
                enabled=True,
                activation_gain_mismatch=0.1,
                activation_threshold=0.2,
                activation_threshold_mismatch=0.05,
                noise_enabled=False,
            )
        )
        values = torch.tensor([-0.5, 0.0, 0.5])
        expected = (values - 0.25) * 1.1
        torch.testing.assert_close(hardware.activation_mismatch(values, "fc1"), expected)

    def test_analog_output_relu_uses_single_channel_parameters(self) -> None:
        hardware = HardwareQAT(
            QATConfig(
                enabled=True,
                activation_gain_mismatch=0.1,
                activation_threshold=0.2,
                activation_threshold_mismatch=0.05,
                noise_enabled=False,
            )
        )
        values = torch.tensor([[-0.5, 0.5, 0.8]])
        expected = torch.relu((values - 0.25) * 1.1)
        torch.testing.assert_close(hardware.output_activation_mismatch(values, "fc3", reference=1.0), expected)

    def test_analog_output_relu_scales_full_scale_threshold(self) -> None:
        hardware = HardwareQAT(
            QATConfig(
                enabled=True,
                activation_threshold=0.2,
                activation_threshold_mode="full_scale",
                activation_threshold_mismatch=0.05,
                activation_threshold_mismatch_mode="full_scale",
                noise_enabled=False,
            )
        )
        values = torch.tensor([[0.4, 0.8]])
        expected = torch.tensor([[0.0, 0.3]])
        torch.testing.assert_close(hardware.output_activation_mismatch(values, "fc3", reference=2.0), expected)

    def test_analog_output_relu_channel_mismatch_is_fixed_per_realization(self) -> None:
        hardware = HardwareQAT(
            QATConfig(
                enabled=True,
                noise_enabled=True,
                noise_seed=31,
                static_mismatch_std={
                    "activation_gain_mismatch": 0.1,
                    "activation_threshold_mismatch": 0.1,
                },
            )
        )
        values = torch.ones(2, 4)
        hardware.begin_noise_realization()
        first = hardware.output_activation_mismatch(values, "fc3")
        second = hardware.output_activation_mismatch(values, "fc3")
        torch.testing.assert_close(first, second)
        first_parameters = hardware._channel_mismatch_cache[("activation_gain_mismatch", "fc3.output_relu")]
        self.assertEqual(first_parameters.numel(), 4)
        self.assertFalse(torch.equal(first_parameters[0:1], first_parameters[1:2]))
        hardware.begin_noise_realization()
        third = hardware.output_activation_mismatch(values, "fc3")
        self.assertFalse(torch.equal(first, third))

    def test_analog_output_relu_precedes_unsigned_output_adc(self) -> None:
        previous_runtime_args = model_module._runtime_args
        model_module.set_runtime_args(runtime.args)
        try:
            runtime.args.output_domain = "nonnegative"
            runtime.args.unity_constraint = "free"
            model = MBAN(
                num_channels=2,
                hidden_width=2,
                hidden_layers=1,
                output_controls=2,
                bias_layers=[],
                bias_implementation="ordinary",
                hardware_config=QATConfig(
                    enabled=True,
                    inter_layer="analog",
                    control_bits=4,
                    activation_threshold=0.2,
                    observer_enabled=False,
                ),
                hardware_enabled=True,
            )
            controls = torch.tensor([[-0.1, 0.6]])
            collector = TraceCollector(model.hardware_qat)
            with HardwareTrace(model, collector):
                collector.begin_run("smoke", 1, 1)
                prepared = model.prepare_output_controls(controls)
            torch.testing.assert_close(prepared, torch.tensor([[0.0, 0.4]]))
            self.assertEqual(collector.calls.get("output_activation_mismatch"), 1)
            self.assertIn("fc2.output_relu.input", collector.signals)
            quantized = model.quantize_controls(prepared)
            torch.testing.assert_close(quantized, torch.tensor([[0.0, 6.0 / 15.0]]))
        finally:
            model_module.set_runtime_args(previous_runtime_args)

    def test_digital_output_does_not_use_analog_output_relu_mismatch(self) -> None:
        previous_runtime_args = model_module._runtime_args
        model_module.set_runtime_args(runtime.args)
        try:
            runtime.args.output_domain = "nonnegative"
            runtime.args.unity_constraint = "free"
            model = MBAN(
                num_channels=2,
                hidden_width=2,
                hidden_layers=1,
                output_controls=2,
                bias_layers=[],
                bias_implementation="array",
                hardware_config=QATConfig(
                    enabled=True,
                    inter_layer="digital",
                    control_bits=4,
                    activation_gain_mismatch=0.5,
                    activation_threshold=0.5,
                    observer_enabled=False,
                ),
                hardware_enabled=True,
            )
            prepared = model.prepare_output_controls(torch.tensor([[-0.2, 0.4]]))
            torch.testing.assert_close(prepared, torch.tensor([[0.0, 3.0 / 7.0]]))
        finally:
            model_module.set_runtime_args(previous_runtime_args)

    def test_network_channel_geometry_is_not_silently_clipped(self) -> None:
        depths = torch.tensor([0.005, 0.010, 0.015])
        network, aperture_sizes = derive_network_channels(depths, 0.0005, 1.0, 40, expected=30)
        self.assertEqual(network, 30)
        self.assertEqual(int(aperture_sizes.max()), 30)
        with self.assertRaises(ValueError):
            derive_network_channels(depths, 0.0005, 1.0, 40, expected=29)
        with self.assertRaises(ValueError):
            derive_network_channels(torch.tensor([0.025]), 0.0005, 1.0, 40)

    def test_static_mismatch_is_fixed_inside_realization(self) -> None:
        config = QATConfig(
            enabled=True,
            noise_enabled=True,
            static_mismatch_std={"control_reference_gain_error": 0.1},
            noise_seed=17,
        )
        hardware = HardwareQAT(config)
        hardware.begin_noise_realization()
        maximum = torch.tensor([[0.5]])
        control_name = output_controls_name("fc3")
        first = hardware.control_reference(maximum, control_name)
        second = hardware.control_reference(maximum, control_name)
        torch.testing.assert_close(first, second)
        hardware.begin_noise_realization()
        third = hardware.control_reference(maximum, control_name)
        self.assertFalse(torch.equal(first, third))

    def test_dynamic_reference_noise_changes_inside_realization(self) -> None:
        config = QATConfig(
            enabled=True,
            noise_enabled=True,
            control_reference_noise_std=0.1,
            noise_seed=23,
        )
        hardware = HardwareQAT(config)
        hardware.begin_noise_realization()
        maximum = torch.tensor([[0.5]])
        control_name = output_controls_name("fc3")
        first = hardware.control_reference(maximum, control_name)
        second = hardware.control_reference(maximum, control_name)
        self.assertFalse(torch.equal(first, second))

    def test_dynamic_reference_noise_replays_after_realization_restore(self) -> None:
        config = QATConfig(
            enabled=True,
            noise_enabled=True,
            control_reference_noise_std=0.1,
            noise_seed=29,
        )
        hardware = HardwareQAT(config)
        hardware.begin_noise_realization()
        state = hardware.realization_state()
        maximum = torch.tensor([[0.5]])
        control_name = output_controls_name("fc3")
        first = hardware.control_reference(maximum, control_name)
        state_after_first = hardware.realization_state()
        second = hardware.control_reference(maximum, control_name)
        hardware.restore_realization_state(state)
        replay_first = hardware.control_reference(maximum, control_name)
        replay_second = hardware.control_reference(maximum, control_name)
        torch.testing.assert_close(replay_first, first)
        torch.testing.assert_close(replay_second, second)
        self.assertFalse(torch.equal(first, second))
        hardware.restore_realization_state(state_after_first)
        replay_second_only = hardware.control_reference(maximum, control_name)
        torch.testing.assert_close(replay_second_only, second)

    def test_hardware_state_dict_restores_realization(self) -> None:
        config = QATConfig(
            enabled=True,
            noise_enabled=True,
            control_reference_noise_std=0.1,
            noise_seed=37,
        )
        hardware = HardwareQAT(config)
        hardware.begin_noise_realization()
        restored = HardwareQAT(config)
        restored.load_state_dict(hardware.checkpoint_state_dict())
        maximum = torch.tensor([[0.5]])
        control_name = output_controls_name("fc3")
        torch.testing.assert_close(
            hardware.control_reference(maximum, control_name),
            restored.control_reference(maximum, control_name),
        )

    def test_hardware_state_dict_restores_runtime_switches(self) -> None:
        hardware = HardwareQAT(QATConfig(enabled=True, noise_enabled=True))
        hardware.disable_observer()
        hardware.disable_noise()
        restored = HardwareQAT(hardware.config)
        restored.load_state_dict(hardware.checkpoint_state_dict())
        self.assertFalse(restored.observer_enabled)
        self.assertFalse(restored._noise_enabled)

    def test_tia_saturation_diagnostic_uses_pre_saturation_signal(self) -> None:
        hardware = HardwareQAT(QATConfig(enabled=True, tia_saturation=1.0, tia_saturation_mode="absolute"))
        hardware.begin_diagnostics()
        common_mode = torch.full((1, 2), 0.15)
        hardware.tia(torch.tensor([[0.75, -0.75]]), name="fc1.tia", common_mode=common_mode)
        diagnostics = hardware.end_diagnostics(16.0)
        self.assertGreater(diagnostics["quantizer_saturation_fraction"]["fc1.tia"], 0.0)

    def test_reference_path_has_finite_nonzero_gradient(self) -> None:
        controls = torch.tensor([[[0.2, 0.4, 0.1, 0.3]]], requires_grad=True)
        normalized = beamforming.normalize_controls_for_adc(controls)
        loss = normalized.square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(controls.grad).all())
        self.assertGreater(float(controls.grad.abs().sum()), 0.0)

    def test_linf_normalization_divides_in_fp32_for_half_inputs(self) -> None:
        controls = torch.tensor([[[0.25, 0.5, 0.125, 0.0]]], dtype=torch.float16)
        normalized = beamforming.normalize_controls_for_adc(controls)
        self.assertEqual(normalized.dtype, controls.dtype)
        self.assertTrue(torch.isfinite(normalized).all())
        torch.testing.assert_close(normalized.amax(dim=-1), torch.ones((1, 1), dtype=torch.float16))

    def test_linf_reference_errors_are_bypassed_by_none_mode(self) -> None:
        hardware = HardwareQAT(
            QATConfig(
                enabled=True,
                control_reference_gain_error=0.1,
                control_reference_offset=0.2,
            )
        )
        controls = torch.tensor([[[0.2, 0.5]]])
        control_name = output_controls_name("fc3")
        runtime.args.control_normalization = "linf"
        normalized = beamforming.normalize_controls_for_adc(
            controls, hardware=hardware, control_range_name=control_name
        )
        torch.testing.assert_close(normalized, controls / 0.65)
        runtime.args.control_normalization = "none"
        bypassed = beamforming.normalize_controls_for_adc(
            controls, hardware=hardware, control_range_name=control_name
        )
        torch.testing.assert_close(bypassed, controls)

    def test_signed_linf_normalization_uses_max_abs(self) -> None:
        runtime.args.output_domain = "signed"
        controls = torch.tensor([[[0.2, -0.6, 0.1]]])
        normalized = beamforming.normalize_controls_for_adc(controls)
        torch.testing.assert_close(normalized, torch.tensor([[[1.0 / 3.0, -1.0, 1.0 / 6.0]]]))

    def test_signed_control_diagnostics_use_symmetric_codes(self) -> None:
        hardware = HardwareQAT(QATConfig(enabled=True, control_bits=3))
        control_name = output_controls_name("fc3")
        hardware.activation_ranges[control_name] = torch.tensor(1.0)
        hardware.begin_diagnostics()
        values = torch.tensor([[[-1.0, -0.2, 0.0, 0.2, 1.1]]])
        hardware.record_controls(values, values, control_name, 3, unsigned=False)
        diagnostics = hardware.end_diagnostics(16.0)
        self.assertEqual(diagnostics["used_control_codes"], [-3, -1, 0, 1, 3])
        self.assertAlmostEqual(diagnostics["zero_control_code_fraction"], 0.2)
        self.assertAlmostEqual(diagnostics["endpoint_control_code_fraction"], 0.4)
        self.assertAlmostEqual(diagnostics["saturated_control_code_fraction"], 0.2)

    def test_signed_adc_uses_symmetric_full_scale(self) -> None:
        hardware = HardwareQAT(QATConfig(enabled=True, control_bits=3))
        values = torch.tensor([[[-1.0, -0.5, 0.0, 0.5, 1.0]]])
        quantized = hardware.adc_signed(values, output_controls_name("fc3"), 3, value_max=1.0)
        expected = torch.tensor([[[-1.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, 1.0]]])
        torch.testing.assert_close(quantized, expected)

    def test_zero_controls_use_representable_active_code(self) -> None:
        controls = torch.zeros((1, 1, 4))
        active_mask = torch.tensor([[True, True, False, False]])
        normalized = beamforming.normalize_controls_for_adc(controls, active_mask)
        expected = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
        torch.testing.assert_close(normalized, expected)

    def test_signed_zero_controls_use_active_uniform_fallback(self) -> None:
        runtime.args.output_domain = "signed"
        controls = torch.zeros((1, 1, 4))
        active_mask = torch.tensor([[True, True, False, False]])
        hardware = HardwareQAT(QATConfig(enabled=True, control_bits=4))
        normalized = beamforming.normalize_controls_for_adc(
            controls,
            active_mask,
            hardware,
            output_controls_name("fc3"),
        )
        expected = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
        torch.testing.assert_close(normalized, expected)
        self.assertFalse(bool(hardware.adaptive_valid.all()))

    def test_valid_controls_set_adaptive_valid(self) -> None:
        controls = torch.tensor([[[0.25, 0.5, 0.0, 0.0]]])
        active_mask = torch.tensor([[True, True, False, False]])
        hardware = HardwareQAT(QATConfig(enabled=True, control_bits=4))
        beamforming.normalize_controls_for_adc(
            controls,
            active_mask,
            hardware,
            output_controls_name("fc3"),
        )
        self.assertTrue(bool(hardware.adaptive_valid.all()))

    def test_control_endpoint_is_not_reported_as_overrange(self) -> None:
        hardware = HardwareQAT(QATConfig(enabled=True, control_bits=4))
        control_name = output_controls_name("fc3")
        hardware.activation_ranges[control_name] = torch.tensor(1.0)
        hardware.begin_diagnostics()
        values = torch.tensor([[[1.0, 1.1, 0.2, 0.0]]])
        hardware.record_controls(values, values, control_name, 4)
        diagnostics = hardware.end_diagnostics(16.0)
        self.assertAlmostEqual(diagnostics["endpoint_control_code_fraction"], 0.5)
        self.assertAlmostEqual(diagnostics["saturated_control_code_fraction"], 0.25)

    def test_factorized_matches_explicit_fixed_with_gradient(self) -> None:
        controls = torch.tensor([[[0.2, 0.6, 0.1], [0.5, 0.3, 0.7]]])
        iq_i = torch.tensor([[[0.4, -0.2, 0.7, 0.1, -0.3], [0.6, 0.2, -0.5, 0.3, 0.8]]])
        iq_q = torch.tensor([[[0.1, 0.5, -0.4, 0.9, 0.2], [-0.1, 0.7, 0.6, -0.2, 0.3]]])
        tx_tof = torch.zeros((1, 2))
        input_scale = torch.ones((1, 1, 1))
        input_mean = torch.zeros((1, 1, 1))

        for interpolation in ("linear", "nearest", "cubic"):
            runtime.args.output_interpolation = interpolation
            explicit_controls = controls.detach().requires_grad_()
            explicit_weights = beamforming.expand_fixed_output_controls(explicit_controls, 5)
            explicit = beamforming.beamform_iq_with_tx_phase(
                explicit_weights,
                iq_i,
                iq_q,
                tx_tof,
                input_scale,
                input_mean,
                controls=explicit_controls,
            )
            (explicit[0] + explicit[1]).sum().backward()

            runtime.args.beamforming_implementation = "factorized"
            factorized_controls = controls.detach().requires_grad_()
            factorized_weights = beamforming.expand_fixed_output_controls(factorized_controls, 5)
            factorized = beamforming.beamform_iq_with_tx_phase(
                factorized_weights,
                iq_i,
                iq_q,
                tx_tof,
                input_scale,
                input_mean,
                controls=factorized_controls,
            )
            (factorized[0] + factorized[1]).sum().backward()

            torch.testing.assert_close(factorized[0], explicit[0], rtol=1.0e-6, atol=1.0e-6)
            torch.testing.assert_close(factorized[1], explicit[1], rtol=1.0e-6, atol=1.0e-6)
            torch.testing.assert_close(factorized_controls.grad, explicit_controls.grad, rtol=1.0e-6, atol=1.0e-6)

    def test_factorized_matches_explicit_dynamic_apertures(self) -> None:
        torch.manual_seed(5)
        iq_i = torch.randn((2, 3, 8))
        iq_q = torch.randn((2, 3, 8))
        tx_tof = torch.zeros((2, 3))
        input_scale = torch.ones((2, 1, 1))
        input_mean = torch.zeros((2, 1, 1))
        aperture_start = torch.tensor([0, 2])
        aperture_size = torch.tensor([1, 4])
        for interpolation in ("linear", "nearest", "cubic"):
            runtime.args.output_interpolation = interpolation
            for control_count in (1, 2, 3, 8):
                controls = torch.rand((2, 3, control_count))
                active = beamforming.dynamic_control_active_mask(aperture_size, control_count, interpolation)
                controls = controls * active.unsqueeze(1).to(controls.dtype)
                weights = beamforming.unpack_dynamic_weights(
                    beamforming.expand_dynamic_output_controls(controls, aperture_size, 8),
                    aperture_start,
                    aperture_size,
                    8,
                )
                runtime.args.beamforming_implementation = "explicit"
                explicit = beamforming.beamform_iq_with_tx_phase(
                    weights,
                    iq_i,
                    iq_q,
                    tx_tof,
                    input_scale,
                    input_mean,
                    controls=controls,
                    aperture_start=aperture_start,
                    aperture_size=aperture_size,
                    network_channels=8,
                )
                runtime.args.beamforming_implementation = "factorized"
                factorized = beamforming.beamform_iq_with_tx_phase(
                    weights,
                    iq_i,
                    iq_q,
                    tx_tof,
                    input_scale,
                    input_mean,
                    controls=controls,
                    aperture_start=aperture_start,
                    aperture_size=aperture_size,
                    network_channels=8,
                )
                torch.testing.assert_close(factorized[0], explicit[0], rtol=1.0e-6, atol=1.0e-6)
                torch.testing.assert_close(factorized[1], explicit[1], rtol=1.0e-6, atol=1.0e-6)

    def test_factorized_dynamic_controls_have_matching_gradients(self) -> None:
        torch.manual_seed(31)
        iq_i = torch.randn((2, 2, 8))
        iq_q = torch.randn_like(iq_i)
        tx_tof = torch.zeros((2, 2))
        input_scale = torch.ones((2, 1, 1))
        input_mean = torch.zeros_like(input_scale)
        aperture_start = torch.tensor([0, 2])
        aperture_size = torch.tensor([1, 4])
        for interpolation in ("linear", "nearest", "cubic"):
            runtime.args.output_interpolation = interpolation
            active = beamforming.dynamic_control_active_mask(aperture_size, 3, interpolation)
            base_controls = torch.rand((2, 2, 3)) * active.unsqueeze(1).to(torch.float32)

            explicit_controls = base_controls.detach().requires_grad_()
            explicit_weights = beamforming.unpack_dynamic_weights(
                beamforming.expand_dynamic_output_controls(explicit_controls, aperture_size, 8),
                aperture_start,
                aperture_size,
                8,
            )
            runtime.args.beamforming_implementation = "explicit"
            explicit = beamforming.beamform_iq_with_tx_phase(
                explicit_weights,
                iq_i,
                iq_q,
                tx_tof,
                input_scale,
                input_mean,
                controls=explicit_controls,
                aperture_start=aperture_start,
                aperture_size=aperture_size,
                network_channels=8,
            )
            (explicit[0] + explicit[1]).sum().backward()

            factorized_controls = base_controls.detach().requires_grad_()
            runtime.args.beamforming_implementation = "factorized"
            factorized = beamforming.beamform_iq_with_tx_phase(
                None,
                iq_i,
                iq_q,
                tx_tof,
                input_scale,
                input_mean,
                controls=factorized_controls,
                aperture_start=aperture_start,
                aperture_size=aperture_size,
                network_channels=8,
            )
            (factorized[0] + factorized[1]).sum().backward()

            torch.testing.assert_close(factorized[0], explicit[0], rtol=1.0e-6, atol=1.0e-6)
            torch.testing.assert_close(factorized[1], explicit[1], rtol=1.0e-6, atol=1.0e-6)
            torch.testing.assert_close(factorized_controls.grad, explicit_controls.grad, rtol=1.0e-6, atol=1.0e-6)

    def test_dynamic_interpolation_cache_matches_direct_path(self) -> None:
        aperture_size = torch.tensor([1, 4, 8])
        for interpolation in ("linear", "nearest", "cubic"):
            runtime.args.output_interpolation = interpolation
            cache = beamforming.build_dynamic_interpolation_cache(8, 3, interpolation, torch.device("cpu"))
            self.assertEqual(cache["dynamic_output_basis_by_size"].dtype, torch.float32)
            self.assertNotIn("dynamic_output_indices_by_size", cache)
            self.assertNotIn("dynamic_output_coefficients_by_size", cache)
            self.assertNotIn("dynamic_output_valid_by_size", cache)
            self.assertFalse(any(key.endswith(("_float16", "_bfloat16")) for key in cache))
            self.assertEqual(
                sum(int(value.numel()) * int(value.element_size()) for value in cache.values()),
                beamforming.dynamic_interpolation_cache_bytes(8, 3, interpolation),
            )
            cached_active = cache["dynamic_control_active_mask_by_size"][aperture_size]
            direct_active = beamforming.dynamic_control_active_mask(aperture_size, 3, interpolation)
            torch.testing.assert_close(cached_active, direct_active)

            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                base_controls = torch.randn((3, 2, 3), dtype=dtype)
                for output_weights in ("real", "complex"):
                    runtime.args.output_weights = output_weights
                    control_values = (
                        base_controls
                        if output_weights == "real"
                        else torch.cat((base_controls, base_controls * 0.7), dim=-1)
                    )
                    direct_controls = control_values.detach().requires_grad_()
                    cached_controls = control_values.detach().requires_grad_()
                    direct = beamforming.expand_dynamic_output_controls(direct_controls, aperture_size, 8)
                    cached = beamforming.expand_dynamic_output_controls(
                        cached_controls,
                        aperture_size,
                        8,
                        interpolation_cache=cache,
                    )
                    output_tolerance = 16.0 * torch.finfo(dtype).eps
                    torch.testing.assert_close(
                        cached,
                        direct,
                        rtol=output_tolerance,
                        atol=output_tolerance,
                    )
                    direct.float().square().sum().backward()
                    cached.float().square().sum().backward()
                    gradient_tolerance = (
                        1.0e-6 if dtype == torch.float32 else 8.0 * torch.finfo(dtype).eps
                    )
                    torch.testing.assert_close(
                        cached_controls.grad,
                        direct_controls.grad,
                        rtol=gradient_tolerance,
                        atol=gradient_tolerance,
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 执行 AMP 插值回归")
    def test_dynamic_interpolation_cache_runs_under_cuda_amp(self) -> None:
        runtime.args.output_interpolation = "linear"
        aperture_size = torch.tensor([1, 4, 8, 6], device="cuda")
        cache = beamforming.build_dynamic_interpolation_cache(8, 3, "linear", torch.device("cuda"))
        direct_controls = torch.randn((4, 2, 3), device="cuda", dtype=torch.float16, requires_grad=True)
        cached_controls = direct_controls.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16):
            direct = beamforming.expand_dynamic_output_controls(direct_controls, aperture_size, 8)
            cached = beamforming.expand_dynamic_output_controls(
                cached_controls,
                aperture_size,
                8,
                interpolation_cache=cache,
            )
        tolerance = 16.0 * torch.finfo(torch.float16).eps
        torch.testing.assert_close(cached, direct, rtol=tolerance, atol=tolerance)
        direct.float().square().sum().backward()
        cached.float().square().sum().backward()
        torch.testing.assert_close(cached_controls.grad, direct_controls.grad, rtol=tolerance, atol=tolerance)

    def test_signed_hard_unity_scope_matches_explicit_and_factorized(self) -> None:
        iq_i = torch.randn((1, 3, 5))
        iq_q = torch.randn_like(iq_i)
        tx_tof = torch.zeros((1, 3))
        input_scale = torch.ones((1, 1, 1))
        input_mean = torch.zeros_like(input_scale)
        runtime.args.output_domain = "signed"
        runtime.args.unity_constraint = "hard"
        runtime.args.output_interpolation = "nearest"

        for output_weights, control_values in (
            ("real", torch.tensor([[[0.2, -0.6, 0.1, 0.4, -0.3],
                                     [-0.5, 0.3, 0.7, -0.2, 0.1],
                                     [0.4, 0.2, -0.1, 0.8, -0.5]]])),
            ("complex", torch.tensor([[[0.2, -0.6, 0.1, 0.4, -0.3,
                                        0.1, -0.2, 0.3, -0.4, 0.5],
                                       [-0.5, 0.3, 0.7, -0.2, 0.1,
                                        -0.3, 0.4, -0.1, 0.2, -0.6],
                                       [0.4, 0.2, -0.1, 0.8, -0.5,
                                        0.6, -0.5, 0.2, -0.3, 0.1]]])),
        ):
            runtime.args.output_weights = output_weights
            for scope in ("global", "per_angle"):
                runtime.args.unity_scope = scope
                weights = beamforming.mask_weights_to_aperture(control_values, None)
                wr, wi = beamforming.split_complex_weights(weights)
                if scope == "global":
                    torch.testing.assert_close(wr.sum(dim=(1, 2)), torch.ones(1))
                    torch.testing.assert_close(wi.sum(dim=(1, 2)), torch.zeros(1))
                else:
                    torch.testing.assert_close(wr.sum(dim=2), torch.ones((1, 3)))
                    torch.testing.assert_close(wi.sum(dim=2), torch.zeros((1, 3)))

                runtime.args.beamforming_implementation = "explicit"
                explicit = beamforming.beamform_iq_with_tx_phase(
                    weights, iq_i, iq_q, tx_tof, input_scale, input_mean, controls=control_values
                )
                runtime.args.beamforming_implementation = "factorized"
                factorized = beamforming.beamform_iq_with_tx_phase(
                    None, iq_i, iq_q, tx_tof, input_scale, input_mean, controls=control_values
                )
                torch.testing.assert_close(factorized[0], explicit[0], rtol=1.0e-6, atol=1.0e-6)
                torch.testing.assert_close(factorized[1], explicit[1], rtol=1.0e-6, atol=1.0e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 执行批量回归")
    def test_factorized_cuda_batch_matches_explicit_without_batched_cublas(self) -> None:
        device = torch.device("cuda")
        torch.manual_seed(23)
        batch_size, angle_count, channels, control_count, network_channels = 8, 3, 192, 8, 160
        iq_i = torch.randn((batch_size, angle_count, channels), device=device)
        iq_q = torch.randn_like(iq_i)
        controls = torch.rand((batch_size, angle_count, control_count), device=device)
        aperture_start = torch.tensor([0, 1, 4, 8, 12, 16, 20, 24], device=device)
        aperture_size = torch.tensor([1, 4, 32, 64, 96, 128, 159, 160], device=device)
        active = beamforming.dynamic_control_active_mask(aperture_size, control_count, "linear")
        controls = controls * active.unsqueeze(1).to(controls.dtype)
        weights = beamforming.unpack_dynamic_weights(
            beamforming.expand_dynamic_output_controls(controls, aperture_size, network_channels),
            aperture_start,
            aperture_size,
            channels,
        )
        tx_tof = torch.zeros((batch_size, angle_count), device=device)
        input_scale = torch.ones((batch_size, 1, 1), device=device)
        input_mean = torch.zeros_like(input_scale)

        runtime.args.beamforming_implementation = "explicit"
        explicit = beamforming.beamform_iq_with_tx_phase(
            weights,
            iq_i,
            iq_q,
            tx_tof,
            input_scale,
            input_mean,
            controls=controls,
            aperture_start=aperture_start,
            aperture_size=aperture_size,
            network_channels=network_channels,
        )
        runtime.args.beamforming_implementation = "factorized"
        factorized = beamforming.beamform_iq_with_tx_phase(
            weights,
            iq_i,
            iq_q,
            tx_tof,
            input_scale,
            input_mean,
            controls=controls,
            aperture_start=aperture_start,
            aperture_size=aperture_size,
            network_channels=network_channels,
        )
        torch.testing.assert_close(factorized[0], explicit[0], rtol=1.0e-5, atol=1.0e-5)
        torch.testing.assert_close(factorized[1], explicit[1], rtol=1.0e-5, atol=1.0e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 执行真实预测路径回归")
    def test_factorized_cuda_prediction_without_materialized_weights(self) -> None:
        class DynamicControlModel(torch.nn.Module):
            network_channels = 160
            hardware_qat = None

            def forward(self, i_data: torch.Tensor, q_data: torch.Tensor) -> torch.Tensor:
                values = i_data.new_tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
                return values.expand(i_data.shape[0], -1)

            def quantize_controls(self, controls: torch.Tensor, **_: object) -> torch.Tensor:
                return controls

        device = torch.device("cuda")
        model = DynamicControlModel().to(device)
        iq_i = torch.randn((4, 3, 192), device=device)
        iq_q = torch.randn_like(iq_i)
        mask = torch.zeros((4, 192), device=device, dtype=torch.bool)
        aperture_start = torch.tensor([0, 8, 16, 24], device=device)
        aperture_size = torch.tensor([32, 64, 128, 160], device=device)
        for index in range(mask.shape[0]):
            mask[index, aperture_start[index]:aperture_start[index] + aperture_size[index]] = True
        tx_tof = torch.zeros((4, 3), device=device)
        runtime.args.output_domain = "nonnegative"
        runtime.args.unity_constraint = "hard"
        runtime.args.projection_controls = 0

        for interpolation in ("linear", "nearest", "cubic"):
            runtime.args.output_interpolation = interpolation
            runtime.args.beamforming_implementation = "factorized"
            factorized = beamforming.predict_aperture_weights(
                model,
                iq_i,
                iq_q,
                mask,
                aperture_start,
                aperture_size,
            )
            self.assertIsNone(factorized[2])
            factorized_output = beamforming.beamform_iq_with_tx_phase(
                None,
                factorized[0],
                factorized[1],
                tx_tof,
                factorized[4],
                factorized[5],
                controls=factorized[3],
                aperture_start=aperture_start,
                aperture_size=aperture_size,
                network_channels=model.network_channels,
            )

            runtime.args.beamforming_implementation = "explicit"
            explicit = beamforming.predict_aperture_weights(
                model,
                iq_i,
                iq_q,
                mask,
                aperture_start,
                aperture_size,
            )
            explicit_output = beamforming.beamform_iq_with_tx_phase(
                explicit[2],
                explicit[0],
                explicit[1],
                tx_tof,
                explicit[4],
                explicit[5],
                controls=explicit[3],
                aperture_start=aperture_start,
                aperture_size=aperture_size,
                network_channels=model.network_channels,
            )
            torch.testing.assert_close(factorized_output[0], explicit_output[0], rtol=1.0e-5, atol=1.0e-5)
            torch.testing.assert_close(factorized_output[1], explicit_output[1], rtol=1.0e-5, atol=1.0e-5)

    def test_factorized_output_constraints_match_explicit(self) -> None:
        iq_i = torch.tensor([[[0.4, -0.2, 0.7, 0.1, -0.3], [0.6, 0.2, -0.5, 0.3, 0.8]]])
        iq_q = torch.tensor([[[0.1, 0.5, -0.4, 0.9, 0.2], [-0.1, 0.7, 0.6, -0.2, 0.3]]])
        tx_tof = torch.zeros((1, 2))
        input_scale = torch.ones((1, 1, 1))
        input_mean = torch.zeros((1, 1, 1))
        values = {
            ("nonnegative", "hard"): torch.tensor([[[0.2, 0.6, 0.1], [0.5, 0.3, 0.7]]]),
            ("nonnegative", "free"): torch.tensor([[[0.2, 0.6, 0.1], [0.5, 0.3, 0.7]]]),
            ("signed", "hard"): torch.tensor([[[0.2, -0.6, 0.1], [-0.5, 0.3, 0.7]]]),
            ("signed", "free"): torch.tensor([[[0.2, -0.6, 0.1], [-0.5, 0.3, 0.7]]]),
        }
        runtime.args.output_interpolation = "nearest"
        for (domain, unity), controls in values.items():
            runtime.args.output_domain = domain
            runtime.args.unity_constraint = unity
            weights = beamforming.expand_fixed_output_controls(controls, 5)
            if domain == "signed" and unity == "hard":
                weights = beamforming.mask_weights_to_aperture(weights, None)
            runtime.args.beamforming_implementation = "explicit"
            explicit = beamforming.beamform_iq_with_tx_phase(
                weights,
                iq_i,
                iq_q,
                tx_tof,
                input_scale,
                input_mean,
                controls=controls,
            )
            runtime.args.beamforming_implementation = "factorized"
            factorized = beamforming.beamform_iq_with_tx_phase(
                weights,
                iq_i,
                iq_q,
                tx_tof,
                input_scale,
                input_mean,
                controls=controls,
            )
            torch.testing.assert_close(factorized[0], explicit[0], rtol=1.0e-6, atol=1.0e-6)
            torch.testing.assert_close(factorized[1], explicit[1], rtol=1.0e-6, atol=1.0e-6)

    def test_factorized_basis_matches_resize_for_all_interpolations(self) -> None:
        values = torch.randn((2, 3, 3))
        for interpolation in ("linear", "nearest", "cubic"):
            runtime.args.output_interpolation = interpolation
            if interpolation == "nearest":
                basis = beamforming._nearest_interpolation_basis(3, 8, values.device, values.dtype)
            elif interpolation == "linear":
                basis = beamforming._linear_interpolation_basis(3, 8, values.device, values.dtype)
            else:
                basis = beamforming._cubic_interpolation_basis(3, 8, values.device, values.dtype)
            actual = torch.mm(values.reshape(-1, 3), basis).reshape(2, 3, 8)
            if interpolation == "nearest":
                expected = torch.nn.functional.interpolate(values.reshape(-1, 1, 3), size=8, mode="nearest").reshape(2, 3, 8)
            elif interpolation == "linear":
                expected = beamforming._linear_resize_last(values, 8)
            else:
                expected = beamforming._cubic_resize_last(values, 8)
            torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_d1_and_d2_loss_terms_are_independently_weighted(self) -> None:
        weights = torch.tensor([[[0.0, 1.0, 3.0, 6.0]]])
        runtime.args.loss["terms"]["d1"] = {"domain": "absolute", "weight": 2.0}
        runtime.args.loss["terms"]["d2"] = {"domain": "absolute", "weight": 3.0}
        d1 = beamforming._aperture_smoothness(weights, None, "absolute", 1)
        d2 = beamforming._aperture_smoothness(weights, None, "absolute", 2)
        actual = beamforming.aperture_weight_regularization(weights, None)
        torch.testing.assert_close(actual, 2.0 * d1 + 3.0 * d2)

    def test_factorized_projection_prediction_is_rejected(self) -> None:
        runtime.args.projection_controls = 2
        runtime.args.beamforming_implementation = "factorized"
        with self.assertRaisesRegex(ValueError, "F/factorized.*projection_controls"):
            beamforming.predict_aperture_weights(
                object(), torch.zeros((1, 1, 8)), torch.zeros((1, 1, 8)), None
            )
        with self.assertRaisesRegex(ValueError, "F/factorized.*projection_controls"):
            beamforming.beamform_iq_with_tx_phase(
                None,
                torch.zeros((1, 1, 8)),
                torch.zeros((1, 1, 8)),
                torch.zeros((1, 1)),
                torch.ones((1, 1, 1)),
                torch.zeros((1, 1, 1)),
                controls=torch.zeros((1, 1, 8)),
            )

    def test_fixed_signed_hard_projection_preserves_unity(self) -> None:
        class SignedProjectionModel(torch.nn.Module):
            network_channels = 8
            hardware_qat = None

            def forward(self, i_data: torch.Tensor, q_data: torch.Tensor) -> torch.Tensor:
                values = i_data.new_tensor([[0.2, -0.6, 0.1, 0.4, -0.3, 0.5, -0.2, 0.7]])
                return values.expand(i_data.shape[0], -1)

            def quantize_controls(self, controls: torch.Tensor, **_: object) -> torch.Tensor:
                return controls

        runtime.args.output_domain = "signed"
        runtime.args.unity_constraint = "hard"
        runtime.args.projection_controls = 2
        model = SignedProjectionModel()
        iq_i = torch.randn((2, 3, 8))
        iq_q = torch.randn_like(iq_i)
        for interpolation in ("linear", "nearest", "cubic"):
            runtime.args.output_interpolation = interpolation
            result = beamforming.predict_aperture_weights(model, iq_i, iq_q, None)
            weights = result[2]
            self.assertIsNotNone(weights)
            real_weights, imag_weights = beamforming.split_complex_weights(weights)
            torch.testing.assert_close(real_weights.sum(dim=(1, 2)), torch.ones(2))
            torch.testing.assert_close(imag_weights.sum(dim=(1, 2)), torch.zeros(2))
            runtime.args.beamforming_implementation = "explicit"
            explicit = beamforming.beamform_iq_with_tx_phase(
                weights, result[0], result[1], torch.zeros((2, 3)),
                torch.ones((2, 1, 1)), torch.zeros((2, 1, 1)), controls=result[3]
            )
            self.assertTrue(torch.isfinite(explicit[0]).all())
            self.assertTrue(torch.isfinite(explicit[1]).all())

    def test_prediction_auto_materialization_follows_operator_requirements(self) -> None:
        class ControlModel(torch.nn.Module):
            network_channels = 8
            hardware_qat = None

            def forward(self, i_data: torch.Tensor, q_data: torch.Tensor) -> torch.Tensor:
                return i_data.new_tensor([[0.2, 0.4, 0.7]]).expand(i_data.shape[0], -1)

            def quantize_controls(self, controls: torch.Tensor, **_: object) -> torch.Tensor:
                return controls

        model = ControlModel()
        iq_i = torch.randn((2, 3, 8))
        iq_q = torch.randn_like(iq_i)
        runtime.args.beamforming_implementation = "factorized"
        runtime.args.projection_controls = 0
        implicit = beamforming.predict_aperture_weights(model, iq_i, iq_q, None)
        self.assertIsNone(implicit[2])
        runtime.args.projection_controls = 2
        with self.assertRaisesRegex(ValueError, "F/factorized.*projection_controls"):
            beamforming.predict_aperture_weights(model, iq_i, iq_q, None)
        runtime.args.beamforming_implementation = "explicit"
        explicit = beamforming.predict_aperture_weights(model, iq_i, iq_q, None)
        self.assertIsNotNone(explicit[2])

    def test_dynamic_factorized_matches_explicit_without_full_weights(self) -> None:
        class DynamicControlModel(torch.nn.Module):
            network_channels = 8
            hardware_qat = None

            def forward(self, i_data: torch.Tensor, q_data: torch.Tensor) -> torch.Tensor:
                return i_data.new_tensor([[0.2, 0.4, 0.7]]).expand(i_data.shape[0], -1)

            def quantize_controls(self, controls: torch.Tensor, **_: object) -> torch.Tensor:
                return controls

        model = DynamicControlModel()
        iq_i = torch.randn((2, 2, 8))
        iq_q = torch.randn_like(iq_i)
        mask = torch.tensor([[True, True, True, True, False, False, False, False],
                             [False, True, True, True, True, True, False, False]])
        aperture_start = torch.tensor([0, 1])
        aperture_size = torch.tensor([4, 5])
        tx_tof = torch.zeros((2, 2))
        input_scale = torch.ones((2, 1, 1))
        input_mean = torch.zeros_like(input_scale)

        runtime.args.beamforming_implementation = "factorized"
        implicit = beamforming.predict_aperture_weights(
            model,
            iq_i,
            iq_q,
            mask,
            aperture_start,
            aperture_size,
        )
        self.assertIsNone(implicit[2])
        implicit_output = beamforming.beamform_iq_with_tx_phase(
            None,
            implicit[0],
            implicit[1],
            tx_tof,
            input_scale,
            input_mean,
            controls=implicit[3],
            aperture_start=aperture_start,
            aperture_size=aperture_size,
            network_channels=model.network_channels,
        )

        runtime.args.beamforming_implementation = "explicit"
        weighted = beamforming.predict_aperture_weights(
            model,
            iq_i,
            iq_q,
            mask,
            aperture_start,
            aperture_size,
        )
        self.assertIsNotNone(weighted[2])
        explicit_output = beamforming.beamform_iq_with_tx_phase(
            weighted[2],
            weighted[0],
            weighted[1],
            tx_tof,
            input_scale,
            input_mean,
            controls=weighted[3],
            aperture_start=aperture_start,
            aperture_size=aperture_size,
            network_channels=model.network_channels,
        )
        torch.testing.assert_close(implicit_output[0], explicit_output[0], rtol=1.0e-6, atol=1.0e-6)
        torch.testing.assert_close(implicit_output[1], explicit_output[1], rtol=1.0e-6, atol=1.0e-6)

    def test_dynamic_projection_explicit_path_materializes(self) -> None:
        class DynamicProjectionModel(torch.nn.Module):
            network_channels = 8
            hardware_qat = None

            def forward(self, i_data: torch.Tensor, q_data: torch.Tensor) -> torch.Tensor:
                values = i_data.new_tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
                return values.expand(i_data.shape[0], -1)

            def quantize_controls(self, controls: torch.Tensor, **_: object) -> torch.Tensor:
                return controls

        model = DynamicProjectionModel()
        iq_i = torch.randn((2, 2, 8))
        iq_q = torch.randn_like(iq_i)
        mask = torch.tensor([[True, True, True, True, False, False, False, False],
                             [False, True, True, True, True, True, False, False]])
        aperture_start = torch.tensor([0, 1])
        aperture_size = torch.tensor([4, 5])
        tx_tof = torch.zeros((2, 2))
        runtime.args.projection_controls = 2
        runtime.args.output_domain = "nonnegative"
        runtime.args.unity_constraint = "hard"

        for interpolation in ("linear", "nearest", "cubic"):
            runtime.args.output_interpolation = interpolation
            runtime.args.beamforming_implementation = "explicit"
            explicit = beamforming.predict_aperture_weights(
                model,
                iq_i,
                iq_q,
                mask,
                aperture_start,
                aperture_size,
            )
            explicit_output = beamforming.beamform_iq_with_tx_phase(
                explicit[2], explicit[0], explicit[1], tx_tof,
                explicit[4], explicit[5], controls=explicit[3],
                aperture_start=aperture_start, aperture_size=aperture_size,
                network_channels=model.network_channels,
            )
            self.assertTrue(torch.isfinite(explicit_output[0]).all())
            self.assertTrue(torch.isfinite(explicit_output[1]).all())

    def test_dynamic_projection_nearest_matches_two_stage_resize(self) -> None:
        values = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
        for interpolation in ("nearest", "cubic"):
            runtime.args.output_interpolation = interpolation
            projected = beamforming.project_active_aperture_weights(values, torch.tensor([5]), 2)
            active = values[..., 1:6]
            if interpolation == "nearest":
                compact = torch.nn.functional.interpolate(active, size=2, mode="nearest")
                restored = torch.nn.functional.interpolate(compact, size=5, mode="nearest")
            else:
                compact = beamforming._cubic_resize_last(active, 2)
                restored = beamforming._cubic_resize_last(compact, 5)
            expected = values.clone()
            expected[..., 1:6] = restored
            torch.testing.assert_close(projected, expected)

    def test_dynamic_projection_cubic_matches_two_stage_resize_for_batch(self) -> None:
        values = torch.arange(3 * 2 * 8, dtype=torch.float32).view(3, 2, 8)
        aperture_size = torch.tensor([4, 5, 6])
        runtime.args.output_interpolation = "cubic"
        projected = beamforming.project_active_aperture_weights(values, aperture_size, 3)
        expected = values.clone()
        for index, size in enumerate(aperture_size.tolist()):
            start = (values.shape[-1] - size) // 2
            active = values[index : index + 1, :, start : start + size]
            compact = beamforming._cubic_resize_last(active, 3)
            restored = beamforming._cubic_resize_last(compact, size)
            expected[index, :, start : start + size] = restored[0]
        torch.testing.assert_close(projected, expected)

    def test_dynamic_factorized_output_constraints_match_explicit(self) -> None:
        class DynamicOutputModel(torch.nn.Module):
            network_channels = 8
            hardware_qat = None

            def forward(self, i_data: torch.Tensor, q_data: torch.Tensor) -> torch.Tensor:
                values = i_data.new_tensor([[-0.2, 0.4, -0.7]])
                return values.expand(i_data.shape[0], -1)

            def quantize_controls(self, controls: torch.Tensor, **_: object) -> torch.Tensor:
                return controls

        model = DynamicOutputModel()
        iq_i = torch.randn((2, 2, 8))
        iq_q = torch.randn_like(iq_i)
        mask = torch.tensor([[True, True, True, True, False, False, False, False],
                             [False, True, True, True, True, True, False, False]])
        aperture_start = torch.tensor([0, 1])
        aperture_size = torch.tensor([4, 5])
        tx_tof = torch.zeros((2, 2))
        for interpolation in ("linear", "nearest", "cubic"):
            for domain, unity in (
                ("nonnegative", "hard"),
                ("nonnegative", "free"),
                ("signed", "hard"),
                ("signed", "free"),
            ):
                runtime.args.output_interpolation = interpolation
                runtime.args.output_domain = domain
                runtime.args.unity_constraint = unity
                runtime.args.projection_controls = 0
                runtime.args.beamforming_implementation = "factorized"
                factorized = beamforming.predict_aperture_weights(
                    model,
                    iq_i,
                    iq_q,
                    mask,
                    aperture_start,
                    aperture_size,
                )
                self.assertIsNone(factorized[2])
                factorized_output = beamforming.beamform_iq_with_tx_phase(
                    factorized[2], factorized[0], factorized[1], tx_tof,
                    factorized[4], factorized[5], controls=factorized[3],
                    aperture_start=aperture_start, aperture_size=aperture_size,
                    network_channels=model.network_channels,
                )

                runtime.args.beamforming_implementation = "explicit"
                explicit = beamforming.predict_aperture_weights(
                    model,
                    iq_i,
                    iq_q,
                    mask,
                    aperture_start,
                    aperture_size,
                )
                explicit_output = beamforming.beamform_iq_with_tx_phase(
                    explicit[2], explicit[0], explicit[1], tx_tof,
                    explicit[4], explicit[5], controls=explicit[3],
                    aperture_start=aperture_start, aperture_size=aperture_size,
                    network_channels=model.network_channels,
                )
                torch.testing.assert_close(factorized_output[0], explicit_output[0], rtol=1.0e-6, atol=1.0e-6)
                torch.testing.assert_close(factorized_output[1], explicit_output[1], rtol=1.0e-6, atol=1.0e-6)

    def test_dynamic_complex_factorized_matches_explicit(self) -> None:
        class DynamicComplexModel(torch.nn.Module):
            network_channels = 8
            hardware_qat = None

            def forward(self, i_data: torch.Tensor, q_data: torch.Tensor) -> torch.Tensor:
                values = i_data.new_tensor([[0.2, -0.4, 0.6, -0.3, 0.5, -0.7]])
                return values.expand(i_data.shape[0], -1)

            def quantize_controls(self, controls: torch.Tensor, **_: object) -> torch.Tensor:
                return controls

        model = DynamicComplexModel()
        iq_i = torch.randn((2, 2, 8))
        iq_q = torch.randn_like(iq_i)
        mask = torch.tensor([[True, True, True, True, False, False, False, False],
                             [False, True, True, True, True, True, False, False]])
        aperture_start = torch.tensor([0, 1])
        aperture_size = torch.tensor([4, 5])
        tx_tof = torch.zeros((2, 2))
        runtime.args.output_weights = "complex"
        runtime.args.output_domain = "signed"
        runtime.args.projection_controls = 0
        runtime.args.unity_scope = "per_angle"

        for interpolation in ("linear", "nearest", "cubic"):
            for unity in ("hard", "free"):
                runtime.args.output_interpolation = interpolation
                runtime.args.unity_constraint = unity
                runtime.args.beamforming_implementation = "factorized"
                factorized = beamforming.predict_aperture_weights(
                    model,
                    iq_i,
                    iq_q,
                    mask,
                    aperture_start,
                    aperture_size,
                )
                factorized_output = beamforming.beamform_iq_with_tx_phase(
                    factorized[2], factorized[0], factorized[1], tx_tof,
                    factorized[4], factorized[5], controls=factorized[3],
                    aperture_start=aperture_start, aperture_size=aperture_size,
                    network_channels=model.network_channels,
                )

                runtime.args.beamforming_implementation = "explicit"
                explicit = beamforming.predict_aperture_weights(
                    model,
                    iq_i,
                    iq_q,
                    mask,
                    aperture_start,
                    aperture_size,
                )
                explicit_output = beamforming.beamform_iq_with_tx_phase(
                    explicit[2], explicit[0], explicit[1], tx_tof,
                    explicit[4], explicit[5], controls=explicit[3],
                    aperture_start=aperture_start, aperture_size=aperture_size,
                    network_channels=model.network_channels,
                )
                torch.testing.assert_close(factorized_output[0], explicit_output[0], rtol=1.0e-6, atol=1.0e-6)
                torch.testing.assert_close(factorized_output[1], explicit_output[1], rtol=1.0e-6, atol=1.0e-6)

    def test_direct_control_dynamic_aperture_uses_local_active_channels(self) -> None:
        class DirectControlModel(torch.nn.Module):
            network_channels = 4
            hardware_qat = None

            def forward(self, i_data: torch.Tensor, q_data: torch.Tensor) -> torch.Tensor:
                values = i_data.new_tensor([[0.1, 0.2, 0.3, 0.4]])
                return values.expand(i_data.shape[0], -1)

            def quantize_controls(self, controls: torch.Tensor, **_: object) -> torch.Tensor:
                return controls

        model = DirectControlModel()
        i_data = torch.zeros((1, 1, 4))
        q_data = torch.zeros_like(i_data)
        aperture_start = torch.tensor([1])
        aperture_size = torch.tensor([2])
        mask = torch.tensor([[False, True, True, False]])
        _, _, _, controls, _, _, _ = beamforming.predict_aperture_weights(
            model,
            i_data,
            q_data,
            mask,
            aperture_start,
            aperture_size,
        )
        torch.testing.assert_close(controls, torch.tensor([[[0.0, 2.0 / 3.0, 1.0, 0.0]]]))


if __name__ == "__main__":
    unittest.main()
