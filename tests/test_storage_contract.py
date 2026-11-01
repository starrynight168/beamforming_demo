import unittest

import numpy as np

from algorithms.shared_core.labels import target_label
from algorithms.shared_core.normalization import normalization_scale, storage_normalization_mode


class StorageContractTests(unittest.TestCase):
    def test_normalization_formulas(self):
        real, imag = np.array([0., 3., 6.]), np.array([0., 4., 8.])
        self.assertEqual(normalization_scale(real, imag, "none"), 1.)
        self.assertAlmostEqual(normalization_scale(real, imag, "rms"), np.sqrt(125. / 3. + 1e-12), places=6)
        self.assertAlmostEqual(normalization_scale(real, imag, "env_std"), np.std([0., 5., 10.]) + 1e-12, places=6)

    def test_domains_share_a_method_and_can_keep_physical_units(self):
        for mode in ("none", "rms", "env_std"):
            inputs = {"rf": {"dtype": "float32", "normalization": "none"},
                      "iq": {"dtype": "float16", "normalization": mode}}
            self.assertEqual(storage_normalization_mode(inputs), mode)

    def test_conflicting_methods_and_obsolete_switch_are_rejected(self):
        for inputs in (
            {"iq": {"dtype": "float16", "normalize": True}},
            {"iq": {"dtype": "float16", "normalization": True}},
            {"rf": {"dtype": "float32", "normalization": "rms"},
             "iq": {"dtype": "float16", "normalization": "env_std"}},
        ):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                storage_normalization_mode(inputs)

    def test_target_titles_use_metadata_angle_counts_and_single_angle(self):
        for count in (3, 87):
            config = {"generation": {
                "input_iq": {"all_steering_angles_rad": [-.1, 0., .1]},
                "training_targets": {"domain_targets": {"iq": {
                    "mv": {"angles": {"selection": "input", "indices": [1]}},
                    "das": {"angles": {"selection": "all", "indices": {
                        "encoding": "arithmetic_sequence", "start": 0, "step": 1, "count": count}}},
                }}},
            }}
            self.assertEqual(target_label(config, "iq", "mv"), "MV (IQ, 1 angle, 0°)")
            self.assertEqual(target_label(config, "iq", "das"), f"DAS (IQ, {count} angles compounded)")

    def test_reference_title_uses_source_domain_without_guessing_angle_count(self):
        config = {"generation": {"training_targets": {"domain_targets": {
            "rf": {"reference": {"kind": "image_reference", "source_domain": "iq"}},
        }}}}
        self.assertEqual(target_label(config, "rf", "reference"), "REFERENCE (IQ)")

    def test_declared_all_or_input_angles_do_not_require_repeated_indices(self):
        config = {"generation": {
            "input_iq": {"all_steering_angles_rad": [-.1, 0., .1], "selected_angle_indices": [1]},
            "training_targets": {"domain_targets": {"iq": {
                "mv": {"angles": {"selection": "input"}},
                "das": {"angles": {"selection": "all"}},
            }}},
        }}
        self.assertEqual(target_label(config, "iq", "mv"), "MV (IQ, 1 angle, 0°)")
        self.assertEqual(target_label(config, "iq", "das"), "DAS (IQ, 3 angles compounded)")
