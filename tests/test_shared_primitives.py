"""Regression tests for the shared numerical primitives used across layers."""

import unittest

import numpy as np
import torch

from scipy.signal import firwin

from shared_core.geometry import aperture_channel_count, element_positions, image_extent_mm
from shared_core.images import (
    db_to_display,
    display_to_db,
    envelope_from_db,
    tgc_attenuation_db,
    tgc_gain_from_depth,
    tgc_gain_from_depth_tensor,
)
from shared_core.selection import select_angle_indices
from shared_core.signals import ddc_lowpass_coefficients


class DisplayMappingTests(unittest.TestCase):
    def test_display_mapping_is_the_inverse_of_db_mapping(self):
        db = np.array([[-60.0, -30.5, 0.0], [-12.25, -58.0, -6.0]])
        display = db_to_display(db, 60.0)
        np.testing.assert_allclose(display_to_db(display, 60.0), db, atol=1e-6)

    def test_display_mapping_clips_to_unit_interval(self):
        np.testing.assert_array_equal(
            db_to_display(np.array([[-120.0, 6.0]]), 60.0),
            np.array([[0.0, 1.0]]),
        )
        np.testing.assert_array_equal(
            display_to_db(np.array([[-1.0, 2.0]]), 60.0),
            np.array([[-60.0, 0.0]]),
        )

    def test_display_mapping_accepts_torch_tensors(self):
        db = torch.tensor([-60.0, -30.0, 0.0])
        display = db_to_display(db, 60.0)
        self.assertIsInstance(display, torch.Tensor)
        torch.testing.assert_close(display, torch.tensor([0.0, 0.5, 1.0]))

    def test_envelope_from_db_matches_amplitude_ratio(self):
        np.testing.assert_allclose(
            envelope_from_db(np.array([0.0, -20.0])), np.array([1.0, 0.1]), rtol=1e-6
        )


class TgcTests(unittest.TestCase):
    def test_tgc_gain_matches_attenuation_formula(self):
        depth = np.array([0.01, 0.02, 0.03])
        gain_db = tgc_attenuation_db(depth, 5e6, 0.5)
        np.testing.assert_allclose(gain_db, 0.5 * 5.0 * depth * 100.0 * 2.0)
        np.testing.assert_allclose(
            tgc_gain_from_depth(depth, 5e6, 0.5), np.power(10.0, gain_db / 20.0)
        )

    def test_tensor_tgc_matches_numpy_gain(self):
        depth = torch.tensor([0.01, 0.02, 0.03])
        expected = tgc_gain_from_depth(depth.numpy(), 5e6, 0.5).astype(np.float32)
        torch.testing.assert_close(
            tgc_gain_from_depth_tensor(depth, 5e6, 0.5),
            torch.from_numpy(expected),
        )


class AngleSelectionTests(unittest.TestCase):
    def test_modes_use_zero_centered_center_and_endpoint_inclusive_counts(self):
        angles = np.array([0.2, -0.1, 0.0, -0.2, 0.1])
        np.testing.assert_array_equal(select_angle_indices(angles, "center"), [2])
        np.testing.assert_array_equal(select_angle_indices(angles, 1), [2])
        np.testing.assert_array_equal(select_angle_indices(angles, "all"), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(select_angle_indices(angles, 3), [3, 2, 0])
        np.testing.assert_array_equal(select_angle_indices(angles, "3"), [3, 2, 0])

    def test_invalid_selections_are_rejected(self):
        angles = np.array([-0.1, 0.0, 0.1])
        for selection in (0, 4, True, 1.5, "single"):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                select_angle_indices(angles, selection)


class ApertureAndFilterTests(unittest.TestCase):
    def test_image_extent_is_millimetres_with_inverted_depth_axis(self):
        self.assertEqual(
            image_extent_mm(np.array([-0.01, 0.01]), np.array([0.0, 0.02])),
            [-10.0, 10.0, 20.0, 0.0],
        )

    def test_aperture_count_matches_identity_for_numpy_and_torch(self):
        depth = np.array([0.01, 0.02, 0.05])
        expected = np.floor(depth / (1.5 * 0.0003)) + 1
        np.testing.assert_array_equal(
            aperture_channel_count(depth, 1.5, 0.0003), expected
        )
        torch.testing.assert_close(
            aperture_channel_count(torch.from_numpy(depth), 1.5, 0.0003).double(),
            torch.from_numpy(expected),
        )

    def test_element_positions_are_centred_on_the_array(self):
        torch.testing.assert_close(
            element_positions(4, 0.0003), torch.tensor([-0.00045, -0.00015, 0.00015, 0.00045])
        )

    def test_ddc_lowpass_matches_the_project_contract(self):
        fs, fc = 20e6, 5e6
        np.testing.assert_allclose(
            ddc_lowpass_coefficients(fs, fc),
            firwin(64, 0.8 * fc / (fs / 2)),
        )
        np.testing.assert_array_equal(
            ddc_lowpass_coefficients(fs, fc, taps=129, window=("kaiser", 8.0)).astype(np.float32),
            firwin(129, 0.8 * fc, window=("kaiser", 8.0), fs=fs).astype(np.float32),
        )


if __name__ == "__main__":
    unittest.main()
