import unittest

import torch

from shared_core.geometry import plane_wave_distance, receive_distance


class GeometryTests(unittest.TestCase):
    def test_receive_distance_broadcasts_pixels_and_elements(self):
        x = torch.tensor([[0.0], [3.0]])
        z = torch.tensor([[4.0], [4.0]])
        elements = torch.tensor([[0.0, 3.0]])
        torch.testing.assert_close(receive_distance(x, z, elements), torch.tensor([[4.0, 5.0], [5.0, 4.0]]))

    def test_plane_wave_distance_broadcasts_angles(self):
        x = torch.tensor([[1.0], [3.0]])
        z = torch.tensor([[2.0], [4.0]])
        cos_angle = torch.tensor([[1.0, 0.0]])
        sin_angle = torch.tensor([[0.0, 1.0]])
        torch.testing.assert_close(plane_wave_distance(x, z, cos_angle, sin_angle), torch.tensor([[2.0, 1.0], [4.0, 3.0]]))
