import io
import unittest

import h5py

from data.checks.check_data import check_time_axis
from shared_core.data import embedded_config


class EmbeddedConfigTests(unittest.TestCase):
    def test_valid_unicode_is_preserved(self):
        with h5py.File(io.BytesIO(), "w") as handle:
            handle.create_dataset("config_yaml", data="dataset: 患者\n")
            self.assertEqual(embedded_config(handle), {"dataset": "患者"})

    def test_invalid_utf8_is_rejected_and_reported_by_data_check(self):
        with h5py.File(io.BytesIO(), "w") as handle:
            handle.create_dataset("config_yaml", data=b"dataset: patient_\xff42\n")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                embedded_config(handle)
            problems = []
            check_time_axis(handle, problems)
            self.assertTrue(any("UTF-8" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
