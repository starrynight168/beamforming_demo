import importlib
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from algorithms.shared_core.common import COMMON_PARAMS


class AlgorithmRuntimeTests(unittest.TestCase):
    names = ("das", "mv", "ccmv", "esbmv", "gcfmv", "fdmas", "cmsaw")

    def test_imports_do_not_read_configuration_or_probe_cuda(self):
        code = """
import importlib
from unittest.mock import patch
import torch
import algorithms.shared_core.common as common
with patch.object(common, 'load_common_params', side_effect=AssertionError('configuration read during import')), patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CUDA probed during import')):
    for name in ('das', 'mv', 'ccmv', 'esbmv', 'gcfmv', 'fdmas', 'cmsaw'):
        importlib.import_module('algorithms.' + name)
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_parser_factories_are_independent_and_accept_explicit_defaults(self):
        defaults = {**COMMON_PARAMS, "dr": 37.0}
        for name in self.names:
            module = importlib.import_module("algorithms." + name)
            with self.subTest(name=name), patch("algorithms.shared_core.common.load_common_params", side_effect=AssertionError("unexpected config read")):
                first = module.build_parser(defaults)
                first.set_defaults(dr=12.0)
                second = module.build_parser(defaults)
                self.assertEqual(second.parse_args([]).dr, 37.0)
                self.assertFalse(hasattr(module, "parser"))
                self.assertFalse(hasattr(module, "device"))

    def test_instances_ignore_cli_globals_and_use_explicit_device(self):
        geometry = (np.array([1.0]), np.array([0.0]), 4, 1.0, 1.0, 0.0, 1.0, np.zeros(1), np.zeros(1))
        for name in self.names:
            module = importlib.import_module("algorithms." + name)
            cls = (module.DASBeamformerIQ if name == "das" else
                   module.FDMASBeamformerIQ if name == "fdmas" else
                   module.CMSAWBeamformerIQ if name == "cmsaw" else
                   module.RowDynamicCCMVBeamformerIQ if name == "ccmv" else module.RowDynamicMVBeamformerIQ)
            with self.subTest(name=name), patch.object(module, "args", SimpleNamespace(signal_domain="invalid"), create=True), patch.object(torch.cuda, "is_available", side_effect=AssertionError("explicit device must not probe CUDA")):
                model = cls(*geometry, device="cpu")
                self.assertEqual(model.options.signal_domain, COMMON_PARAMS["signal_domain"])
                self.assertEqual(model.device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
