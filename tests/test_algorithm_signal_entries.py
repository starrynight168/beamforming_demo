import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import yaml

from algorithms.common import beamforming_depth_grid, beamformed_image_db


class AlgorithmSignalEntryTests(unittest.TestCase):
    def test_analytic_cache_only_cli_runs_joint_mv_without_rf_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "analytic.h5"
            source = np.random.default_rng(83).normal(size=(1, 3, 256, 4)).astype(np.float32)
            with h5py.File(path, "w") as h:
                for name, value in dict(ar_i=source, ar_q=source*.2,
                        ar_fs=20e6, ar_t0=np.zeros((1,3)), ar_len=[256],
                        ar_angles=[-.1,0.,.1], all_scale_ref=[1.], c=1540.,fc=5e6,pitch=.0003,
                        num_channels=4,z_grid=np.linspace(.003,.006,4),x_grid=np.linspace(-.0003,.0003,3)).items():
                    h[name] = value
                h["config_yaml"] = yaml.safe_dump({"generation": {"storage": {"inputs": {
                    "analytic_rf": {"dtype": "float32", "normalize": True}}},
                    "ground_truth": {}, "training_targets": {}}})
            module = importlib.import_module("algorithms.mv")
            argv = ["mv", "--h5_path", str(path), "--output_dir", str(root),
                    "--signal_domain", "analytic_rf", "--mv_axis", "joint", "--select_angles", "all", "--no_tgc"]
            with patch.object(sys, "argv", argv), patch("algorithms.shared_core.data.rf_to_analytic", side_effect=AssertionError("cache must be used")), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            image = np.load(root / "mv/mv.npy")
            self.assertEqual(image.shape, (4, 3))
            self.assertTrue(np.isfinite(image).all())

    def test_mv_variants_support_complex_domains_on_each_adaptive_axis(self):
        rng = np.random.default_rng(67)
        source = rng.normal(size=(3, 256, 4)).astype(np.float32)
        angles = np.array([-.1, 0., .1], np.float32)
        z = np.array([.004, .005], np.float32)
        x = np.array([-.0001, .0001], np.float32)
        for name in ("mv", "ccmv", "esbmv", "gcfmv", "cmsaw"):
            module = importlib.import_module("algorithms." + name)
            cls = (module.RowDynamicCCMVBeamformerIQ if name == "ccmv" else
                   module.CMSAWBeamformerIQ if name == "cmsaw" else module.RowDynamicMVBeamformerIQ)
            for axis in ("rx", "tx", "joint"):
                for domain in ("rf", "analytic_rf"):
                    settings = module.parser.parse_args(["--signal_domain", domain, "--mv_axis", axis, "--interp", "linear"])
                    bf = cls(z, x, 4, .0003, 1540., 5e6, 20e6, 0., angles,
                             mv_dl=.1, temporal_win=3, subarray_ratio=.5, options=settings)
                    imag = np.zeros_like(source) if domain == "rf" else source * .2
                    with self.subTest(method=name, axis=axis, domain=domain):
                        real_out, imag_out = bf(source, imag, angles, 0., 20e6)
                        self.assertEqual(real_out.shape, (2, 2))
                        self.assertTrue(np.isfinite(real_out).all() and np.isfinite(imag_out).all())
                        if domain == "rf":
                            np.testing.assert_allclose(imag_out, 0., atol=1e-6)
                        else:
                            np.testing.assert_allclose(imag_out, real_out * .2, atol=2e-5, rtol=2e-4)

    def test_rf_only_h5_runs_supported_algorithm_domains_and_records_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'rf.h5'
            fs, fc, c = 30e6, 5e6, 1540.
            time = np.arange(512)/fs
            rf = np.exp(-((time-6e-6)/1e-6)**2)*np.cos(2*np.pi*fc*time)
            z = np.linspace(.003, .006, 5, dtype=np.float32)
            with h5py.File(path,'w') as h:
                for key,value in dict(rf=np.broadcast_to(rf[None,None,:,None],(1,1,512,4)),rf_fs=fs,rf_t0=np.zeros((1,1)),rf_length=[512],rf_angles=[0.],all_scale_ref=[1.],c=c,fc=fc,pitch=.0003,num_channels=4,z_grid=z,x_grid=np.linspace(-.0003,.0003,3)).items():
                    h[key]=value
                h['config_yaml']=yaml.safe_dump({'acquisition':{'fc_hz':fc},'generation':{'storage':{'inputs':{'rf':{'dtype':'float32','normalize':False}}},'input_iq':{'source_sampling_frequency_hz':fs,'packed_sampling_frequency_hz':fs,'decimation_factor':1,'ddc':{'phase_sign_q':-1,'mixing_gain':1.,'cutoff_hz':4e6,'num_taps':64,'lowpass':'firwin_64_then_filtfilt','resampling':'scipy_resample_poly_kaiser_12'},'normalization':{'mode':'complex_rms','epsilon':1e-12}},'ground_truth':{},'training_targets':{}}})
            for method in ('das','mv','ccmv','esbmv','gcfmv','cmsaw','fdmas'):
                module=importlib.import_module('algorithms.'+method)
                for domain in ('iq','analytic_rf','rf'):
                    argv=[method,'--h5_path',str(path),'--output_dir',str(root/domain),'--signal_domain',domain,'--no_tgc','--interp','linear']
                    if method!='das' and method!='fdmas':
                        argv+=['--temporal_win','3','--subarray_ratio','0.5']
                    with self.subTest(method=method,domain=domain), patch.object(sys,'argv',argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        module.main()
                        image=np.load(root/domain/method/(method+'.npy'))
                        self.assertEqual(image.shape,(5,3))
                        self.assertTrue(np.isfinite(image).all())
                        params=json.loads((root/domain/method/'params.json').read_text())
                        self.assertEqual(params['signal_domain'],domain)
                        if domain=='rf':
                            if method == 'fdmas':
                                compute_z = module.fdmas_depth_grid(z, c, fc, fs, domain)
                                expected_rows = compute_z.size
                                self.assertEqual(compute_z[0], z[0])
                                self.assertGreater(compute_z[-1], z[-1])
                            else:
                                expected_rows = beamforming_depth_grid(z, c, fs, domain).size
                            self.assertEqual(params['beamforming_depth_rows'], expected_rows)

    def test_rf_display_requires_real_output_and_dense_hilbert(self):
        z=np.linspace(.001,.002,5,dtype=np.float32)
        dense=beamforming_depth_grid(z,1540.,20e6,'rf')
        self.assertGreater(len(dense),len(z))
        real=np.cos(2*np.pi*5e6*2*dense/1540)[:,None]*np.ones((1,3))
        with self.assertRaisesRegex(ValueError,'虚部'):
            beamformed_image_db(real,np.ones_like(real),dense,z,5e6,False,.5,'rf')
        db=beamformed_image_db(real,np.zeros_like(real),dense,z,5e6,False,.5,'rf')
        self.assertEqual(db.shape,(5,3))
        self.assertTrue(np.isfinite(db).all())
