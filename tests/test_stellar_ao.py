import unittest
import numpy as np
import torch
from vajra.reconstructor import TomographicReconstructor, WavefrontReconstructor
from vajra.science import SpeckleNullingController
from vajra.controller import LQGStructuralTracker, ActuatorController
from vajra import config
from run_stellar_vajra import run_stellar_loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestStellarAOChallenges(unittest.TestCase):
    def test_tomographic_reconstructor(self):
        # Create 4 reconstructors for the LGS directions
        num_lgs = 4
        pupil_mask = np.ones((256, 256)).flatten()
        pupil_grid = type('PupilGrid', (object,), {'size': 256*256})()
        subaps_pos = [(0, 0)] * config.MLA_GRID_SIZE**2
        subap_masks = [np.ones((256, 256)).flatten()] * config.MLA_GRID_SIZE**2
        
        # Mock class for WavefrontReconstructor to bypass complex optics init
        class MockReconstructor:
            def __init__(self):
                self.num_zernikes = config.ZERNIKE_MODES_MAX
            def reconstruct_wls(self, centroids, confidences, illum):
                # Return ones to verify scaling behavior
                return np.ones(self.num_zernikes), None
                
        reconstructors = [MockReconstructor() for _ in range(num_lgs)]
        tomo = TomographicReconstructor(reconstructors)
        
        centroids_list = [np.zeros((config.MLA_GRID_SIZE**2, 2)) for _ in range(num_lgs)]
        confidences_list = [np.ones(config.MLA_GRID_SIZE**2) for _ in range(num_lgs)]
        illumination_list = [np.ones(config.MLA_GRID_SIZE**2) for _ in range(num_lgs)]
        
        z_tomo = tomo.reconstruct_tomography(centroids_list, confidences_list, illumination_list)
        
        self.assertEqual(z_tomo.shape, (config.ZERNIKE_MODES_MAX,))
        # Check that high-order cone scaling scaling was successfully applied (> 1.0)
        self.assertGreater(z_tomo[10], 1.0)

    def test_speckle_nulling(self):
        nuller = SpeckleNullingController(num_actuators=config.DM_ACTUATORS_TOTAL)
        
        # Test probe generation (4 phases)
        p0 = nuller.generate_probe(0)
        p1 = nuller.generate_probe(1)
        self.assertEqual(p0.shape, (config.DM_ACTUATORS_TOTAL,))
        self.assertEqual(p1.shape, (config.DM_ACTUATORS_TOTAL,))
        
        # Ensure probes are orthoconjugate phase-shifted (different values)
        self.assertFalse(np.array_equal(p0, p1))
        
        # Test nulling calculation
        # 4 mock focal plane intensity maps
        i0 = np.ones((32, 32)) * 1.5
        i1 = np.ones((32, 32)) * 1.2
        i2 = np.ones((32, 32)) * 0.9
        i3 = np.ones((32, 32)) * 0.6
        
        null_cmds = nuller.compute_nulling_commands([i0, i1, i2, i3])
        self.assertEqual(null_cmds.shape, (config.DM_ACTUATORS_TOTAL,))

    def test_lqg_vibration_tracker(self):
        tracker = LQGStructuralTracker(dt=0.001)
        
        # Feed steady measured tip-tilt at 52 Hz
        # Simulate tip-tilt measurement containing a 52 Hz vibration
        for i in range(100):
            t = i * 0.001
            z_measured = np.sin(2.0 * np.pi * 52.0 * t)
            tracked_f = tracker.update_and_track(z_measured)
            
        # The tracked frequency should start moving toward 52 Hz
        self.assertTrue(30.0 <= tracked_f <= 80.0)

    def test_stellar_loop_benchmark(self):
        # Run a short stellar closed-loop run (5 frames)
        metrics = run_stellar_loop(
            n_frames=5, use_tomography=True, use_nulling=True, use_lqg=True
        )
        self.assertIn("strehl_history", metrics)
        self.assertEqual(len(metrics["strehl_history"]), 5)

if __name__ == "__main__":
    unittest.main()
