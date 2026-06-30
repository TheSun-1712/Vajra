import unittest
import numpy as np
from vajra.wfs_less import SPGDOptimizer
from vajra import config

class TestWFSLessOptimizer(unittest.TestCase):
    def test_sharpness_metric(self):
        optimizer = SPGDOptimizer()
        
        # Test with a uniform flat image (low sharpness)
        flat_img = np.ones((32, 32))
        metric_flat = optimizer.compute_sharpness_metric(flat_img)
        
        # Test with a sharp delta-function spot (high sharpness)
        sharp_img = np.zeros((32, 32))
        sharp_img[16, 16] = 1000.0
        metric_sharp = optimizer.compute_sharpness_metric(sharp_img)
        
        self.assertGreater(metric_sharp, metric_flat)

    def test_perturbation_generation(self):
        optimizer = SPGDOptimizer()
        pert = optimizer.generate_perturbation()
        
        self.assertEqual(pert.shape, (config.DM_ACTUATORS_TOTAL,))
        # Assert each perturbation is +/- amplitude
        for val in pert:
            self.assertAlmostEqual(np.abs(val), optimizer.delta_u_amplitude)

    def test_calculate_update(self):
        optimizer = SPGDOptimizer()
        pert = np.random.choice([-1.0, 1.0], size=config.DM_ACTUATORS_TOTAL) * 1e-8
        
        # J_plus > J_minus should generate positive updates along perturbation direction
        update = optimizer.calculate_update(pert, metric_plus=10.0, metric_minus=5.0)
        
        # Confirm sign matching
        np.testing.assert_array_equal(np.sign(update), np.sign(pert))

if __name__ == "__main__":
    unittest.main()
