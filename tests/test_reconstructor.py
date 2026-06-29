import unittest
import numpy as np
import torch
import hcipy
from vajra import config
from vajra.reconstructor import WavefrontReconstructor, Layer0AttentionCNN, UARCBayesianReconstructor

class TestWavefrontReconstructor(unittest.TestCase):
    def setUp(self):
        # Create minimal grid for testing speed
        self.pupil_grid = hcipy.make_pupil_grid(64, config.D_APERTURE)
        self.pupil_mask = hcipy.make_circular_aperture(config.D_APERTURE)(self.pupil_grid)
        
        # Subapertures positions
        self.grid_size = config.MLA_GRID_SIZE
        self.subap_pitch = config.MLA_PITCH
        half_grid = (self.grid_size - 1) / 2.0
        self.subaps_pos = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x = (c - half_grid) * self.subap_pitch
                y = (r - half_grid) * self.subap_pitch
                self.subaps_pos.append((x, y))
                
        # Subaperture masks (simplification for testing)
        self.subap_masks = []
        for cx, cy in self.subaps_pos:
            mask = (np.abs(self.pupil_grid.x - cx) <= self.subap_pitch / 2) & \
                   (np.abs(self.pupil_grid.y - cy) <= self.subap_pitch / 2)
            self.subap_masks.append(mask * self.pupil_mask)
            
        self.reconstructor = WavefrontReconstructor(
            self.pupil_grid, self.pupil_mask, self.subaps_pos, self.subap_masks
        )

    def test_geometry_matrix_size(self):
        # G matrix should map 2 * subapertures to zernikes
        expected_shape = (2 * len(self.subaps_pos), config.ZERNIKE_MODES_MAX)
        self.assertEqual(self.reconstructor.G_full.shape, expected_shape)

    def test_wls_reconstruct(self):
        centroids = np.zeros((len(self.subaps_pos), 2))
        centroids[:, 0] = 0.5 # constant X slope
        
        confidences = np.ones(len(self.subaps_pos))
        illum = np.ones(len(self.subaps_pos))
        
        z_wls, uarc_vars = self.reconstructor.reconstruct_wls(centroids, confidences, illum)
        
        self.assertEqual(len(z_wls), config.ZERNIKE_MODES_MAX)
        self.assertEqual(len(uarc_vars), config.ZERNIKE_MODES_MAX)
        self.assertNotEqual(z_wls[0], 0.0)

    def test_bnn_reconstruct(self):
        centroids = np.zeros((len(self.subaps_pos), 2))
        confidences = np.ones(len(self.subaps_pos))
        
        z_bnn, bnn_vars = self.reconstructor.reconstruct_bnn(centroids, confidences)
        
        self.assertEqual(len(z_bnn), config.ZERNIKE_MODES_MAX)
        self.assertEqual(len(bnn_vars), config.ZERNIKE_MODES_MAX)
        self.assertTrue(np.all(bnn_vars > 0))

    def test_cnn_sensor_forward(self):
        cnn = Layer0AttentionCNN()
        # Input shape: (Batch=2, Ch=1, H=256, W=256)
        x = torch.zeros(2, 1, 256, 256)
        out = cnn(x)
        self.assertEqual(out.shape, (2, config.ZERNIKE_MODES_MAX))
        
        # Test heteroscedastic confidence head outputs
        zern, log_var = cnn.predict_with_confidence(x)
        self.assertEqual(zern.shape, (2, config.ZERNIKE_MODES_MAX))
        self.assertEqual(log_var.shape, (2, 256))

    def test_bnn_mc_dropout(self):
        # BNN inputs are slopes (256*2 = 512) + confidences (256) = 768
        bnn = UARCBayesianReconstructor(input_dim=768, output_dim=10)
        x = torch.zeros(2, 768)
        means, vars_ = bnn.forward_mc(x, num_samples=5)
        self.assertEqual(means.shape, (2, 10))
        self.assertEqual(vars_.shape, (2, 10))

if __name__ == '__main__':
    unittest.main()
