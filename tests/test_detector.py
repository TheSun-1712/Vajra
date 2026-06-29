import unittest
import numpy as np
from vajra.config import SUBAP_PIXELS, MLA_GRID_SIZE
from vajra.detector import DetectorProcessor

class TestDetectorProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DetectorProcessor()
        self.grid_size = MLA_GRID_SIZE
        self.subap_w = SUBAP_PIXELS
        
    def test_background_subtraction(self):
        # Create a mock frame with high uniform background
        frame = np.ones((self.grid_size * self.subap_w, self.grid_size * self.subap_w)) * 50.0
        # Add some signal in the center of first subaperture
        frame[4:12, 4:12] += 100.0
        
        cleaned = self.processor.subtract_background(frame)
        
        # Guard border of subaperture 0 should be 0 after subtraction
        self.assertEqual(cleaned[0, 0], 0.0)
        self.assertEqual(cleaned[0, 15], 0.0)
        self.assertGreater(cleaned[8, 8], 50.0)

    def test_scintillation_detection(self):
        fluxes = np.ones(self.grid_size ** 2) * 1000.0
        illum = np.ones(self.grid_size ** 2)
        
        # Initial frame setup
        self.processor.prev_fluxes = fluxes.copy()
        
        # Create a localized drop in subaperture 5
        fluxes[5] = 200.0
        weights = self.processor.detect_scintillation(fluxes, illum)
        
        # Scintillation should down-weight subaperture 5
        self.assertLess(weights[5], 0.5)
        self.assertEqual(weights[0], 1.0)

    def test_local_gradient_outliers(self):
        centroids = np.zeros((self.grid_size ** 2, 2))
        illum = np.ones(self.grid_size ** 2)
        
        # Introduce a large outlier at subaperture 10
        centroids[10] = [5.0, -5.0]
        
        weights = self.processor.compute_local_gradient_outliers(centroids, illum)
        
        # Outlier subaperture should be down-weighted
        self.assertLess(weights[10], 0.5)
        self.assertEqual(weights[0], 1.0)
        
    def test_process_frame(self):
        frame = np.random.normal(10.0, 1.0, size=(self.grid_size * self.subap_w, self.grid_size * self.subap_w))
        for i in range(self.grid_size**2):
            row = i // self.grid_size
            col = i % self.grid_size
            frame[row*self.subap_w + 8, col*self.subap_w + 8] += 500.0
            
        illum = np.ones(self.grid_size ** 2)
        cents, fluxes, fwhms, confs = self.processor.process_frame(frame, illum)
        
        self.assertEqual(cents.shape, (self.grid_size ** 2, 2))
        self.assertEqual(fluxes.shape, (self.grid_size ** 2,))
        self.assertEqual(fwhms.shape, (self.grid_size ** 2,))
        self.assertEqual(confs.shape, (self.grid_size ** 2,))

if __name__ == '__main__':
    unittest.main()
