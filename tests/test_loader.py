import unittest
import numpy as np
from vajra.dataset_loader import StarCatalogQuery, TelemetryReplayer
from vajra import config

class TestDatasetLoader(unittest.TestCase):
    def test_star_catalog_emulation(self):
        query = StarCatalogQuery()
        # Force emulator mode
        query.enabled = False
        star = query.query_nearest_guide_star(ra_deg=100.0, dec_deg=25.0)
        
        self.assertTrue(star["emulated"])
        self.assertIn("phot_g_mean_mag", star)
        self.assertGreaterEqual(star["phot_g_mean_mag"], 4.0)

    def test_photon_flux_calculation(self):
        query = StarCatalogQuery()
        flux_bright = query.calculate_photon_flux(5.0)
        flux_faint = query.calculate_photon_flux(10.0)
        
        # A brighter star should produce more photons
        self.assertGreater(flux_bright, flux_faint)
        self.assertGreater(flux_bright, 0)

    def test_telemetry_replayer_emulation(self):
        replayer = TelemetryReplayer(data_dir="empty_directory_mock")
        centroids, dm_commands, is_emulated = replayer.load_telemetry_frame()
        
        self.assertTrue(is_emulated)
        self.assertEqual(centroids.shape, (config.MLA_GRID_SIZE**2, 2))
        self.assertEqual(dm_commands.shape, (config.DM_ACTUATORS_TOTAL,))

if __name__ == "__main__":
    unittest.main()
