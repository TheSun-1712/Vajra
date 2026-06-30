import os
import numpy as np
from vajra import config

class StarCatalogQuery:
    """Queries astronomical catalogs (like Gaia DR3) to fetch guide star characteristics."""
    def __init__(self):
        self.enabled = True
        try:
            from astroquery.gaia import Gaia
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            self.Gaia = Gaia
            self.SkyCoord = SkyCoord
            self.u = u
        except ImportError:
            self.enabled = False
            print("[Dataset Loader] astroquery or astropy not installed. Gaia queries will fall back to emulation.")

    def query_nearest_guide_star(self, ra_deg, dec_deg, radius_arcmin=5.0):
        """Finds the brightest star near the given coordinates."""
        if not self.enabled:
            return self._emulate_query(ra_deg, dec_deg)
            
        try:
            coord = self.SkyCoord(ra=ra_deg*self.u.degree, dec=dec_deg*self.u.degree, frame='icrs')
            j = self.Gaia.cone_search_async(coord, radius=radius_arcmin*self.u.arcmin)
            results = j.get_results()
            
            if len(results) == 0:
                print(f"[Dataset Loader] No stars found within {radius_arcmin} arcmin of coordinates ({ra_deg}, {dec_deg}).")
                return self._emulate_query(ra_deg, dec_deg)
                
            # Sort by brightness (G-band magnitude phot_g_mean_mag)
            results.sort('phot_g_mean_mag')
            brightest = results[0]
            
            star_info = {
                "source_id": int(brightest['source_id']),
                "ra": float(brightest['ra']),
                "dec": float(brightest['dec']),
                "phot_g_mean_mag": float(brightest['phot_g_mean_mag']),
                "parallax": float(brightest['parallax']) if not np.isnan(brightest['parallax']) else 0.0,
                "emulated": False
            }
            return star_info
        except Exception as e:
            print(f"[Dataset Loader] Gaia query encountered error: {e}. Falling back to emulation.")
            return self._emulate_query(ra_deg, dec_deg)

    def calculate_photon_flux(self, g_mag, exposure_time_ms=1.0, aperture_diameter=2.0):
        """Calculates expected photons per subaperture per frame based on magnitude.
        Uses a zero-point flux value (Vega zero-point in G-band is approx 3e10 photons/s/m2).
        """
        # Zero-point flux in G-band for a mag 0 star: ~3e9 photons/s/m^2/nm
        # Integrated over standard WFS bandwidth (approx 100 nm): ~3e11 photons/s/m^2
        zero_point_flux = 3e11 
        
        # Total aperture area (m^2)
        aperture_area = np.pi * (aperture_diameter / 2.0)**2
        
        # Area of a single subaperture in the 16x16 grid
        subap_area = aperture_area / (config.MLA_GRID_SIZE ** 2)
        
        # Flux at subaperture for mag G = g_mag
        photons_per_sec = subap_area * zero_point_flux * (10 ** (-0.4 * g_mag))
        
        # Total photons for exposure duration
        photons = photons_per_sec * (exposure_time_ms / 1000.0)
        return np.maximum(photons, 1.0)

    def _emulate_query(self, ra_deg, dec_deg):
        """Emulates a query search when external APIs are offline."""
        # Standard calibration stars
        # Mock search: return a nearby bright guide star based on coordinates
        seed = int(ra_deg + dec_deg) % 100
        np.random.seed(seed)
        
        # Procedurally generate a star close to target coords
        offset_ra = np.random.normal(0, 0.01)
        offset_dec = np.random.normal(0, 0.01)
        # Mag between 4.0 (very bright) and 12.0 (faint)
        g_mag = 5.0 + np.random.rand() * 7.0 
        
        star_info = {
            "source_id": int(100000000 + seed),
            "ra": ra_deg + offset_ra,
            "dec": dec_deg + offset_dec,
            "phot_g_mean_mag": g_mag,
            "parallax": 10.0 + np.random.rand() * 50.0,
            "emulated": True
        }
        return star_info


class TelemetryReplayer:
    """Parses Shack-Hartmann WFS fits telemetry sequences from standard archives (Keck, Subaru)."""
    def __init__(self, data_dir="data/stellar_telemetry/"):
        self.data_dir = data_dir
        self.has_data = False
        self.file_list = []
        self.current_idx = 0
        
        if os.path.exists(data_dir):
            import glob
            self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
            if self.file_list:
                self.has_data = True
                print(f"[Dataset Loader] Telemetry replayer initialized with {len(self.file_list)} files.")

    def load_telemetry_frame(self):
        """Loads and returns centroid slopes and voltages from the current index file.
        Falls back to realistic random-walk turbulence simulation if no files exist.
        """
        if not self.has_data:
            return self._emulate_telemetry_frame()
            
        try:
            from astropy.io import fits
            filepath = self.file_list[self.current_idx]
            
            with fits.open(filepath) as hdul:
                # Expect standard Keck/Subaru telemetry format
                # Extension 0 or 1: centroid slopes (N_frames, 512)
                # Extension 2: DM voltages (N_frames, 276)
                centroids = hdul[0].data
                dm_commands = hdul[1].data if len(hdul) > 1 else np.zeros(config.DM_ACTUATORS_TOTAL)
                
            self.current_idx = (self.current_idx + 1) % len(self.file_list)
            return centroids, dm_commands, False
        except Exception as e:
            print(f"[Dataset Loader] Failed to load telemetry file: {e}. Emulating frame.")
            return self._emulate_telemetry_frame()

    def _emulate_telemetry_frame(self):
        """Generates realistic telemetry frames representing atmospheric seeing."""
        # Generate random walk centroids matching 512-dim WFS slope array
        num_subaps = config.MLA_GRID_SIZE ** 2
        centroids = np.random.normal(0, 0.15, size=(num_subaps, 2))
        
        # Generate correlating DM commands
        dm_commands = np.random.normal(0, 1e-7, size=config.DM_ACTUATORS_TOTAL)
        return centroids, dm_commands, True
