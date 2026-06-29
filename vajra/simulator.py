import numpy as np
import scipy.ndimage
from scipy.signal import fftconvolve
import hcipy
from vajra import config

class SolarGranulation:
    """Loads solar granulation structures from FITS files or procedurally generates them."""
    def __init__(self, size_pixels=128, scale=6.0, data_dir=None):
        self.size = size_pixels
        self.scale = scale
        self.data_dir = data_dir or config.SST_DKIST_DATA_DIR
        self.image = self._load_or_generate()
        
    def _load_or_generate(self):
        import glob
        import os
        from astropy.io import fits
        
        fits_files = glob.glob(os.path.join(self.data_dir, "*.fits"))
        if fits_files:
            try:
                # Load the first FITS file for granulation background
                fits_path = fits_files[0]
                with fits.open(fits_path) as hdul:
                    # In DKIST VBI Level 1 data, HDU 1 contains observation array of size (1, 4096, 4096)
                    raw_data = hdul[1].data[0] # Extract the 2D array (4096, 4096)
                    raw_data = raw_data.astype(np.float32)
                    # Normalize raw data
                    normalized = (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data) + 1e-8)
                    # Crop a patch of self.size x self.size from the center
                    h, w = normalized.shape
                    start_y = (h - self.size) // 2
                    start_x = (w - self.size) // 2
                    cropped = normalized[start_y:start_y+self.size, start_x:start_x+self.size]
                    # Standardize contrast
                    granulation = 0.25 + 0.75 * cropped
                    print(f"[VAJRA Simulator] Successfully loaded solar granulation from FITS: {os.path.basename(fits_path)}")
                    return granulation
            except Exception as e:
                print(f"[VAJRA Simulator] Failed to load FITS granulation ({e}), falling back to procedural.")
        
        return self._generate_procedural()

    def _generate_procedural(self):
        """Generates a realistic procedural solar granulation image.
        Granules have bright centers and thin, dark intergranular lanes.
        """
        # Start with random noise
        np.random.seed(42)
        noise = np.random.normal(size=(self.size, self.size))
        
        # Smooth noise to get granule characteristic scale
        smoothed = scipy.ndimage.gaussian_filter(noise, sigma=self.scale)
        
        # Create cell structure: make centers flatter/brighter, lanes sharper/darker
        normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
        granulation = np.exp(1.5 * normalized) - 1.0
        
        # Apply a secondary high-frequency texture for fine detail
        texture = scipy.ndimage.gaussian_filter(np.random.normal(size=(self.size, self.size)), sigma=1.5)
        granulation += 0.05 * texture
        
        # Normalize to [0, 1] range and scale contrast (mean ~ 0.6, peak ~ 1.0, lanes ~ 0.25)
        granulation = (granulation - granulation.min()) / (granulation.max() - granulation.min())
        granulation = 0.25 + 0.75 * granulation
        return granulation

    def get_patch(self, shift_x=0.0, shift_y=0.0):
        """Returns a shifted patch of the granulation.
        Simulates image movement or shift per subaperture.
        """
        if shift_x == 0.0 and shift_y == 0.0:
            return self.image
        return scipy.ndimage.shift(self.image, [shift_y, shift_x], mode='wrap')


class AOPipelineSimulator:
    """End-to-end physical optics simulation of the telescope AO system using HCIPy."""
    def __init__(self, mode="point"):
        """mode: 'point' for stellar PSF, 'solar' for convolved solar granulation."""
        self.mode = mode
        
        # 1. Setup grids and pupil
        self.pupil_grid = hcipy.make_pupil_grid(256, config.D_APERTURE)
        self.pupil_mask = hcipy.make_circular_aperture(config.D_APERTURE)(self.pupil_grid)
        
        # 2. Setup DM (aligned to Boston Micromachines Standard 276-3.5)
        # Create influence functions (Gaussian)
        self.influence_functions = hcipy.make_gaussian_influence_functions(
            self.pupil_grid, 
            config.DM_ACTUATORS_SIDE, 
            config.D_APERTURE / (config.DM_ACTUATORS_SIDE - 1)
        )
        self.dm = hcipy.DeformableMirror(self.influence_functions)
        
        # 3. Setup Microlens Array (MLA) geometry
        self.mla_grid_size = config.MLA_GRID_SIZE
        self.subap_pitch = config.MLA_PITCH
        self.subaps_pos = self._get_subaperture_positions()
        
        # 4. Generate solar granulation target if in solar mode
        self.granulation = SolarGranulation(size_pixels=128, scale=6.0)
        
        self.atmosphere_layers = []
        for h, w, v_spd, v_dir in zip(config.LAYER_HEIGHTS, config.LAYER_WEIGHTS, config.WIND_SPEEDS, config.WIND_DIRECTIONS):
            cn2 = w * hcipy.Cn_squared_from_fried_parameter(config.R0_500, config.LAMBDA_SENSING)
            velocity = np.array([v_spd * np.cos(v_dir), v_spd * np.sin(v_dir)])
            # HCIPy atmospheric layer
            layer = hcipy.InfiniteAtmosphericLayer(self.pupil_grid, cn2, config.OUTER_SCALE, velocity, h)
            self.atmosphere_layers.append(layer)
            
        # Zernike expansion basis on the pupil grid for validation
        self.zernike_basis = hcipy.make_zernike_basis(config.ZERNIKE_MODES_MAX + 1, config.D_APERTURE, self.pupil_grid, starting_mode=2)
        
        # Build reference grid mappings for subapertures
        self.subap_masks = []
        for x, y in self.subaps_pos:
            mask = self._get_subap_mask(x, y)
            self.subap_masks.append(mask)

    def _get_subaperture_positions(self):
        """Returns the center coordinates of all subapertures in the grid."""
        half_grid = (self.mla_grid_size - 1) / 2.0
        coords = []
        for r in range(self.mla_grid_size):
            for c in range(self.mla_grid_size):
                x = (c - half_grid) * self.subap_pitch
                y = (r - half_grid) * self.subap_pitch
                coords.append((x, y))
        return coords

    def _get_subap_mask(self, cx, cy):
        """Returns a binary mask on the pupil grid for a subaperture centered at (cx, cy)."""
        x_coords = self.pupil_grid.x
        y_coords = self.pupil_grid.y
        mask = (np.abs(x_coords - cx) <= self.subap_pitch / 2) & (np.abs(y_coords - cy) <= self.subap_pitch / 2)
        # Intersect with the circular pupil mask to account for vignetting/pupil boundary
        return mask * self.pupil_mask

    def evolve_atmosphere(self, dt):
        """Evolves the atmospheric screens by time step dt."""
        for layer in self.atmosphere_layers:
            layer.evolve_until(dt)

    def get_phase_screen(self):
        """Returns the integrated atmospheric phase screen on the pupil grid (in radians at sensing wavelength)."""
        phase = np.zeros(self.pupil_grid.size)
        for layer in self.atmosphere_layers:
            phase += layer.phase_for(config.LAMBDA_SENSING)
        return phase * self.pupil_mask

    def generate_hartmannogram(self, dm_commands=None, flux_nominal=config.PHOTON_FLUX_NOMINAL, 
                              background_pedestal=config.SKY_BACKGROUND_PEDESTAL, registration_offset=(0.0,0.0)):
        """Generates a raw detector Hartmannogram image convolved with solar granulation or stellar point sources.
        
        dm_commands: Actuator commands to apply to the DM.
        flux_nominal: Target photons per subaperture.
        background_pedestal: Uniform background pedestal value.
        registration_offset: Global thermal/mechanical MLA alignment drift (pixels).
        """
        # Apply DM shape if commanded
        if dm_commands is not None:
            self.dm.actuators = dm_commands
        else:
            self.dm.actuators = np.zeros(config.DM_ACTUATORS_TOTAL)
            
        # Get atmospheric phase and DM correction phase
        atmosphere_phase = self.get_phase_screen()
        dm_phase = self.dm.phase_for(config.LAMBDA_SENSING)
        residual_phase = (atmosphere_phase + dm_phase) * self.pupil_mask
        
        # Initialize detector frame (16x16 subapertures * 16x16 pixels/subap = 256x256 pixels)
        subap_w = config.SUBAP_PIXELS
        det_size = self.mla_grid_size * subap_w
        detector_frame = np.zeros((det_size, det_size))
        
        # Subaperture illumination fractions (for dynamic pupil masking)
        illumination_fractions = []
        
        for i, (cx, cy) in enumerate(self.subaps_pos):
            mask = self.subap_masks[i]
            area_fraction = np.sum(mask) / np.sum(self.pupil_grid.x**2 + self.pupil_grid.y**2 <= (self.subap_pitch/2)**2)
            illumination_fractions.append(area_fraction)
            
            row = i // self.mla_grid_size
            col = i % self.mla_grid_size
            
            # Row/col slice on detector
            r_start = row * subap_w
            c_start = col * subap_w
            
            # If the subaperture has no light, it's black (vignetted)
            if area_fraction < 0.1:
                detector_frame[r_start:r_start+subap_w, c_start:c_start+subap_w] = background_pedestal
                continue
                
            # Extract local complex field
            mask_bool = (mask > 0)
            local_phase = residual_phase[mask_bool]
            
            # Average phase slope (tip-tilt) across the subaperture
            local_x = self.pupil_grid.x[mask_bool] - cx
            local_y = self.pupil_grid.y[mask_bool] - cy
            
            # Estimate slopes directly from the simulated phase screen for reference/ground truth
            if len(local_phase) > 2:
                A = np.column_stack([local_x, local_y, np.ones_like(local_x)])
                slopes = np.linalg.lstsq(A, local_phase, rcond=None)[0]
                slope_x, slope_y = slopes[0], slopes[1]
            else:
                slope_x, slope_y = 0.0, 0.0
                
            # Perform physical optics propagation to create subaperture PSF
            # Physical scaling: 1 rad/m slope shift equates to ~0.11 WFS pixels
            pixel_shift_x = slope_x * 0.11
            pixel_shift_y = slope_y * 0.11
            
            # Incorporate registration drift (Problem 19)
            pixel_shift_x += registration_offset[0]
            pixel_shift_y += registration_offset[1]
            
            # Defensive clipping to prevent spots shifting completely out of the subaperture frame
            max_shift = (subap_w / 2.0) - 1.5
            pixel_shift_x = np.clip(pixel_shift_x, -max_shift, max_shift)
            pixel_shift_y = np.clip(pixel_shift_y, -max_shift, max_shift)
            
            # Generate the local subaperture PSF
            psf_grid = np.arange(subap_w) - (subap_w - 1) / 2.0
            x_g, y_g = np.meshgrid(psf_grid, psf_grid)
            
            # PSF under aberrations: local shift + elongation/smearing under high phase variance
            phase_var = np.var(local_phase) if len(local_phase) > 0 else 0.0
            psf_width = 1.2 + 0.4 * phase_var
            
            r2 = (x_g - pixel_shift_x)**2 + (y_g - pixel_shift_y)**2
            psf = np.exp(-r2 / (2 * psf_width**2))
            
            psf_sum = psf.sum()
            if psf_sum > 0:
                psf /= psf_sum
            else:
                psf = np.zeros_like(psf)
                psf[subap_w // 2, subap_w // 2] = 1.0
            
            # Calculate final subaperture target scene based on mode
            if self.mode == "solar":
                # Convolve PSF with the solar granulation scene
                granulation_scene = self.granulation.get_patch()
                scene_cropped = granulation_scene[
                    (row*8) % (128-subap_w) : (row*8) % (128-subap_w) + subap_w,
                    (col*8) % (128-subap_w) : (col*8) % (128-subap_w) + subap_w
                ]
                subimage = fftconvolve(scene_cropped, psf, mode='same')
            else:
                # Point-source mode: subimage is simply the local star PSF
                subimage = psf.copy()
                
            # Scale flux according to illumination fraction and target nominal flux
            subap_flux = flux_nominal * area_fraction
            
            # Scintillation simulation: apply transient flux dropouts (Problem 18)
            scintillation_factor = 1.0
            if np.random.rand() < 0.02: # 2% chance of transient drop
                scintillation_factor = 0.2 + 0.3 * np.random.rand()
            subap_flux *= scintillation_factor
            
            subimage = subimage / subimage.sum() * subap_flux
            
            # Add uniform background pedestal (Problem 24)
            subimage += background_pedestal
            
            # Apply Photon Noise (Poisson) and Detector Read Noise (Gaussian)
            noisy_subimage = np.random.poisson(np.maximum(subimage, 0.0))
            noisy_subimage = noisy_subimage + np.random.normal(0, config.READ_NOISE, size=subimage.shape)
            
            detector_frame[r_start:r_start+subap_w, c_start:c_start+subap_w] = np.maximum(noisy_subimage, 0.0)
            
        return detector_frame, np.array(illumination_fractions)

    def get_ground_truth_zernikes(self):
        """Extracts the true input atmospheric phase decomposed into the first Zernike modes."""
        phase = self.get_phase_screen()
        z_coeffs = []
        for z_mode in self.zernike_basis:
            coeff = np.sum(phase * z_mode * self.pupil_mask) / np.sum(z_mode**2 * self.pupil_mask)
            z_coeffs.append(coeff)
        return np.array(z_coeffs)
