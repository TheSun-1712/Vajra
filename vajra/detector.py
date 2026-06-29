import numpy as np
from vajra import config

class DetectorProcessor:
    """Processes raw detector frames to extract subaperture fluxes, centroids, sizes, and confidence scores."""
    def __init__(self):
        self.grid_size = config.MLA_GRID_SIZE
        self.subap_w = config.SUBAP_PIXELS
        self.num_subaps = self.grid_size ** 2
        
        # Track registration correction
        self.registration_correction = np.zeros(2) # [dx, dy] in pixels
        self.long_term_centroids = [] # List of history for registration drift calibration
        self.registration_alpha = 0.05 # Smoothing factor for drift update
        
        # Track previous fluxes for scintillation detection
        self.prev_fluxes = None

    def process_frame(self, raw_frame, illumination_fractions):
        """Processes a raw detector frame.
        
        raw_frame: (256, 256) array of pixel intensities.
        illumination_fractions: (256,) array indicating which subapertures are illuminated.
        
        Returns:
            centroids: (256, 2) array of raw spot deviations.
            fluxes: (256,) array of subaperture fluxes.
            fwhms: (256,) array of spot widths.
            confidences: (256,) array of confidence weights.
        """
        # 1. Problem 24: Real-time background pedestal subtraction
        clean_frame = self.subtract_background(raw_frame)
        
        centroids = np.zeros((self.num_subaps, 2))
        fluxes = np.zeros(self.num_subaps)
        fwhms = np.zeros(self.num_subaps)
        confidences = np.zeros(self.num_subaps)
        
        # Coordinate grids for centroid computation
        psf_grid = np.arange(self.subap_w) - (self.subap_w - 1) / 2.0
        x_grid, y_grid = np.meshgrid(psf_grid, psf_grid)
        
        # 2. Extract subaperture stats
        for i in range(self.num_subaps):
            if illumination_fractions[i] < 0.1:
                continue
                
            row = i // self.grid_size
            col = i % self.grid_size
            
            r_start = row * self.subap_w
            c_start = col * self.subap_w
            
            subimage = clean_frame[r_start:r_start+self.subap_w, c_start:c_start+self.subap_w]
            
            # Total flux in this subaperture
            flux = np.sum(subimage)
            fluxes[i] = flux
            
            if flux <= 10.0: # Faint or dark subaperture
                continue
                
            # Compute centroid (center of mass)
            cx = np.sum(subimage * x_grid) / flux
            cy = np.sum(subimage * y_grid) / flux
            
            # Apply registration correction (Problem 19)
            centroids[i, 0] = cx - self.registration_correction[0]
            centroids[i, 1] = cy - self.registration_correction[1]
            
            # Compute variance and FWHM
            var_x = np.sum(subimage * (x_grid - cx)**2) / flux
            var_y = np.sum(subimage * (y_grid - cy)**2) / flux
            fwhm = 2.355 * np.sqrt(np.maximum(var_x + var_y, 1e-5) / 2.0)
            fwhms[i] = fwhm

        # 3. Problem 18: Scintillation intensity-normalized confidence weighting
        scintillation_weights = self.detect_scintillation(fluxes, illumination_fractions)
        
        # 4. Problem 2: Photon-statistics confidence weighting (Layer 1a)
        # Confidence decays under low flux and large FWHM
        flux_confidence = fluxes / (fluxes + config.READ_NOISE**2 * (self.subap_w**2))
        
        # Smeared spot penalty
        nominal_fwhm = 1.2
        fwhm_confidence = 1.0 / (1.0 + np.maximum(fwhms - nominal_fwhm, 0.0)**2)
        
        # Neighbor-outlier deviation confidence (local slope consistency check)
        outlier_weights = self.compute_local_gradient_outliers(centroids, illumination_fractions)
        
        # Combine all confidences
        for i in range(self.num_subaps):
            if illumination_fractions[i] < 0.1:
                continue
            conf_val = flux_confidence[i] * fwhm_confidence[i] * scintillation_weights[i] * outlier_weights[i]
            conf_val = np.clip(conf_val, 1e-4, 1.0)
            confidences[i] = conf_val
            
        # 5. Problem 19: Registration drift accumulation
        self.accumulate_registration_history(centroids, illumination_fractions)
        
        self.prev_fluxes = fluxes.copy()
        
        return centroids, fluxes, fwhms, confidences

    def subtract_background(self, raw_frame):
        """Estimates and subtracts a uniform background pedestal per subaperture.
        Uses the border pixels (guard zone) of each subaperture box.
        """
        clean_frame = raw_frame.copy()
        for i in range(self.num_subaps):
            row = i // self.grid_size
            col = i % self.grid_size
            
            r_start = row * self.subap_w
            c_start = col * self.subap_w
            
            subimage = raw_frame[r_start:r_start+self.subap_w, c_start:c_start+self.subap_w]
            
            # Extract border pixels (top, bottom, left, right edges)
            border_pixels = np.concatenate([
                subimage[0, :],      # Top row
                subimage[-1, :],     # Bottom row
                subimage[1:-1, 0],   # Left column (excluding corners)
                subimage[1:-1, -1]   # Right column (excluding corners)
            ])
            
            # Estimate pedestal as the median of border pixels (robust to spot intrusion)
            pedestal = np.median(border_pixels)
            
            # Subtract pedestal and clip at 0
            clean_frame[r_start:r_start+self.subap_w, c_start:c_start+self.subap_w] = np.maximum(subimage - pedestal, 0.0)
            
        return clean_frame

    def detect_scintillation(self, fluxes, illumination_fractions):
        """Identifies transient localized scintillation dropouts.
        Compares flux to previous frame and surrounding neighbors.
        """
        weights = np.ones(self.num_subaps)
        if self.prev_fluxes is None:
            return weights
            
        for i in range(self.num_subaps):
            if illumination_fractions[i] < 0.1:
                continue
                
            # Previous frame ratio
            prev_flux = self.prev_fluxes[i]
            if prev_flux > 50.0:
                ratio = fluxes[i] / prev_flux
                # Problem 18: Transient drop detection (z-score/threshold outlier)
                if ratio < 0.7:  # More than 30% transient drop
                    weights[i] = 0.3 # Significantly down-weight this subaperture
                    
        return weights

    def compute_local_gradient_outliers(self, centroids, illumination_fractions):
        """Problem 2: Computes outlier z-scores comparing each subaperture centroid to its spatial neighbors."""
        weights = np.ones(self.num_subaps)
        for i in range(self.num_subaps):
            if illumination_fractions[i] < 0.1:
                continue
                
            row = i // self.grid_size
            col = i % self.grid_size
            
            # Find 4-connected spatial neighbors
            neighbor_indices = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    idx = nr * self.grid_size + nc
                    if illumination_fractions[idx] >= 0.1:
                        neighbor_indices.append(idx)
                        
            if not neighbor_indices:
                continue
                
            # Average neighbor centroids
            neighbor_cents = centroids[neighbor_indices]
            mean_c = np.mean(neighbor_cents, axis=0)
            std_c = np.std(neighbor_cents, axis=0) + 1e-3
            
            # Compute Z-score of local deviation
            z_score = np.abs(centroids[i] - mean_c) / std_c
            max_z = np.max(z_score)
            
            # Decay confidence if z-score > 2.5
            if max_z > 2.5:
                weights[i] = 1.0 / (1.0 + (max_z - 2.5)**2)
                
        return weights

    def accumulate_registration_history(self, centroids, illumination_fractions):
        """Problem 19: Long-term MLA-to-detector registration drift tracking."""
        active_centroids = centroids[illumination_fractions >= 0.1]
        if len(active_centroids) == 0:
            return
            
        mean_drift = np.mean(active_centroids, axis=0)
        self.long_term_centroids.append(mean_drift)
        
        # Perform rolling average over last 1200 frames (roughly 1.2s at 1kHz, mocked for drift speed)
        if len(self.long_term_centroids) > 500:
            self.long_term_centroids.pop(0)
            
        # Global common-mode translation
        rolling_mean = np.mean(self.long_term_centroids, axis=0)
        
        # Exponential smoothing update of correction grid offset
        self.registration_correction += self.registration_alpha * (rolling_mean - self.registration_correction)
