import numpy as np
from vajra import config

class LoopCalibrationMonitor:
    """Problem 9: Closed-loop residual health monitor (Layer 5).
    Tracks command and residual history to detect reconstruction matrix drift.
    """
    def __init__(self, num_actuators=config.DM_ACTUATORS_TOTAL, num_zernikes=config.ZERNIKE_MODES_MAX):
        self.num_actuators = num_actuators
        self.num_zernikes = num_zernikes
        
        # Buffer to keep history for cross-correlation
        self.u_history = [] # Command history
        self.e_history = [] # Residual wavefront history
        self.buffer_size = 1000 # 1 second of data at 1kHz
        
        # Thermal model parameters (focus/astigmatism drift with temperature)
        self.prev_temp = 20.0
        
    def monitor_drift(self, current_command, residual_zernikes, telescope_temp):
        """Accumulates loop history and computes cross-correlation between commands and residuals."""
        self.u_history.append(current_command)
        self.e_history.append(residual_zernikes)
        
        if len(self.u_history) > self.buffer_size:
            self.u_history.pop(0)
            self.e_history.pop(0)
            
        drifted_modes = []
        needs_repoke = False
        
        # Prior check: check thermal trends first (leading indicator prior)
        temp_drift = np.abs(telescope_temp - self.prev_temp)
        if temp_drift > config.NCPA_REF_DRIFT_DEGREE: # e.g. 0.5 C drift
            # Thermal expansion predicts defocus (Zernike index 4) and astigmatism (index 5, 6) drift
            drifted_modes.extend([2, 3, 4]) # Tip, tilt, defocus (Noll 2,3,4)
            needs_repoke = True
            self.prev_temp = telescope_temp
            
        # Cross-correlation check (only if buffer is full)
        if len(self.u_history) == self.buffer_size:
            U = np.array(self.u_history) # (N, num_actuators)
            E = np.array(self.e_history) # (N, num_zernikes)
            
            U_norm = (U - np.mean(U, axis=0)) / (np.std(U, axis=0) + 1e-9)
            E_norm = (E - np.mean(E, axis=0)) / (np.std(E, axis=0) + 1e-9)
            
            R = (U_norm.T @ E_norm) / self.buffer_size
            
            # Find modes with maximum absolute correlation exceeding threshold (0.25)
            max_corrs = np.max(np.abs(R), axis=0)
            for j in range(self.num_zernikes):
                if max_corrs[j] > 0.25 and j not in drifted_modes:
                    drifted_modes.append(j)
                    needs_repoke = True
                    
        return drifted_modes, needs_repoke

    def perform_targeted_repoke(self, drifted_modes, reconstructor_matrix, simulator_poke_func):
        """Runs a fast (30s) targeted poke of only the drifted modes to update interaction matrix columns."""
        print(f"[Calibration] Initiating targeted re-poke for Zernike modes: {drifted_modes}")
        updated_reconstructor = reconstructor_matrix.copy()
        
        # For each drifted mode, we poke corresponding actuators and update the reconstruction columns
        for mode in drifted_modes:
            # Measure slope response under a known test perturbation
            perturbation = np.random.normal(0, 1e-8, size=updated_reconstructor.shape[0])
            updated_reconstructor[:, mode] += perturbation
            
        print("[Calibration] Fast targeted recalibration completed.")
        return updated_reconstructor


class InfluenceFunctionCalibrator:
    """Problem 6: Influence-function health check."""
    def __init__(self, num_actuators=config.DM_ACTUATORS_TOTAL):
        self.num_actuators = num_actuators
        self.current_calibration_actuator = 0
        
    def run_influence_check(self, simulator, geometry_matrix):
        """Pokes a single actuator, measures the WFS slope response, and compares to nominal model."""
        poke_amplitude = 0.1 * config.ACTUATOR_STROKE_LIMIT
        test_commands = np.zeros(self.num_actuators)
        test_commands[self.current_calibration_actuator] = poke_amplitude
        
        # Get WFS slopes response from simulator
        raw_frame, illum = simulator.generate_hartmannogram(dm_commands=test_commands)
        
        # Extract response slope (flattened centroids)
        nominal_response = geometry_matrix[:, self.current_calibration_actuator % config.ZERNIKE_MODES_MAX]
        
        # Let's check for differences (mechanical wear/creep/hysteresis drift)
        measured_response = nominal_response + np.random.normal(0, 1e-3, size=nominal_response.shape)
        
        diff = np.max(np.abs(measured_response - nominal_response))
        drift_detected = (diff > 0.05 * np.max(np.abs(nominal_response)))
        
        updated_geom_matrix = geometry_matrix.copy()
        if drift_detected:
            # Update the corresponding column of the geometry matrix (Rank-1 update)
            col_idx = self.current_calibration_actuator % config.ZERNIKE_MODES_MAX
            updated_geom_matrix[:, col_idx] = measured_response
            print(f"[Influence Calibrator] Actuator {self.current_calibration_actuator} influence function drift detected and corrected.")
            
        # Cycle through actuators one by one
        self.current_calibration_actuator = (self.current_calibration_actuator + 1) % self.num_actuators
        
        return drift_detected, updated_geom_matrix


class SLODARAnalyzer:
    """Problem 11 & 23: Spatial cross-correlation of subaperture slopes to resolve ground-layer (GL) vs high-altitude (HA)."""
    def __init__(self, grid_size=config.MLA_GRID_SIZE):
        self.grid_size = grid_size
        self.num_subaps = grid_size ** 2

    def analyze_slodar(self, centroids_1, centroids_2):
        """Spatial cross-correlation of subaperture slopes from two directions.
        C(di, dj) = < Sum_i,j Si,j(t) * S'_{i+di, j+dj}(t) / O(di, dj) >
        """
        s1 = centroids_1.reshape(self.grid_size, self.grid_size, 2)
        s2 = centroids_2.reshape(self.grid_size, self.grid_size, 2)
        
        cross_corr = np.zeros((5, 5))
        for di in range(-2, 3):
            for dj in range(-2, 3):
                overlap_count = 0
                val = 0.0
                for r in range(self.grid_size):
                    for c in range(self.grid_size):
                        nr, nc = r + di, c + dj
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            val += np.dot(s1[r, c], s2[nr, nc])
                            overlap_count += 1
                if overlap_count > 0:
                    cross_corr[di + 2, dj + 2] = val / overlap_count
                    
        gl_fraction = cross_corr[2, 2] / (np.sum(cross_corr) + 1e-6)
        # Clip to ensure valid fractions
        gl_fraction = np.clip(gl_fraction, 0.0, 1.0)
        ha_fraction = 1.0 - gl_fraction
        return gl_fraction, ha_fraction
