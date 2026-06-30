import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
from vajra import config

# Check if CUDA is available for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpeckleCNN(nn.Module):
    """Problem 10: PyTorch CNN regressing NCPA Zernikes from science camera speckles."""
    def __init__(self, output_dim=config.ZERNIKE_MODES_MAX):
        super(SpeckleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # 4x4
            nn.Flatten()
        )
        self.regressor = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        # x shape: (Batch, 1, 32, 32)
        feat = self.features(x)
        out = self.regressor(feat)
        return out


class KalmanFilterNCPAFusion:
    """Fuses the thermal model predictions with Speckle CNN measurements."""
    def __init__(self, num_zernikes=config.ZERNIKE_MODES_MAX):
        self.num_zernikes = num_zernikes
        self.state = np.zeros(num_zernikes) # Estimated NCPA Zernikes
        self.P = np.eye(num_zernikes) * 1e-12 # State covariance
        self.Q = np.eye(num_zernikes) * 1e-15 # Process noise covariance
        self.R = np.eye(num_zernikes) * 1e-11 # Measurement noise covariance
        
        # Thermal sensitivities (tip/tilt, defocus, astigmatism)
        self.thermal_sens = np.zeros(num_zernikes)
        self.thermal_sens[2] = 2e-8 # Defocus (Noll index 4) sensitivity (m/C)
        self.thermal_sens[4] = 1e-8 # Astigmatism sensitivity

    def predict(self, dT):
        """Predicts state evolution from thermal temperature change dT."""
        self.state = self.state + self.thermal_sens * dT
        self.P = self.P + self.Q

    def update(self, measurement):
        """Updates state estimation using Speckle CNN Zernike measurement."""
        H = np.eye(self.num_zernikes)
        S_cov = self.P + self.R
        K = self.P @ np.linalg.inv(S_cov)
        
        y_residual = measurement - self.state
        self.state = self.state + K @ y_residual
        self.P = (np.eye(self.num_zernikes) - K) @ self.P
        return self.state


class NCPATracker:
    """Problem 10: Speckle-plus-thermal NCPA tracker (Layer 6)."""
    def __init__(self, num_zernikes=config.ZERNIKE_MODES_MAX):
        self.num_zernikes = num_zernikes
        self.speckle_cnn = SpeckleCNN(output_dim=num_zernikes).to(device)
        self.speckle_cnn.eval()
        self.kf = KalmanFilterNCPAFusion(num_zernikes=num_zernikes)
        self.prev_temp = 20.0
        self._initialize_cnn()

    def _initialize_cnn(self):
        """Initializes the CNN with mock weights."""
        with torch.no_grad():
            nn.init.constant_(self.speckle_cnn.regressor[2].weight, 0.0)
            nn.init.constant_(self.speckle_cnn.regressor[2].bias, 0.0)

    def estimate_ncpa(self, science_image, telescope_temp):
        """Estimates and updates the NCPA Zernike bias vector via Kalman Filter sensor fusion."""
        # 1. Predict step: thermal expansion
        dT = telescope_temp - self.prev_temp
        self.kf.predict(dT)
        self.prev_temp = telescope_temp
        
        # 2. Speckle CNN measurement
        h, w = science_image.shape
        start_y = (h - 32) // 2
        start_x = (w - 32) // 2
        cropped = science_image[start_y:start_y+32, start_x:start_x+32]
        
        # Normalize and convert to tensor
        norm = cropped / (cropped.max() + 1e-6)
        input_tensor = torch.tensor(norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            cnn_meas_t = self.speckle_cnn(input_tensor)
            cnn_meas = cnn_meas_t.squeeze().cpu().numpy()
            
        # 3. Kalman correction step
        ncpa_bias = self.kf.update(cnn_meas)
        return ncpa_bias


class PSFReconstructor:
    """Problem 16: Telemetry-driven PSF reconstruction (Layer 7)."""
    def __init__(self, pupil_grid, zernike_basis):
        self.pupil_grid = pupil_grid
        self.zernike_basis = zernike_basis
        self.num_zernikes = config.ZERNIKE_MODES_MAX
        
        # Telemetry storage
        self.residual_zernike_history = []
        self.dm_command_history = []
        self.uarc_variance_history = []
        self.ncpa_history = []
        
    def record_telemetry(self, residual_zernikes, dm_command, uarc_variances, ncpa_bias):
        """Records loop telemetry frame-to-frame."""
        self.residual_zernike_history.append(residual_zernikes)
        self.dm_command_history.append(dm_command)
        self.uarc_variance_history.append(uarc_variances)
        self.ncpa_history.append(ncpa_bias)
        
        if len(self.residual_zernike_history) > 5000:
            self.residual_zernike_history.pop(0)
            self.dm_command_history.pop(0)
            self.uarc_variance_history.pop(0)
            self.ncpa_history.pop(0)

    def reconstruct_psf(self, field_angle_arcsec=0.0):
        """Reconstructs the parametric PSF from loop telemetry."""
        if len(self.residual_zernike_history) == 0:
            psf_grid = np.arange(32) - 15.5
            x, y = np.meshgrid(psf_grid, psf_grid)
            r2 = x**2 + y**2
            psf = np.exp(-r2 / 4.0)
            return psf / psf.sum()
            
        avg_res = np.mean(self.residual_zernike_history, axis=0)
        avg_var = np.mean(self.uarc_variance_history, axis=0)
        avg_ncpa = np.mean(self.ncpa_history, axis=0)
        
        total_phase_error = np.zeros(self.pupil_grid.size)
        for j in range(self.num_zernikes):
            mode = self.zernike_basis[j]
            total_phase_error += (avg_res[j] + avg_ncpa[j]) * mode
            
        # Add high-order DM fitting error (Problem 4 truncation)
        fitting_variance = 0.3 * (config.MLA_PITCH / config.R0_500) ** (5.0 / 3.0)
        
        # Incorporate spatial anisoplanatism (Problem 11)
        theta_0 = 7.0 # Isoplanatic angle at Hanle (7 arcsec)
        anisoplanatic_variance = (field_angle_arcsec / theta_0) ** (5.0 / 3.0) if field_angle_arcsec > 0 else 0.0
        
        # Total wavefront phase variance (including UARC uncertainty)
        total_phase_variance = np.var(total_phase_error) + np.mean(avg_var) + fitting_variance + anisoplanatic_variance
        
        # Synthesize the PSF
        psf_width = 1.0 + 0.5 * total_phase_variance
        
        psf_grid = np.arange(config.SUBAP_PIXELS * 2) - (config.SUBAP_PIXELS * 2 - 1) / 2.0
        x, y = np.meshgrid(psf_grid, psf_grid)
        r2 = x**2 + y**2
        
        psf = np.exp(-r2 / (2 * psf_width**2))
        psf /= psf.sum()
        
        return psf


class SpeckleNullingController:
    """Problem 26: Active Speckle Nulling Controller (ExAO Contrast Improvement).
    Generates phase probe shapes on the Deformable Mirror to measure and null science focal plane speckles.
    """
    def __init__(self, num_actuators=config.DM_ACTUATORS_TOTAL):
        self.num_actuators = num_actuators
        self.probe_amplitude = 1e-8 # 10 nanometers probe amplitude
        self.nulling_gain = 0.3
        self.nulling_commands = np.zeros(num_actuators)
        
    def generate_probe(self, probe_index):
        """Generates a spatial sine-wave probe pattern on the actuator grid.
        probe_index: 0, 1, 2, or 3 corresponding to phases 0, pi/2, pi, 3pi/2.
        """
        probe = np.zeros(self.num_actuators)
        side = config.DM_ACTUATORS_SIDE
        phase_offset = probe_index * (np.pi / 2.0)
        
        # Spatial frequency for the target dark hole region
        kx, ky = 1.5, 1.5 
        
        for i in range(self.num_actuators):
            row = i // side
            col = i % side
            probe[i] = self.probe_amplitude * np.sin(kx * col + ky * row + phase_offset)
            
        return probe

    def compute_nulling_commands(self, intensity_measurements):
        """Computes the actuator cancellation commands from 4 probe intensity measurements.
        intensity_measurements: list of 4 float arrays, each representing the science camera cropped region.
        """
        # Pairwise temporal phase shifting equations to reconstruct electric field
        i0, i1, i2, i3 = intensity_measurements
        
        field_real = (i0 - i2) / (4.0 * self.probe_amplitude + 1e-12)
        field_imag = (i3 - i1) / (4.0 * self.probe_amplitude + 1e-12)
        
        avg_real = np.mean(field_real)
        avg_imag = np.mean(field_imag)
        
        correction = np.zeros(self.num_actuators)
        side = config.DM_ACTUATORS_SIDE
        for i in range(self.num_actuators):
            row = i // side
            col = i % side
            correction[i] = -self.nulling_gain * (avg_real * np.sin(row) + avg_imag * np.cos(col))
            
        # Scale correction to a stable sub-nanometer step size (Problem 26 stabilization)
        max_corr = np.max(np.abs(correction))
        if max_corr > 0:
            correction = (correction / max_corr) * (0.005 * config.ACTUATOR_STROKE_LIMIT)
            
        self.nulling_commands += correction
        self.nulling_commands = np.clip(self.nulling_commands, -0.1 * config.ACTUATOR_STROKE_LIMIT, 0.1 * config.ACTUATOR_STROKE_LIMIT)
        return self.nulling_commands
