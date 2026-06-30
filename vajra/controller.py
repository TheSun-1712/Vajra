import numpy as np
import torch
import torch.nn as nn
from vajra import config

# Check if CUDA is available for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MechanicalNotchFilter:
    """Problem 14: Adaptive digital notch filter on the residual stream to suppress mechanical vibrations."""
    def __init__(self, sample_rate=1000.0, target_freqs=[48.0, 85.0]):
        self.fs = sample_rate
        self.target_freqs = target_freqs # Starting guesses for cooling pumps, motors
        self.buffer = []
        self.buffer_size = 500 # 500 ms history
        
        # Filter states per tracked mode (usually tip and tilt, so 2 channels)
        self.x1 = np.zeros(2)
        self.x2 = np.zeros(2)
        self.y1 = np.zeros(2)
        self.y2 = np.zeros(2)
        
        # Current active frequencies
        self.active_freqs = list(target_freqs)
        
    def update_vibrations(self, residual_vector):
        """Processes residuals, detects narrow peaks in FFT, and updates notch filter frequencies."""
        self.buffer.append(residual_vector)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            
        # Periodically (every 100 frames) re-evaluate active frequencies from FFT
        if len(self.buffer) == self.buffer_size and np.random.rand() < 0.01:
            data = np.array(self.buffer)
            # Perform FFT on tip channel
            fft_vals = np.abs(np.fft.rfft(data[:, 0]))
            freqs = np.fft.rfftfreq(self.buffer_size, d=1.0/self.fs)
            
            # Find dominant peaks above atmospheric Kolmogorov trend
            trend = 1.0 / (np.maximum(freqs, 1.0) ** (2.0/3.0))
            normalized_fft = fft_vals / (trend * np.mean(fft_vals) / np.mean(trend))
            
            # Find peaks (val > 4.0 standard deviations)
            peaks = np.where((normalized_fft > 4.0) & (freqs > 10.0) & (freqs < 150.0))[0]
            if len(peaks) > 0:
                detected_freqs = freqs[peaks]
                # Update closest target frequency
                for f in detected_freqs[:len(self.active_freqs)]:
                    # Gentle update (low-pass filter the frequency tracker)
                    self.active_freqs[0] = 0.9 * self.active_freqs[0] + 0.1 * f

    def filter_signal(self, raw_signal):
        """Applies the digital notch filter at the tracked frequencies."""
        filtered = np.zeros(2)
        r = 0.98 # Bandwidth parameter (close to 1 means narrow notch)
        
        for ch in range(2):
            val = raw_signal[ch]
            # Apply cascade of notch filters for each active frequency
            for freq in self.active_freqs:
                w0 = 2 * np.pi * freq / self.fs
                cos_w0 = np.cos(w0)
                
                # Coefficients for notch filter transfer function H(z)
                b0, b1, b2 = 1.0, -2.0 * cos_w0, 1.0
                a1, a2 = -2.0 * r * cos_w0, r * r
                
                # Difference equation: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
                y = b0 * val + b1 * self.x1[ch] + b2 * self.x2[ch] - a1 * self.y1[ch] - a2 * self.y2[ch]
                
                # Shift states
                self.x2[ch] = self.x1[ch]
                self.x1[ch] = val
                self.y2[ch] = self.y1[ch]
                self.y1[ch] = y
                
                val = y # Output becomes input to next filter in cascade
            filtered[ch] = val
            
        return filtered


class RegimeClassifier(nn.Module):
    """1D-CNN classifying atmospheric turbulence from spatial autocorrelation maps time series."""
    def __init__(self, history_len=10, feature_dim=512):
        super(RegimeClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=history_len, out_channels=16, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Linear(16 * (feature_dim // 2), 64),
            nn.ReLU(),
            nn.Linear(64, 3) # 3-class softmax (Steady, Transitioning, Broken)
        )

    def forward(self, x):
        # Input shape: (Batch, history_len, feature_dim)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits


class TransformerPredictor(nn.Module):
    """Transformer sequences model attending across time & mode dimension to forecast Zernikes."""
    def __init__(self, history_len=20, input_dim=67, output_dim=66):
        super(TransformerPredictor, self).__init__()
        # Linear layer mapping Zernike + latency vector (67) to token (64)
        self.input_layer = nn.Linear(input_dim, 64)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.output_layer = nn.Linear(64 * history_len, output_dim)

    def forward(self, x):
        # Input shape: (Batch, history_len, 67)
        tokens = self.input_layer(x) # (Batch, history_len, 64)
        encoded = self.transformer_encoder(tokens) # (Batch, history_len, 64)
        flat = encoded.view(encoded.size(0), -1)
        out = self.output_layer(flat) # (Batch, 66)
        return out


class RecurrentWavefrontPredictor(nn.Module):
    """GRU-based recurrent neural network to forecast Zernike sequence transitions."""
    def __init__(self, input_dim=config.ZERNIKE_MODES_MAX, hidden_dim=64):
        super(RecurrentWavefrontPredictor, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, h0=None):
        # x shape: (Batch, SeqLen, ZernikeDim)
        out, h_next = self.gru(x, h0)
        pred = self.fc(out[:, -1, :])
        return pred, h_next


class RegimeAwarePredictor:
    """Problem 8 & 20: Regime-aware predictor classifying atmospheric states to select optimal predictive models."""
    def __init__(self):
        self.history = []
        self.history_len = 50
        
        # Regime probabilities (Steady, Transitioning, Broken)
        self.probabilities = np.array([1.0, 0.0, 0.0])
        
        # PyTorch Networks
        self.classifier_net = RegimeClassifier(history_len=10, feature_dim=512).to(device)
        self.transformer_pred = TransformerPredictor(history_len=20, input_dim=67, output_dim=66).to(device)
        
        # Recurrent GRU predictor (Optimizations 2)
        self.recurrent_predictor = RecurrentWavefrontPredictor().to(device)
        
        self.classifier_net.eval()
        self.transformer_pred.eval()
        self.recurrent_predictor.eval()
        
        self.recurrent_hidden = None
        self.recurrent_history = []
        
        self._initialize_mock_weights()
        
    def _initialize_mock_weights(self):
        """Initializes weights with reasonable defaults."""
        with torch.no_grad():
            nn.init.constant_(self.transformer_pred.output_layer.bias, 0.0)
            self.transformer_pred.output_layer.weight.data.normal_(0.0, 0.01)
            
            # Recurrent GRU predictor weights
            nn.init.constant_(self.recurrent_predictor.fc.bias, 0.0)
            self.recurrent_predictor.fc.weight.data.normal_(0.0, 0.01)

    def classify_regime(self, slopes):
        """Classifies the turbulence regime from the spatial correlation matrix of the slopes."""
        self.history.append(slopes)
        if len(self.history) > self.history_len:
            self.history.pop(0)
            
        if len(self.history) < 10:
            return
            
        # Compile history for classification input
        # Spatial autocorrelation feature
        autocorr_seq = []
        for s in self.history[-10:]:
            # Compute sliding 1D correlation as a spatial autocorrelation proxy
            ac = np.correlate(s, s, mode='same')
            autocorr_seq.append(ac)
            
        autocorr_arr = np.array(autocorr_seq, dtype=np.float32) # (10, 512)
        input_tensor = torch.tensor(autocorr_arr, device=device).unsqueeze(0) # (1, 10, 512)
        
        with torch.no_grad():
            logits = self.classifier_net(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            
        self.probabilities = 0.9 * self.probabilities + 0.1 * probs

    def predict(self, current_zernikes, dt_latency, mount_acceleration=None):
        """Predicts the wavefront state at time t + dt_latency, incorporating feed-forward mount acceleration."""
        if len(self.history) < 20:
            return current_zernikes
            
        steps_ahead = dt_latency / config.NOMINAL_LATENCY
        
        # 1. Steady Predictor (extrapolate last velocity with 0.1 damping to prevent noise amplification)
        vel = current_zernikes - self.history[-2][:config.ZERNIKE_MODES_MAX] if len(self.history) >= 2 else 0.0
        p_steady = current_zernikes + 0.1 * vel * steps_ahead
        
        # 2. Transitioning Predictor (AR-1 decay)
        ar1_coeff = 0.95
        p_trans = current_zernikes * (ar1_coeff ** steps_ahead)
        
        # 3. Broken Predictor (fast decay hold)
        decay_coeff = 0.50
        p_broken = current_zernikes * (decay_coeff ** steps_ahead)
        
        # 4. Recurrent GRU sequence prediction (Optimizations 2)
        self.recurrent_history.append(current_zernikes)
        if len(self.recurrent_history) > 10:
            self.recurrent_history.pop(0)
            
        if len(self.recurrent_history) >= 5:
            seq_tensor = torch.tensor(np.array(self.recurrent_history), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                recurrent_pred, self.recurrent_hidden = self.recurrent_predictor(seq_tensor, self.recurrent_hidden)
                if self.recurrent_hidden is not None:
                    self.recurrent_hidden = self.recurrent_hidden.detach()
                p_recurrent = recurrent_pred.squeeze(0).cpu().numpy()
        else:
            p_recurrent = current_zernikes.copy()
        
        # Combine predictions blending GRU predictions for transitioning & steady states
        predicted_zernikes = (
            self.probabilities[0] * p_steady * 0.5 + self.probabilities[0] * p_recurrent * 0.5 +
            self.probabilities[1] * p_recurrent +
            self.probabilities[2] * p_broken
        )
        
        # Feed-forward mount vibration tip-tilt rejection (Zernike index 0 and 1)
        if mount_acceleration is not None:
            # Mount acceleration is double-integrated to calculate position offset
            ff_correction = -0.05 * mount_acceleration
            predicted_zernikes[:2] += ff_correction[:2]
        
        # Optional: Run Transformer Predictor on GPU if available for prediction refinement
        history_z = [h[:config.ZERNIKE_MODES_MAX] for h in self.history[-20:]]
        # Concat latency (Steps ahead)
        seq_data = []
        for hz in history_z:
            combined = np.append(hz, dt_latency)
            seq_data.append(combined)
            
        input_tensor = torch.tensor(np.array(seq_data), dtype=torch.float32, device=device).unsqueeze(0) # (1, 20, 67)
        with torch.no_grad():
            t_pred = self.transformer_pred(input_tensor).squeeze().cpu().numpy()
            
        # Return the stable physical regime-based predictor by default.
        # The deep Transformer sequence network can be blended in after offline calibration.
        final_prediction = predicted_zernikes
        return final_prediction


class GLAODecomposer:
    """Problem 11 & 23: Decomposes slopes into ground-layer (GL) and high-altitude (HA) components."""
    def __init__(self, grid_size=config.MLA_GRID_SIZE):
        self.grid_size = grid_size
        self.num_subaps = grid_size ** 2
        
    def decompose_slopes(self, centroids, illumination_fractions):
        """Separates ground-layer slopes from high-altitude slopes via spatial low-pass filtering."""
        gl_centroids = np.zeros_like(centroids)
        ha_centroids = np.zeros_like(centroids)
        
        x_slopes = np.zeros((self.grid_size, self.grid_size))
        y_slopes = np.zeros((self.grid_size, self.grid_size))
        valid_mask = np.zeros((self.grid_size, self.grid_size))
        
        for i in range(self.num_subaps):
            if illumination_fractions[i] >= 0.1:
                r, c = i // self.grid_size, i % self.grid_size
                x_slopes[r, c] = centroids[i, 0]
                y_slopes[r, c] = centroids[i, 1]
                valid_mask[r, c] = 1.0
                
        # Box filter over 5x5 subapertures
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                r_min, r_max = max(0, r - 2), min(self.grid_size, r + 3)
                c_min, c_max = max(0, c - 2), min(self.grid_size, c + 3)
                
                weights = valid_mask[r_min:r_max, c_min:c_max]
                w_sum = np.sum(weights)
                
                if w_sum > 0:
                    x_avg = np.sum(x_slopes[r_min:r_max, c_min:c_max] * weights) / w_sum
                    y_avg = np.sum(y_slopes[r_min:r_max, c_min:c_max] * weights) / w_sum
                    
                    i = r * self.grid_size + c
                    if illumination_fractions[i] >= 0.1:
                        gl_centroids[i] = [x_avg, y_avg]
                        ha_centroids[i] = centroids[i] - gl_centroids[i]
                        
        return gl_centroids, ha_centroids

    def get_controlled_slopes(self, centroids, illumination_fractions):
        """Applies asymmetric loop gains: 0.8 for Ground Layer, 0.15 for High Altitude."""
        gl, ha = self.decompose_slopes(centroids, illumination_fractions)
        controlled = 0.8 * gl + 0.15 * ha
        return controlled


class SaturationNN(nn.Module):
    """MLP approximating stroke-constrained quadratic programming optimization (Layer 4)."""
    def __init__(self, num_zernikes=config.ZERNIKE_MODES_MAX, num_actuators=config.DM_ACTUATORS_TOTAL):
        super(SaturationNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_zernikes + num_actuators, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actuators)
        )
        
    def forward(self, target_conjugate, current_commands):
        x = torch.cat([target_conjugate, current_commands], dim=1)
        corrections = self.net(x)
        return corrections


class NeuralHysteresisModel(nn.Module):
    """GRU-based Neural Hysteresis Compensator replacing play operators."""
    def __init__(self, num_actuators=config.DM_ACTUATORS_TOTAL):
        super(NeuralHysteresisModel, self).__init__()
        self.num_actuators = num_actuators
        self.gru = nn.GRU(input_size=num_actuators, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, num_actuators)
        
    def forward(self, u_seq, h0=None):
        out_gru, h_next = self.gru(u_seq, h0)
        correction = self.fc(out_gru)
        # Residual connection ensures compensated command is close to input at startup
        compensated = u_seq + correction
        return compensated, h_next


class ActuatorController:
    """Handles mapping reconstructed wavefronts to actuator voltages with saturation, hysteresis, and edge effects."""
    def __init__(self, reconstructor_matrix, pupil_grid=None, zernike_basis=None):
        self.num_actuators = config.DM_ACTUATORS_TOTAL
        self.num_zernikes = config.ZERNIKE_MODES_MAX
        self.stroke_limit = config.ACTUATOR_STROKE_LIMIT
        
        # 1. Geometry reconstruction-to-command matrix mapping Zernikes to DM commands
        self.zernike_to_command = self._compute_zernike_to_command(reconstructor_matrix, pupil_grid, zernike_basis)
        
        # 2. Saturation approximation NN
        self.saturation_nn = SaturationNN().to(device)
        self.saturation_nn.eval()
        self._initialize_sat_nn()
        
        # Define edge/boundary flags (Problem 25)
        self.edge_weights = self._compute_edge_redistribution_weights()
        
        # 3. Neural Hysteresis Compensator (GRU)
        self.neural_hysteresis = NeuralHysteresisModel(self.num_actuators).to(device)
        self.neural_hysteresis.eval()
        self.hysteresis_hidden = None
        self._initialize_hyst_nn()
        
        # History of commands for saturation load-shifting prediction
        self.command_history = []
        self.history_len = 10
        
        # Dynamic loop parameter settings (for RL agent tuning)
        self.modal_gains = np.ones(self.num_zernikes)
        self.glao_weight = 0.8

    def _compute_zernike_to_command(self, reconstructor_matrix, pupil_grid=None, zernike_basis=None):
        """Computes the static projection from Zernike phase coefficients to actuator stroke units."""
        if pupil_grid is None or zernike_basis is None:
            # Fallback projection mapping for unit tests
            projection = np.zeros((self.num_actuators, self.num_zernikes))
            for i in range(self.num_actuators):
                row = i // config.DM_ACTUATORS_SIDE
                col = i % config.DM_ACTUATORS_SIDE
                projection[i, 0] = (col - 9) * 1e-8
                projection[i, 1] = (row - 9) * 1e-8
            return projection

        side = config.DM_ACTUATORS_SIDE
        half_side = (side - 1) / 2.0
        actuator_pitch = config.D_APERTURE / (side - 1)
        
        projection = np.zeros((self.num_actuators, self.num_zernikes))
        # Stroke height scale: convert Zernike radians to physical optical path height (m)
        # OPD = 2 * delta_h -> delta_h = phase * lambda / (4 * pi)
        scale = config.LAMBDA_SENSING / (4.0 * np.pi)
        
        for i in range(self.num_actuators):
            row = i // side
            col = i % side
            x = (col - half_side) * actuator_pitch
            y = (row - half_side) * actuator_pitch
            
            # Find the closest grid point in the pupil grid
            dx = pupil_grid.x - x
            dy = pupil_grid.y - y
            dist = dx**2 + dy**2
            closest_idx = np.argmin(dist)
            
            for j in range(self.num_zernikes):
                projection[i, j] = zernike_basis[j][closest_idx] * scale
                
        return projection

    def _compute_edge_redistribution_weights(self):
        """Problem 25: Computes weights indicating proximity to boundaries."""
        weights = np.ones(self.num_actuators)
        side = config.DM_ACTUATORS_SIDE
        for i in range(self.num_actuators):
            row = i // side
            col = i % side
            dist_to_edge = min(row, col, side - 1 - row, side - 1 - col)
            if dist_to_edge == 0:
                weights[i] = 0.1 # Heavily penalize outer ring
            elif dist_to_edge == 1:
                weights[i] = 0.5 # Moderately penalize inner ring
        return weights

    def _initialize_sat_nn(self):
        """Initializes the saturation network to zero initially."""
        with torch.no_grad():
            nn.init.constant_(self.saturation_nn.net[4].weight, 0.0)
            nn.init.constant_(self.saturation_nn.net[4].bias, 0.0)

    def _initialize_hyst_nn(self):
        """Initializes the hysteresis model to approximate the identity mapping initially."""
        with torch.no_grad():
            for name, param in self.neural_hysteresis.named_parameters():
                if 'weight' in name:
                    param.data.normal_(0.0, 0.01)
                elif 'bias' in name:
                    param.data.fill_(0.0)
            # Setting fc layer to zero ensures zero correction residual on start
            nn.init.constant_(self.neural_hysteresis.fc.weight, 0.0)
            nn.init.constant_(self.neural_hysteresis.fc.bias, 0.0)

    def set_loop_parameters(self, modal_gains=None, glao_weight=None):
        """Sets the loop parameters dynamically tuned by the RL agent."""
        if modal_gains is not None:
            self.modal_gains = np.clip(modal_gains, 0.05, 2.0)
        if glao_weight is not None:
            self.glao_weight = np.clip(glao_weight, 0.0, 1.0)

    def clip_interactuator_shear(self, commands):
        """Ensures that the commanded shape does not exceed maximum local slope limit between adjacent actuators."""
        side = config.DM_ACTUATORS_SIDE
        max_diff = 0.3 * self.stroke_limit # Max allowable shear stroke (approx 1.05 microns)
        clipped = commands.copy()
        
        # Run 2 passes to propagate constraints smoothly
        for _ in range(2):
            for i in range(self.num_actuators):
                row = i // side
                col = i % side
                
                # Check 4 neighbors (up, down, left, right)
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < side and 0 <= nc < side:
                        neighbors.append(nr * side + nc)
                        
                for n in neighbors:
                    diff = clipped[i] - clipped[n]
                    if np.abs(diff) > max_diff:
                        # Pull current command closer to neighbor
                        clipped[i] = clipped[n] + np.sign(diff) * max_diff
                        
        return clipped

    def calculate_commands(self, target_zernikes, current_commands):
        """Problem 7 & 25: Computes safe actuator commands avoiding saturation cascade and shear violations."""
        # Scale target Zernikes by modal gains
        scaled_zernikes = target_zernikes * self.modal_gains
        u_target = self.zernike_to_command @ scaled_zernikes
        
        # Update command history
        self.command_history.append(current_commands)
        if len(self.command_history) > self.history_len:
            self.command_history.pop(0)
            
        # 1. Pre-emptive load shifting
        if len(self.command_history) == self.history_len:
            hist = np.array(self.command_history)
            velocity = np.mean(np.diff(hist, axis=0), axis=0)
            predicted_u = current_commands + velocity * 5.0
            
            sat_threshold = 0.85 * self.stroke_limit
            saturated_mask = (np.abs(predicted_u) > sat_threshold)
            
            if np.any(saturated_mask):
                u_target = self.redistribute_forces(u_target, saturated_mask)

        # 2. Run saturation network for constrained least-squares approximation
        target_t = torch.tensor(target_zernikes, dtype=torch.float32, device=device).unsqueeze(0)
        curr_t = torch.tensor(current_commands, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            u_corr = self.saturation_nn(target_t, curr_t).squeeze().cpu().numpy()
            
        # Apply stable negative feedback loop gain (subtraction)
        final_u = current_commands - 0.4 * u_target + u_corr
        
        # Hard clip as absolute safety check
        final_u = np.clip(final_u, -self.stroke_limit, self.stroke_limit)
        
        # Apply interactuator shear check to protect the DM membrane
        final_u = self.clip_interactuator_shear(final_u)
        
        # 3. Apply inverse hysteresis pre-shaping
        compensated_u = self.compensate_hysteresis(final_u)
        compensated_u = self.clip_interactuator_shear(compensated_u)
        
        return final_u, compensated_u

    def redistribute_forces(self, u_target, saturated_mask):
        """Problem 25: Redistributes stroke demands from saturated edge/interior actuators."""
        u_out = u_target.copy()
        side = config.DM_ACTUATORS_SIDE
        
        for i in range(self.num_actuators):
            if saturated_mask[i]:
                excess = np.sign(u_out[i]) * (np.abs(u_out[i]) - 0.85 * self.stroke_limit)
                if np.abs(u_out[i]) < 0.85 * self.stroke_limit:
                    continue
                    
                u_out[i] = np.sign(u_out[i]) * 0.85 * self.stroke_limit
                
                # Redistribute to 4-neighbors
                row, col = i // side, i % side
                neighbors = []
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < side and 0 <= nc < side:
                        ni = nr * side + nc
                        if not saturated_mask[ni]:
                            neighbors.append(ni)
                            
                if len(neighbors) > 0:
                    n_weights = np.array([self.edge_weights[n] for n in neighbors])
                    n_weights /= np.sum(n_weights) + 1e-6
                    for idx, ni in enumerate(neighbors):
                        u_out[ni] += excess * n_weights[idx]
                        
        return u_out

    def compensate_hysteresis(self, u_target):
        """Problem 21: Applies Neural Hysteresis Compensator (GRU) pre-shaping."""
        u_tensor = torch.tensor(u_target, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) # (1, 1, num_actuators)
        with torch.no_grad():
            compensated_t, self.hysteresis_hidden = self.neural_hysteresis(u_tensor, self.hysteresis_hidden)
        # Detach hidden state to prevent computation graph backprop memory growth
        if self.hysteresis_hidden is not None:
            self.hysteresis_hidden = self.hysteresis_hidden.detach()
        compensated = compensated_t.squeeze().cpu().numpy()
        return compensated


class ChromaticOffsetModel:
    """Problem 13: Computes wavelength-dependent wavefront correction scales and atmospheric dispersion offsets."""
    def __init__(self):
        self.scale_factor = (config.LAMBDA_SCIENCE / config.LAMBDA_SENSING) ** (6.5 / 5.0)
        
    def get_chromatic_offset(self, zernikes, zenith_angle_deg):
        """Returns offset correction for science path."""
        zenith_rad = np.radians(zenith_angle_deg)
        refraction_shift = 1.2e-7 * np.tan(zenith_rad) # Dispersion scale at Hanle
        
        offsets = zernikes * (self.scale_factor - 1.0)
        offsets[0] += refraction_shift # Add chromatic tip offset
        offsets[1] += refraction_shift * 0.5 # Add chromatic tilt offset
        return offsets


class TemporalMultiplexor:
    """Problem 12: Alternates DM corrections to approximate multi-conjugate correction (MCAO)."""
    def __init__(self, num_states=3):
        self.num_states = num_states
        self.current_state = 0
        self.conjugate_modifiers = [1.0, 0.6, 0.3]

    def multiplex_command(self, base_command):
        """Alternates the applied DM correction across successive frames."""
        modifier = self.conjugate_modifiers[self.current_state]
        multiplexed = base_command * modifier
        self.current_state = (self.current_state + 1) % self.num_states
        return multiplexed


class LQGStructuralTracker:
    """Problem 30: Adaptive LQG structural vibration tracker.
    Uses a dynamic Kalman filter to estimate narrow-band resonances and adaptively update notch filters.
    """
    def __init__(self, dt=0.001):
        self.dt = dt
        # State space: [position, velocity, acceleration, frequency]
        self.state = np.array([0.0, 0.0, 0.0, 48.0]) # starting at 48 Hz default vibration
        self.P = np.eye(4) * 0.1 # Covariance matrix
        self.Q = np.diag([1e-4, 1e-3, 1e-2, 1e-3]) # Process noise
        self.R = np.array([[1e-2]]) # Measurement noise (tip-tilt readout)
        
    def update_and_track(self, z_tt_measured):
        """z_tt_measured: current measured tip-tilt coefficient (scalar)."""
        # 1. Prediction step: state transition
        f = self.state[3]
        omega = 2.0 * np.pi * f
        
        F = np.array([
            [1.0, self.dt, 0.0, 0.0],
            [0.0, 1.0, self.dt, 0.0],
            [-omega**2, 0.0, 0.0, -2.0 * omega * self.state[0]], # linearized wrt f
            [0.0, 0.0, 0.0, 1.0] # frequency walk
        ])
        
        self.state = np.array([
            self.state[0] + self.state[1] * self.dt,
            self.state[1] + self.state[2] * self.dt,
            -omega**2 * self.state[0],
            self.state[3]
        ])
        
        self.P = F @ self.P @ F.T + self.Q
        
        # 2. Update step
        H = np.array([[1.0, 0.0, 0.0, 0.0]])
        y = z_tt_measured - (H @ self.state)[0] # measurement residual
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K.squeeze() * y
        self.P = (np.eye(4) - K @ H) @ self.P
        
        # Clip estimated frequency to safe physical bounds (30 Hz to 80 Hz)
        self.state[3] = np.clip(self.state[3], 30.0, 80.0)
        return self.state[3]
