import numpy as np
import torch
import torch.nn as nn
import hcipy
from vajra import config

# Check if CUDA is available for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNetBlock(nn.Module):
    """Residual Convolutional Block for the subaperture patch feature extraction."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Layer0AttentionCNN(nn.Module):
    """Upgraded Layer 0 Neural WFS featuring a ResNet patch encoder + Transformer self-attention.
    Ingests raw 256x256 detector Hartmannograms and outputs Zernike coefficients and uncertainties.
    """
    def __init__(self, num_zernikes=config.ZERNIKE_MODES_MAX):
        super(Layer0AttentionCNN, self).__init__()
        self.num_zernikes = num_zernikes
        
        # 1. Shared ResNet Backbone: Maps each 16x16 subaperture frame to a 64-d token
        self.resnet_backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResNetBlock(16, 32, stride=2), # 8x8
            ResNetBlock(32, 64, stride=2), # 4x4
            nn.AdaptiveAvgPool2d((1, 1)), # 1x1
            nn.Flatten()
        )
        
        # 2. Learnable 2D Grid Positional Embeddings for the 256 subaperture tokens
        self.pos_embedding = nn.Parameter(torch.randn(256, 64))
        
        # 3. Cross-Subaperture Spatial Self-Attention Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 4. Multi-Task Output heads
        self.regressor = nn.Sequential(
            nn.Linear(64 + 1, 128), # Average global token + background pedestal prior (Problem 24)
            nn.ReLU(),
            nn.Linear(128, num_zernikes)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Predicts 1 log-variance per subaperture token
        )
        
    def forward(self, x, background_pedestal=10.0):
        # Input shape: (Batch, 1, 256, 256)
        batch_size = x.size(0)
        
        # Slice into 256 subaperture patches of size 16x16
        patches = x.unfold(2, 16, 16).unfold(3, 16, 16) # (Batch, 1, 16, 16, 16, 16)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous() # (Batch, 16, 16, 1, 16, 16)
        patches = patches.view(-1, 1, 16, 16) # (Batch * 256, 1, 16, 16)
        
        # Ingest weight-shared ResNet features
        tokens = self.resnet_backbone(patches) # (Batch * 256, 64)
        tokens = tokens.view(batch_size, 256, 64) # (Batch, 256, 64)
        
        # Inject positional information
        tokens = tokens + self.pos_embedding.unsqueeze(0)
        
        # Pass to Cross-Subaperture Transformer Encoder
        tokens_encoded = self.transformer_encoder(tokens) # (Batch, 256, 64)
        
        # Average pooling across subapertures for global modal reconstruction
        global_token = torch.mean(tokens_encoded, dim=1) # (Batch, 64)
        
        # Concatenate background pedestal prior
        bg = torch.tensor([[background_pedestal]], dtype=x.dtype, device=x.device).repeat(batch_size, 1)
        combined = torch.cat([global_token, bg], dim=1)
        
        zernikes = self.regressor(combined)
        return zernikes

    def predict_with_confidence(self, x, background_pedestal=10.0):
        batch_size = x.size(0)
        patches = x.unfold(2, 16, 16).unfold(3, 16, 16)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(-1, 1, 16, 16)
        
        tokens = self.resnet_backbone(patches)
        tokens = tokens.view(batch_size, 256, 64)
        tokens = tokens + self.pos_embedding.unsqueeze(0)
        tokens_encoded = self.transformer_encoder(tokens)
        
        global_token = torch.mean(tokens_encoded, dim=1)
        bg = torch.tensor([[background_pedestal]], dtype=x.dtype, device=x.device).repeat(batch_size, 1)
        combined = torch.cat([global_token, bg], dim=1)
        zernikes = self.regressor(combined)
        
        # Heteroscedastic log-variance outputs per subaperture (Problem 2)
        log_var = self.confidence_head(tokens_encoded).squeeze(-1) # (Batch, 256)
        return zernikes, log_var


class UARCBayesianReconstructor(nn.Module):
    """Bayesian neural network reconstructor for slope-to-Zernike mappings (Layer 1 / UARC).
    Outputs both Zernike coefficients (mean) and calibrated variance (uncertainty) per mode.
    """
    def __init__(self, input_dim=768, output_dim=config.ZERNIKE_MODES_MAX):
        super(UARCBayesianReconstructor, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.mean_head = nn.Linear(128, output_dim)
        self.var_head = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.Softplus() # Variance must be positive
        )
        
    def forward(self, x):
        feat = self.shared(x)
        means = self.mean_head(feat)
        variances = self.var_head(feat) + 1e-6
        return means, variances

    def forward_mc(self, x, num_samples=10):
        """Performs Monte Carlo dropout to obtain empirical mean and variance estimates."""
        self.train() # Enable dropout during inference
        means_list, vars_list = [], []
        with torch.no_grad():
            for _ in range(num_samples):
                means, variances = self.forward(x)
                means_list.append(means)
                vars_list.append(variances)
                
        means_stack = torch.stack(means_list) # (samples, batch, output_dim)
        vars_stack = torch.stack(vars_list)
        
        total_mean = torch.mean(means_stack, dim=0)
        epistemic_var = torch.var(means_stack, dim=0)
        aleatoric_var = torch.mean(vars_stack, dim=0)
        total_var = epistemic_var + aleatoric_var
        
        self.eval()
        return total_mean, total_var


class WavefrontReconstructor:
    """Wavefront reconstructor handling classical WLS, BNN UARC, dynamic masking, and CNN WFS."""
    def __init__(self, pupil_grid, pupil_mask, subaps_pos, subap_masks):
        self.pupil_grid = pupil_grid
        self.pupil_mask = pupil_mask
        self.subaps_pos = subaps_pos
        self.subap_masks = subap_masks
        
        self.grid_size = config.MLA_GRID_SIZE
        self.num_subaps = len(subaps_pos)
        self.num_zernikes = config.ZERNIKE_MODES_MAX
        
        self.zernike_basis = hcipy.make_zernike_basis(
            self.num_zernikes + 1, config.D_APERTURE, self.pupil_grid, starting_mode=2
        )
        
        self.G_full = self._precompute_geometry_matrix()
        
        self.cnn_sensor = Layer0AttentionCNN().to(device)
        self.bnn_reconstructor = UARCBayesianReconstructor(
            input_dim=self.num_subaps * 2 + self.num_subaps, # slopes (x, y) + confidences
            output_dim=self.num_zernikes
        ).to(device)
        
        self.cnn_sensor.eval()
        self.bnn_reconstructor.eval()
        self._initialize_mock_weights()
        
    def _precompute_geometry_matrix(self):
        """Computes the interaction matrix G mapping Zernike coefficients to subaperture slopes."""
        G = np.zeros((2 * self.num_subaps, self.num_zernikes))
        
        dx_operator = hcipy.make_derivative_matrix(self.pupil_grid, axis='x')
        dy_operator = hcipy.make_derivative_matrix(self.pupil_grid, axis='y')
        
        for j in range(self.num_zernikes):
            z_mode = self.zernike_basis[j]
            dz_dx = dx_operator.dot(z_mode) * self.pupil_mask
            dz_dy = dy_operator.dot(z_mode) * self.pupil_mask
            
            for i, mask in enumerate(self.subap_masks):
                if np.sum(mask) > 0:
                    mask_bool = (mask > 0)
                    G[i, j] = np.mean(dz_dx[mask_bool])
                    G[i + self.num_subaps, j] = np.mean(dz_dy[mask_bool])
                    
        return G

    def reconstruct_wls(self, centroids, confidences, illumination_fractions, prev_r0=config.R0_500):
        """Problem 2 & 22 & 3: Classical Weighted Least Squares (WLS) wavefront reconstruction."""
        active_indices = []
        for i in range(self.num_subaps):
            if illumination_fractions[i] >= 0.1 and confidences[i] > 1e-3:
                active_indices.append(i)
                
        num_active = len(active_indices)
        if num_active < 3:
            return np.zeros(self.num_zernikes), np.ones(self.num_zernikes) * 0.1
            
        # Problem 3: Centroid gain range checking
        adjusted_confidences = confidences.copy()
        if prev_r0 < 0.06:
            centroid_mag = np.sum(centroids**2, axis=1)
            truncation_regime = (centroid_mag > 2.0)
            adjusted_confidences[truncation_regime] *= 0.3
            
        y_active = np.zeros(2 * num_active)
        w_active = np.zeros(2 * num_active)
        
        for idx, subap_idx in enumerate(active_indices):
            y_active[idx] = centroids[subap_idx, 0]
            y_active[idx + num_active] = centroids[subap_idx, 1]
            w_active[idx] = adjusted_confidences[subap_idx]
            w_active[idx + num_active] = adjusted_confidences[subap_idx]
            
        W = np.diag(w_active)
        
        G_active = np.zeros((2 * num_active, self.num_zernikes))
        for idx, subap_idx in enumerate(active_indices):
            G_active[idx, :] = self.G_full[subap_idx, :]
            G_active[idx + num_active, :] = self.G_full[subap_idx + self.num_subaps, :]
            
        # Problem 4: Aliasing modal truncation
        supportable_modes = int(np.minimum(num_active * 0.8, self.num_zernikes))
        G_active = G_active[:, :supportable_modes]
        
        alpha = 0.05
        reg_term = alpha * np.eye(supportable_modes)
        
        try:
            A_matrix = G_active.T @ W @ G_active + reg_term
            b_vector = G_active.T @ W @ y_active
            a_sol = np.linalg.solve(A_matrix, b_vector)
            
            zernikes = np.zeros(self.num_zernikes)
            zernikes[:supportable_modes] = a_sol
            
            residuals = y_active - G_active @ a_sol
            var_fit = np.var(residuals) if len(residuals) > 0 else 0.1
            uncertainties = np.ones(self.num_zernikes) * var_fit
            
            return zernikes, uncertainties
        except np.linalg.LinAlgError:
            return np.zeros(self.num_zernikes), np.ones(self.num_zernikes) * 0.1

    def reconstruct_bnn(self, centroids, confidences):
        """Problem 15: Bayesian Uncertainty-Aware Reconstruction (UARC)."""
        flat_centroids = centroids.flatten()
        input_data = np.concatenate([flat_centroids, confidences])
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device=device).unsqueeze(0)
        
        mean_t, var_t = self.bnn_reconstructor.forward_mc(input_tensor, num_samples=10)
        
        z_mean = mean_t.squeeze().cpu().numpy()
        z_var = var_t.squeeze().cpu().numpy()
        return z_mean, z_var

    def reconstruct_cnn(self, raw_detector_frame, background_pedestal=10.0):
        """Problem 1: Reference-free WFS CNN (Layer 0)."""
        frame_norm = (raw_detector_frame - background_pedestal) / (raw_detector_frame.max() + 1e-5)
        input_tensor = torch.tensor(frame_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            zernikes_t = self.cnn_sensor(input_tensor, background_pedestal)
            
        return zernikes_t.squeeze().cpu().numpy()

    def _initialize_mock_weights(self):
        """Initializes weights to mimic pre-trained states."""
        with torch.no_grad():
            # Setup BNN mean mapping based on G pseudo-inverse
            G_torch = torch.tensor(self.G_full, dtype=torch.float32).to(device)
            G_pinv = torch.linalg.pinv(G_torch).T # size (num_subaps*2, num_zernikes)
            
            # Direct mock mapping projection weight inject
            nn.init.constant_(self.bnn_reconstructor.mean_head.bias, 0.0)
            nn.init.constant_(self.bnn_reconstructor.var_head[0].weight, 0.0)
            nn.init.constant_(self.bnn_reconstructor.var_head[0].bias, -2.0)
            
            # CNN regressor weights init
            nn.init.normal_(self.cnn_sensor.regressor[2].weight, mean=0.0, std=0.02)
            nn.init.constant_(self.cnn_sensor.regressor[2].bias, 0.0)
