import numpy as np
import torch
import torch.nn as nn
from vajra import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DifferentiablePropagator(nn.Module):
    """PyTorch-based differentiable physical optics wave propagator.
    Uses Fourier optics to simulate subaperture PSFs and convolve them with solar scenes.
    """
    def __init__(self, pupil_mask_numpy, zernike_basis_numpy):
        super(DifferentiablePropagator, self).__init__()
        self.pupil_mask = torch.tensor(pupil_mask_numpy, dtype=torch.float32, device=device)
        basis_arr = np.array([basis for basis in zernike_basis_numpy], dtype=np.float32) # (66, 65536)
        self.basis = torch.tensor(basis_arr, dtype=torch.float32, device=device) # (66, 65536)

    def forward(self, predicted_zernikes, granulation_patches, raw_subimages, background_pedestal=10.0, target_mode='solar'):
        """Computes the image-to-image reconstruction loss.
        
        predicted_zernikes: (Batch, 66)
        granulation_patches: (Batch * 256, 16, 16) or None if target_mode is 'point'
        raw_subimages: (Batch * 256, 16, 16)
        """
        batch_size = predicted_zernikes.size(0)
        
        # 1. Reconstruct phase screen on pupil grid
        phase = torch.matmul(predicted_zernikes, self.basis) # (Batch, 65536)
        phase = phase * self.pupil_mask.unsqueeze(0)
        phase_2d = phase.view(batch_size, 256, 256)
        
        # 2. Slice pupil phase into 256 subaperture grids of size 16x16
        subap_phases = phase_2d.unfold(1, 16, 16).unfold(2, 16, 16) # (Batch, 16, 16, 16, 16)
        subap_phases = subap_phases.permute(0, 1, 2, 3, 4).contiguous()
        subap_phases = subap_phases.view(-1, 16, 16) # (Batch * 256, 16, 16)
        
        # 3. Compute diffraction subaperture PSFs using PyTorch FFT2
        # E(x, y) = A(x, y) * exp(i * phi(x, y))
        # Since subapertures are square, amplitude mask is 1 inside the sub-frame.
        complex_sub = torch.complex(torch.cos(subap_phases), torch.sin(subap_phases))
        focal_field = torch.fft.fft2(complex_sub)
        focal_field = torch.fft.fftshift(focal_field, dim=(-2, -1))
        
        # PSF = |E_focal|^2
        psf = torch.abs(focal_field) ** 2
        psf_sum = torch.sum(psf, dim=(-2, -1), keepdim=True) + 1e-8
        psf = psf / psf_sum
        
        # 4. FFT Convolution or direct PSF mapping depending on mode
        if target_mode == 'solar':
            scene_fft = torch.fft.fft2(granulation_patches)
            psf_fft = torch.fft.fft2(psf)
            convolved_fft = scene_fft * psf_fft
            convolved = torch.real(torch.fft.ifft2(convolved_fft))
            convolved = torch.fft.fftshift(convolved, dim=(-2, -1))
        else:
            # For a point star guide source, convolving with a delta function
            # yields the diffraction spot PSF itself.
            convolved = psf
            
        # Normalize and match flux of the simulated images with the observed raw subimages
        convolved_sum = torch.sum(convolved, dim=(-2, -1), keepdim=True) + 1e-8
        raw_flux = torch.sum(raw_subimages, dim=(-2, -1), keepdim=True)
        simulated_subimages = convolved / convolved_sum * raw_flux + background_pedestal
        
        # 5. Compute self-supervised loss (MSE of images)
        loss = torch.mean((simulated_subimages - raw_subimages) ** 2)
        return loss, simulated_subimages
