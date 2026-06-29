import unittest
import numpy as np
import torch
import hcipy
from vajra import config
from vajra.differentiable_optics import DifferentiablePropagator
from vajra.controller import NeuralHysteresisModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestAdvancedFeatures(unittest.TestCase):
    def setUp(self):
        # Create minimal pupil grid and basis for quick testing
        self.pupil_grid = hcipy.make_pupil_grid(256, config.D_APERTURE)
        self.pupil_mask = hcipy.make_circular_aperture(config.D_APERTURE)(self.pupil_grid)
        self.zernike_basis = hcipy.make_zernike_basis(
            config.ZERNIKE_MODES_MAX + 1, config.D_APERTURE, self.pupil_grid, starting_mode=2
        )
        
    def test_differentiable_propagator(self):
        propagator = DifferentiablePropagator(self.pupil_mask, self.zernike_basis[:config.ZERNIKE_MODES_MAX])
        
        # Ingest inputs
        batch_size = 2
        pred_z = torch.randn(batch_size, config.ZERNIKE_MODES_MAX, device=device) * 0.1
        pred_z.requires_grad = True
        
        # 256 subapertures convolved with 16x16 solar patches
        gran_patches = torch.rand(batch_size * 256, 16, 16, device=device)
        raw_subs = torch.rand(batch_size * 256, 16, 16, device=device)
        
        loss, simulated = propagator(pred_z, gran_patches, raw_subs)
        
        # Verify output shape
        self.assertEqual(simulated.shape, (batch_size * 256, 16, 16))
        
        # Verify gradient flow: backpropagate the self-supervised image loss
        loss.backward()
        self.assertIsNotNone(pred_z.grad)
        # Gradient should not be all zeros
        self.assertTrue(torch.any(pred_z.grad != 0.0))

    def test_neural_hysteresis_model(self):
        num_act = config.DM_ACTUATORS_TOTAL
        model = NeuralHysteresisModel(num_actuators=num_act).to(device)
        
        # Sequence input: Batch=2, Seq=5, Channels=num_act
        u_seq = torch.rand(2, 5, num_act, requires_grad=True, device=device)
        
        comp_seq, h_next = model(u_seq)
        
        # Verify shape
        self.assertEqual(comp_seq.shape, (2, 5, num_act))
        self.assertEqual(h_next.shape, (1, 2, 64)) # (num_layers, batch, hidden_size)
        
        # Verify backpropagation
        loss = torch.mean(comp_seq ** 2)
        loss.backward()
        
        self.assertIsNotNone(u_seq.grad)
        self.assertTrue(torch.any(u_seq.grad != 0.0))

if __name__ == '__main__':
    unittest.main()
