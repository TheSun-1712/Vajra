import unittest
import os
import numpy as np
import torch
from vajra.controller import RecurrentWavefrontPredictor, RegimeAwarePredictor
from vajra.differentiable_optics import DifferentiablePropagator
from vajra import config
from run_vajra import quantize_model_dynamic, run_loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestAdvancedOptimizations(unittest.TestCase):
    def test_recurrent_gru_predictor(self):
        predictor = RecurrentWavefrontPredictor(input_dim=config.ZERNIKE_MODES_MAX, hidden_dim=16).to(device)
        predictor.eval()
        
        # Ingest state history: (Batch=1, SeqLen=8, ZernikeDim=66)
        dummy_history = torch.randn(1, 8, config.ZERNIKE_MODES_MAX, device=device)
        pred, h_next = predictor(dummy_history)
        
        self.assertEqual(pred.shape, (1, config.ZERNIKE_MODES_MAX))
        self.assertEqual(h_next.shape, (1, 1, 16))

    def test_dynamic_quantization(self):
        # Create a mock model file to quantize
        predictor = RecurrentWavefrontPredictor(input_dim=config.ZERNIKE_MODES_MAX, hidden_dim=16)
        mock_path = "data/mock_predictor.pth"
        quant_path = "data/mock_predictor_quant.pth"
        
        torch.save(predictor.state_dict(), mock_path)
        
        # Test quantize function
        success = quantize_model_dynamic(mock_path, quant_path)
        
        # Clean up files
        if os.path.exists(mock_path):
            os.remove(mock_path)
        if os.path.exists(quant_path):
            os.remove(quant_path)
            
        self.assertTrue(success)

    def test_multi_rate_loop(self):
        # Run a small multi-rate simulation loop (10 frames)
        metrics = run_loop(sim_mode="solar", condition="steady", n_frames=10, multi_rate=True)
        self.assertIn("closed_rms", metrics)
        self.assertGreater(metrics["closed_rms"], 0)

    def test_differentiable_propagator_gradients(self):
        # Setup dummy inputs
        batch_size = 2
        zernike_basis = np.random.normal(0, 0.1, size=(config.ZERNIKE_MODES_MAX, 256 * 256))
        pupil_mask = np.ones(256 * 256)
        
        propagator = DifferentiablePropagator(pupil_mask, zernike_basis)
        
        # Dynamic inputs with gradients enabled
        predicted_z = torch.randn(batch_size, config.ZERNIKE_MODES_MAX, requires_grad=True, device=device)
        granulation = torch.randn(batch_size * 256, 16, 16, device=device)
        raw_subs = torch.randn(batch_size * 256, 16, 16, device=device)
        
        loss, simulated = propagator(
            predicted_zernikes=predicted_z,
            granulation_patches=granulation,
            raw_subimages=raw_subs,
            target_mode="solar"
        )
        
        self.assertGreater(loss.item(), 0)
        
        # Call backward and assert gradients flow
        loss.backward()
        self.assertIsNotNone(predicted_z.grad)
        self.assertEqual(predicted_z.grad.shape, predicted_z.shape)

if __name__ == "__main__":
    unittest.main()
