import unittest
import numpy as np
import torch
from vajra.rl_agent import VajraAOEnv, SACAgent, SimpleReplayBuffer
from vajra.controller import ActuatorController
from vajra import config

class TestRLMetaController(unittest.TestCase):
    def test_env_step_reset(self):
        env = VajraAOEnv(sim_mode="solar", history_len=3)
        obs, info = env.reset()
        
        # Grid size (16x16) * 3 variables (centroids x, centroids y, confidences) * 3 history frames
        expected_shape = (16 * 16 * 3 * 3,)
        self.assertEqual(obs.shape, expected_shape)
        
        # Test step executing
        action = np.array([0.5, 0.8, 0.95], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        self.assertEqual(next_obs.shape, expected_shape)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated) # The episode is not yet truncated on step 1

    def test_shear_protection_clipping(self):
        # Instantiate controller
        reconstructor_matrix = np.zeros((256 * 2, config.ZERNIKE_MODES_MAX))
        controller = ActuatorController(reconstructor_matrix)
        
        # Create extreme adjacent actuator command differences
        bad_commands = np.zeros(config.DM_ACTUATORS_TOTAL)
        bad_commands[0] = config.ACTUATOR_STROKE_LIMIT
        bad_commands[1] = -config.ACTUATOR_STROKE_LIMIT
        
        clipped = controller.clip_interactuator_shear(bad_commands)
        
        # The difference should be capped to 30% of stroke limit
        diff = np.abs(clipped[0] - clipped[1])
        max_allowed_diff = 0.3 * config.ACTUATOR_STROKE_LIMIT
        self.assertLessEqual(diff, max_allowed_diff + 1e-9)

    def test_sac_policy_selection(self):
        state_dim = 16 * 16 * 3 * 5
        agent = SACAgent(state_dim=state_dim, action_dim=3)
        
        dummy_state = np.random.normal(0, 0.1, size=(state_dim,))
        action = agent.select_action(dummy_state, evaluate=True)
        
        # Action shapes loop gain, glao weight, notch r
        self.assertEqual(action.shape, (3,))
        self.assertTrue(0.1 <= action[0] <= 1.2)
        self.assertTrue(0.0 <= action[1] <= 1.0)
        self.assertTrue(0.90 <= action[2] <= 0.99)

if __name__ == "__main__":
    unittest.main()
