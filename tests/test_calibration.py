import unittest
import numpy as np
import hcipy
from vajra import config
from vajra.simulator import AOPipelineSimulator
from vajra.calibration import LoopCalibrationMonitor, InfluenceFunctionCalibrator, SLODARAnalyzer

class TestCalibrationComponents(unittest.TestCase):
    def setUp(self):
        self.num_actuators = config.DM_ACTUATORS_TOTAL
        self.num_zernikes = config.ZERNIKE_MODES_MAX
        self.monitor = LoopCalibrationMonitor(self.num_actuators, self.num_zernikes)
        self.calibrator = InfluenceFunctionCalibrator(self.num_actuators)
        self.slodar = SLODARAnalyzer()

    def test_loop_calibration_monitor_thermal(self):
        curr_cmd = np.zeros(self.num_actuators)
        residual = np.zeros(self.num_zernikes)
        
        # Ingest with high thermal drift
        drifted_modes, needs_repoke = self.monitor.monitor_drift(curr_cmd, residual, telescope_temp=21.0)
        
        self.assertTrue(needs_repoke)
        self.assertIn(2, drifted_modes) # Tip
        self.assertIn(4, drifted_modes) # Defocus

    def test_loop_calibration_monitor_correlation(self):
        # Fill buffer with highly correlated commands and residuals (simulating drift)
        for _ in range(self.monitor.buffer_size):
            cmd = np.random.normal(0, 1.0, size=self.num_actuators)
            res = np.zeros(self.num_zernikes)
            # Make Zernike mode 5 (astigmatism) track actuator 0's command
            res[5] = cmd[0] * 0.5 + np.random.normal(0, 0.05)
            self.monitor.monitor_drift(cmd, res, telescope_temp=20.0)
            
        cmd = np.zeros(self.num_actuators)
        res = np.zeros(self.num_zernikes)
        drifted_modes, needs_repoke = self.monitor.monitor_drift(cmd, res, telescope_temp=20.0)
        
        self.assertTrue(needs_repoke)
        self.assertIn(5, drifted_modes)

    def test_influence_function_calibrator(self):
        sim = AOPipelineSimulator()
        geom_matrix = np.random.normal(0, 0.1, size=(2 * len(sim.subaps_pos), config.ZERNIKE_MODES_MAX))
        
        drift_detected, updated_matrix = self.calibrator.run_influence_check(sim, geom_matrix)
        
        self.assertEqual(updated_matrix.shape, geom_matrix.shape)
        self.assertEqual(self.calibrator.current_calibration_actuator, 1)

    def test_slodar_analyzer(self):
        # Create uniform slopes
        c1 = np.ones((config.MLA_GRID_SIZE**2, 2)) * 0.5
        c2 = np.ones((config.MLA_GRID_SIZE**2, 2)) * 0.5
        
        gl, ha = self.slodar.analyze_slodar(c1, c2)
        
        # In a fully correlated uniform case, GL fraction should be high
        self.assertGreaterEqual(gl, 0.0)
        self.assertLessEqual(gl, 1.0)
        self.assertEqual(gl + ha, 1.0)

if __name__ == '__main__':
    unittest.main()
