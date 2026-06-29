import unittest
import numpy as np
from vajra import config
from vajra.controller import (
    MechanicalNotchFilter, 
    RegimeAwarePredictor, 
    GLAODecomposer, 
    ActuatorController,
    ChromaticOffsetModel,
    TemporalMultiplexor
)

class TestControllerComponents(unittest.TestCase):
    def test_mechanical_notch_filter(self):
        filter_sys = MechanicalNotchFilter(sample_rate=1000.0, target_freqs=[50.0])
        t = np.arange(100) / 1000.0
        signal_50hz = np.sin(2 * np.pi * 50.0 * t)
        
        outputs = []
        for s in signal_50hz:
            filtered = filter_sys.filter_signal(np.array([s, s]))
            outputs.append(filtered[0])
            
        outputs = np.array(outputs)
        self.assertLess(np.var(outputs[50:]), np.var(signal_50hz[50:]))

    def test_regime_aware_predictor(self):
        predictor = RegimeAwarePredictor()
        for k in range(30):
            slopes = np.sin(k * 0.2) * np.ones(512) + np.random.normal(0, 0.01, 512)
            predictor.classify_regime(slopes)
            
        # The default initialization has probabilities[0] high or classified
        self.assertEqual(len(predictor.probabilities), 3)
        
        current_z = np.ones(config.ZERNIKE_MODES_MAX) * 1.5
        predicted_z = predictor.predict(current_z, 0.002)
        self.assertEqual(len(predicted_z), config.ZERNIKE_MODES_MAX)

    def test_glao_decomposer(self):
        decomposer = GLAODecomposer()
        centroids = np.ones((config.MLA_GRID_SIZE**2, 2)) * 0.5
        illum = np.ones(config.MLA_GRID_SIZE**2)
        
        gl, ha = decomposer.decompose_slopes(centroids, illum)
        
        # Ground layer should absorb the common mode slopes (tip-tilt)
        self.assertTrue(np.all(np.abs(gl[illum > 0]) > 0.4))
        self.assertTrue(np.all(np.abs(ha[illum > 0]) < 0.1))

    def test_actuator_controller(self):
        # Create a mock geometry matrix matching 18x18 grid coordinates
        geom_matrix = np.random.normal(0, 0.1, size=(2 * config.MLA_GRID_SIZE**2, config.ZERNIKE_MODES_MAX))
        controller = ActuatorController(geom_matrix)
        
        target_z = np.ones(config.ZERNIKE_MODES_MAX) * 0.1
        current_u = np.zeros(config.DM_ACTUATORS_TOTAL)
        
        u, u_comp = controller.calculate_commands(target_z, current_u)
        
        self.assertEqual(len(u), config.DM_ACTUATORS_TOTAL)
        self.assertEqual(len(u_comp), config.DM_ACTUATORS_TOTAL)
        self.assertTrue(np.all(np.abs(u) <= config.ACTUATOR_STROKE_LIMIT))
        self.assertTrue(np.all(np.abs(u_comp) <= config.ACTUATOR_STROKE_LIMIT * 1.2))

    def test_chromatic_offset_model(self):
        model = ChromaticOffsetModel()
        z = np.ones(config.ZERNIKE_MODES_MAX) * 0.1
        offsets = model.get_chromatic_offset(z, zenith_angle_deg=30.0)
        self.assertEqual(len(offsets), config.ZERNIKE_MODES_MAX)
        self.assertGreater(offsets[4], 0.0)

    def test_temporal_multiplexor(self):
        multiplexor = TemporalMultiplexor(num_states=3)
        base = np.ones(10)
        
        c1 = multiplexor.multiplex_command(base)
        c2 = multiplexor.multiplex_command(base)
        c3 = multiplexor.multiplex_command(base)
        
        self.assertNotEqual(c1[0], c2[0])
        self.assertNotEqual(c2[0], c3[0])
        
        c4 = multiplexor.multiplex_command(base)
        self.assertEqual(c1[0], c4[0])

if __name__ == '__main__':
    unittest.main()
