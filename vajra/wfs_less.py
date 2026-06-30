import numpy as np
from vajra import config

class SPGDOptimizer:
    """Wavefront-sensorless optimizer using Stochastic Parallel Gradient Descent (SPGD).
    Directly maximizes focal plane science image sharpness/contrast without WFS measurement.
    """
    def __init__(self, num_actuators=config.DM_ACTUATORS_TOTAL, learning_rate=0.1, perturbation_amplitude=1e-8):
        self.num_actuators = num_actuators
        self.gamma = learning_rate # Gain/learning rate
        self.delta_u_amplitude = perturbation_amplitude # Base perturbation step size
        self.prev_metric = 0.0

    def compute_sharpness_metric(self, science_image):
        """Computes the sharpness metric: J = sum(I^2).
        Higher J corresponds to a cleaner, more diffraction-limited image.
        """
        # Ensure image is normalized to avoid amplitude scaling issues
        img = science_image.astype(np.float32)
        norm_img = img / (np.sum(img) + 1e-8)
        # Intensity squared metric (Standard AO sharpness metric)
        return float(np.sum(norm_img ** 2))

    def generate_perturbation(self, current_metric=None):
        """Generates a random parallel perturbation vector (elements are +/- delta_u)."""
        # Random binary vector (+/- 1)
        rand_vector = np.random.choice([-1.0, 1.0], size=self.num_actuators)
        
        # Dynamic scaling based on contrast metric
        scale = self.delta_u_amplitude
        if current_metric is not None and self.prev_metric > 0:
            # If contrast is already high (close to diffraction limit), reduce step size to fine-tune
            diff = np.abs(current_metric - self.prev_metric)
            if diff < 1e-4:
                scale *= 0.5
                
        perturbation = rand_vector * scale
        return perturbation

    def calculate_update(self, perturbation, metric_plus, metric_minus):
        """Calculates the parameter update based on difference in metrics:
        delta_u_commands = gamma * delta_J * perturbation
        """
        delta_J = metric_plus - metric_minus
        
        # Gradient update vector
        update = self.gamma * delta_J * perturbation
        
        # Store latest metric for historical comparison
        self.prev_metric = (metric_plus + metric_minus) / 2.0
        
        return update

    def run_optimization_step(self, simulator, current_commands):
        """Executes a full two-sided perturbation step in the simulator context.
        Returns the updated DM command.
        """
        # 1. Generate perturbation
        pert = self.generate_perturbation()
        
        # 2. Apply positive perturbation and measure metric
        u_plus = current_commands + pert
        raw_frame_plus, _ = simulator.generate_hartmannogram(dm_commands=u_plus)
        # We crop the central 32x32 area of the frame as a proxy for the science camera focus
        h, w = raw_frame_plus.shape
        crop_plus = raw_frame_plus[h//2-16:h//2+16, w//2-16:w//2+16]
        J_plus = self.compute_sharpness_metric(crop_plus)
        
        # 3. Apply negative perturbation and measure metric
        u_minus = current_commands - pert
        raw_frame_minus, _ = simulator.generate_hartmannogram(dm_commands=u_minus)
        crop_minus = raw_frame_minus[h//2-16:h//2+16, w//2-16:w//2+16]
        J_minus = self.compute_sharpness_metric(crop_minus)
        
        # 4. Calculate gradient update
        dm_update = self.calculate_update(pert, J_plus, J_minus)
        new_commands = current_commands + dm_update
        
        # Ensure commands are within physical limits
        new_commands = np.clip(new_commands, -config.ACTUATOR_STROKE_LIMIT, config.ACTUATOR_STROKE_LIMIT)
        return new_commands
