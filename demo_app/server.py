import os
import sys
import io
import base64
import numpy as np
import threading
from PIL import Image
from flask import Flask, jsonify, render_template, request

# Add parent path to import vajra
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vajra import config
from vajra.simulator import AOPipelineSimulator
from vajra.detector import DetectorProcessor
from vajra.reconstructor import WavefrontReconstructor
from vajra.controller import ActuatorController, MechanicalNotchFilter, RegimeAwarePredictor, GLAODecomposer

app = Flask(__name__, template_folder='templates')

class AOLoopState:
    def __init__(self):
        self.lock = threading.Lock()
        self.sim = None
        self.detector = None
        self.reconstructor = None
        self.dm_controller = None
        self.notch_filter = None
        self.regime_predictor = None
        
        self.current_commands = None
        self.applied_commands = None
        self.loop_time = 0.0
        self.rms_history = []
        self.strehl_history = []
        
        self.mode = 'vajra' # 'vajra' or 'classical'
        self.r0 = 0.08
        self.wind_speed = 10.0
        self.target_mode = 'solar' # 'solar' or 'point'
        
    def reset(self, mode='vajra', r0=0.08, wind_speed=10.0, target_mode='solar'):
        self.mode = mode
        self.r0 = r0
        self.wind_speed = wind_speed
        self.target_mode = target_mode
        
        # Instantiate simulator
        self.sim = AOPipelineSimulator(mode=target_mode)
        
        # Configure wind profile
        for idx, layer in enumerate(self.sim.atmosphere_layers):
            layer.velocity = np.array([wind_speed * (1.5 if idx == 0 else 0.5), 0.0])
            
        self.detector = DetectorProcessor()
        self.reconstructor = WavefrontReconstructor(
            self.sim.pupil_grid, self.sim.pupil_mask, self.sim.subaps_pos, self.sim.subap_masks
        )
        self.dm_controller = ActuatorController(
            self.reconstructor.G_full, self.reconstructor.pupil_grid, self.reconstructor.zernike_basis
        )
        self.notch_filter = MechanicalNotchFilter()
        self.regime_predictor = RegimeAwarePredictor()
        
        self.current_commands = np.zeros(self.dm_controller.num_actuators)
        self.applied_commands = np.zeros(self.dm_controller.num_actuators)
        self.loop_time = 0.0
        self.rms_history = []
        self.strehl_history = []

loop_state = AOLoopState()

def to_colormap_base64(arr, cmap_name='inferno'):
    """Maps a 2D numpy array to color map values and converts to base64 encoded PNG."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min > 1e-8:
        norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        norm = np.zeros_like(arr)
        
    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    if cmap_name == 'inferno':
        # Custom approximation of the inferno colormap
        rgb[:, :, 0] = (norm ** 1.5 * 255).astype(np.uint8)
        rgb[:, :, 1] = (norm ** 2.5 * 255).astype(np.uint8)
        rgb[:, :, 2] = (norm ** 4.0 * 255).astype(np.uint8)
    elif cmap_name == 'coolwarm':
        # Blue to red mapping
        rgb[:, :, 0] = (norm * 255).astype(np.uint8)
        rgb[:, :, 1] = (np.sin(norm * np.pi) * 100).astype(np.uint8)
        rgb[:, :, 2] = ((1 - norm) * 255).astype(np.uint8)
    else:
        # Grayscale
        val = (norm * 255).astype(np.uint8)
        rgb[:, :, 0] = val
        rgb[:, :, 1] = val
        rgb[:, :, 2] = val
        
    img = Image.fromarray(rgb, 'RGB')
    
    # Resize small arrays for clear web viewing
    if h < 64:
        img = img.resize((256, 256), Image.Resampling.NEAREST)
        
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/reset', methods=['POST'])
def reset_loop():
    with loop_state.lock:
        data = request.json or {}
        mode = data.get('mode', 'vajra')
        r0 = float(data.get('r0', 0.08))
        wind_speed = float(data.get('wind_speed', 10.0))
        target_mode = data.get('target_mode', 'solar')
        
        loop_state.reset(mode=mode, r0=r0, wind_speed=wind_speed, target_mode=target_mode)
        return jsonify({"status": "success", "mode": mode, "r0": r0, "wind_speed": wind_speed, "target_mode": target_mode})

@app.route('/api/run_step', methods=['POST'])
def run_step():
    with loop_state.lock:
        if loop_state.sim is None:
            loop_state.reset()
            
        loop_state.loop_time += 0.001
        loop_state.sim.evolve_atmosphere(loop_state.loop_time)
        
        # 1. Physical Kolmogorov phase screen scaling based on r0
        scale = (0.08 / loop_state.r0) ** (5.0 / 6.0)
        true_phase = loop_state.sim.get_phase_screen() * scale
        
        # 2. Synthesize Shack-Hartmann spot frame (solar granulation convolution)
        raw_frame, illum = loop_state.sim.generate_hartmannogram(dm_commands=loop_state.applied_commands)
        
        # 3. Detect spots centroids
        centroids, fluxes, fwhms, confidences = loop_state.detector.process_frame(raw_frame, illum)
        
        # 4. Wavefront Control Logic
        if loop_state.mode == 'vajra':
            # VAJRA Reconstructor & controller
            z_wls, uarc_vars = loop_state.reconstructor.reconstruct_wls(centroids, confidences, illum, prev_r0=loop_state.r0)
            loop_state.regime_predictor.classify_regime(centroids.flatten())
            predicted_z = loop_state.regime_predictor.predict(z_wls, config.NOMINAL_LATENCY)
            
            # GLAO decomposition
            glao_slopes = GLAODecomposer().get_controlled_slopes(centroids, illum)
            z_glao, _ = loop_state.reconstructor.reconstruct_wls(glao_slopes, confidences, illum, prev_r0=loop_state.r0)
            z_ctrl = 0.7 * predicted_z + 0.3 * z_glao
            
            # GRU Neural Hysteresis compensation & Saturation MLP checks
            loop_state.current_commands, comp_commands = loop_state.dm_controller.calculate_commands(z_ctrl, loop_state.current_commands)
            comp_commands[:2] = loop_state.notch_filter.filter_signal(comp_commands[:2])
            loop_state.applied_commands = comp_commands
        else:
            # Classical baseline integrator loop
            z_wls, _ = loop_state.reconstructor.reconstruct_wls(centroids, confidences, illum)
            u_target = loop_state.dm_controller.zernike_to_command @ z_wls
            loop_state.current_commands = loop_state.current_commands - 0.4 * u_target
            loop_state.current_commands = np.clip(loop_state.current_commands, -loop_state.dm_controller.stroke_limit, loop_state.dm_controller.stroke_limit)
            loop_state.current_commands[:2] = loop_state.notch_filter.filter_signal(loop_state.current_commands[:2])
            loop_state.applied_commands = loop_state.current_commands.copy()
            
        # 5. Compute residual phase screen
        dm_phase = loop_state.sim.dm.phase_for(config.LAMBDA_SENSING)
        residual_phase = true_phase + dm_phase
        
        # Crop phase screen to 2D pupil arrays
        pupil_2d = loop_state.sim.pupil_mask.reshape(256, 256)
        atm_2d = true_phase.reshape(256, 256) * pupil_2d
        res_2d = residual_phase.reshape(256, 256) * pupil_2d
        
        # Calculate Loop stats
        pupil_idx = (pupil_2d > 0)
        rms_val = float(np.std(res_2d[pupil_idx]))
        strehl_val = float(np.exp(-rms_val**2))
        
        loop_state.rms_history.append(rms_val)
        loop_state.strehl_history.append(strehl_val)
        if len(loop_state.rms_history) > 100:
            loop_state.rms_history.pop(0)
            loop_state.strehl_history.pop(0)
            
        # Convert arrays to colored base64 images
        atm_img = to_colormap_base64(atm_2d, 'coolwarm')
        res_img = to_colormap_base64(res_2d, 'coolwarm')
        wfs_img = to_colormap_base64(raw_frame, 'inferno')
        
        # Actuator deformations map
        dm_grid = loop_state.applied_commands.reshape(18, 18)
        dm_img = to_colormap_base64(dm_grid, 'gray')
        
        regime = "Steady"
        if loop_state.mode == 'vajra':
            idx = np.argmax(loop_state.regime_predictor.probabilities)
            regime = ["Steady", "Transitioning", "Broken"][idx]
            
        return jsonify({
            "rms": float(rms_val),
            "strehl": float(strehl_val),
            "regime": regime,
            "atm_img": atm_img,
            "res_img": res_img,
            "wfs_img": wfs_img,
            "dm_img": dm_img,
            "rms_history": loop_state.rms_history,
            "strehl_history": loop_state.strehl_history
        })

if __name__ == '__main__':
    # Force loading FITS before starting server
    loop_state.reset()
    app.run(host='127.0.0.1', port=5000, debug=False)
