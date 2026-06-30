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
            self.sim.pupil_grid, self.sim.pupil_mask, self.sim.subaps_pos, self.sim.subap_masks, target_mode=target_mode
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

def to_colormap_base64(arr, cmap_name='inferno', mask_2d=None):
    """Maps a 2D numpy array to color map values and converts to base64 encoded PNG."""
    if mask_2d is not None:
        mask_idx = (mask_2d > 0)
        arr_masked = arr[mask_idx]
        if len(arr_masked) > 0:
            arr_min, arr_max = arr_masked.min(), arr_masked.max()
        else:
            arr_min, arr_max = 0.0, 1.0
            
        if arr_max - arr_min > 1e-8:
            norm = (arr - arr_min) / (arr_max - arr_min)
        else:
            norm = np.zeros_like(arr)
        norm = np.clip(norm, 0.0, 1.0)
    else:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min > 1e-8:
            norm = (arr - arr_min) / (arr_max - arr_min)
        else:
            norm = np.zeros_like(arr)
        
    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    if cmap_name == 'inferno':
        # Apply a gamma stretch to boost visibility of the low-intensity details (halo, Airy rings)
        norm_clipped = np.clip(norm, 0.0, 1.0)
        norm_stretch = norm_clipped ** 0.5
        # Custom approximation of the inferno colormap
        rgb[:, :, 0] = (norm_stretch * 255).astype(np.uint8)
        rgb[:, :, 1] = (norm_stretch ** 2.5 * 255).astype(np.uint8)
        rgb[:, :, 2] = (norm_stretch ** 6.0 * 255).astype(np.uint8)
    elif cmap_name == 'coolwarm':
        # High contrast white-balanced coolwarm colormap
        rgb[:, :, 0] = (norm * 255).astype(np.uint8)
        rgb[:, :, 1] = (np.sin(norm * np.pi) * 120 + (1.0 - np.abs(norm - 0.5)*2.0)*80).astype(np.uint8)
        rgb[:, :, 2] = ((1 - norm) * 255).astype(np.uint8)
    else:
        # Grayscale
        val = (norm * 255).astype(np.uint8)
        rgb[:, :, 0] = val
        rgb[:, :, 1] = val
        rgb[:, :, 2] = val
        
    # Mask out-of-pupil background pixels to card dark blue
    if mask_2d is not None:
        rgb[~mask_idx] = [10, 15, 26]
        
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
    from flask import make_response
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

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
        atm_img = to_colormap_base64(atm_2d, 'coolwarm', pupil_2d)
        res_img = to_colormap_base64(res_2d, 'coolwarm', pupil_2d)
        wfs_img = to_colormap_base64(raw_frame, 'inferno')
        
        # Actuator deformations map
        dm_grid = loop_state.applied_commands.reshape(18, 18)
        dm_mask = np.zeros((18, 18), dtype=bool)
        for r in range(18):
            for c in range(18):
                if (r - 8.5)**2 + (c - 8.5)**2 <= 9.0**2:
                    dm_mask[r, c] = True
        dm_img = to_colormap_base64(dm_grid, 'gray', dm_mask)
        
        regime = "Steady"
        if loop_state.mode == 'vajra':
            idx = np.argmax(loop_state.regime_predictor.probabilities)
            regime = ["Steady", "Transitioning", "Broken"][idx]
            
        # ── Grid 1: Atmosphere HO wavefront (tip-tilt removed) ─────────────────
        # Used by the "3D Reconstructed Wavefront (HO)" panel.
        atm_ds   = atm_2d[::8, ::8]      # (32, 32)
        pupil_ds = pupil_2d[::8, ::8]    # (32, 32) binary mask
        
        rows_ds, cols_ds = atm_ds.shape
        yy_ds, xx_ds = np.mgrid[0:rows_ds, 0:cols_ds]
        
        pupil_flat = (pupil_ds > 0).flatten()
        if pupil_flat.sum() > 6:
            A_fit = np.column_stack([
                xx_ds.flatten()[pupil_flat],
                yy_ds.flatten()[pupil_flat],
                np.ones(pupil_flat.sum())
            ])
            b_fit = atm_ds.flatten()[pupil_flat]
            coeffs, _, _, _ = np.linalg.lstsq(A_fit, b_fit, rcond=None)
            plane = coeffs[0] * xx_ds + coeffs[1] * yy_ds + coeffs[2]
            atm_ho = atm_ds - plane
        else:
            atm_ho = atm_ds

        pupil_vals = atm_ho[pupil_ds > 0]
        if len(pupil_vals) > 0:
            atm_mean = float(pupil_vals.mean())
            atm_std  = float(pupil_vals.std()) if float(pupil_vals.std()) > 1e-9 else 1.0
        else:
            atm_mean, atm_std = 0.0, 1.0
        atm_norm = (atm_ho - atm_mean) / atm_std

        atm_grid = [
            [round(float(atm_norm[r, c]), 4) if pupil_ds[r, c] > 0 else None
             for c in range(cols_ds)]
            for r in range(rows_ds)
        ]

        # ── Grid 2: Full atmosphere phase (no tip-tilt removal) ────────────────
        # Used by the "Atmosphere Phase Screen" 3D panel.
        # Shows the raw Kolmogorov turbulence including tip-tilt.
        pupil_vals_full = atm_ds[pupil_ds > 0]
        if len(pupil_vals_full) > 0:
            af_mean = float(pupil_vals_full.mean())
            af_std  = float(pupil_vals_full.std()) if float(pupil_vals_full.std()) > 1e-9 else 1.0
        else:
            af_mean, af_std = 0.0, 1.0
        atm_full_norm = (atm_ds - af_mean) / af_std

        atm_full_grid = [
            [round(float(atm_full_norm[r, c]), 4) if pupil_ds[r, c] > 0 else None
             for c in range(cols_ds)]
            for r in range(rows_ds)
        ]

        # ── Grid 3: DM actuator command surface ────────────────────────────────
        # Used by the "DM Actuator Map" 3D panel.
        # The DM surface shape IS a physical 3D mirror deformation.
        dm_vals = loop_state.applied_commands  # shape (N_act,) flat
        dm_g = dm_grid.copy().astype(float)   # (18, 18)
        dm_valid = dm_mask.copy()              # (18, 18) boolean
        dm_active = dm_g[dm_valid]
        if len(dm_active) > 0:
            dm_mean = float(dm_active.mean())
            dm_std  = float(dm_active.std()) if float(dm_active.std()) > 1e-9 else 1.0
        else:
            dm_mean, dm_std = 0.0, 1.0
        dm_norm = (dm_g - dm_mean) / dm_std

        dm_grid_3d = [
            [round(float(dm_norm[r, c]), 4) if dm_mask[r, c] else None
             for c in range(18)]
            for r in range(18)
        ]
            
        return jsonify({
            "rms": float(rms_val),
            "strehl": float(strehl_val),
            "regime": regime,
            "atm_img": atm_img,
            "res_img": res_img,
            "wfs_img": wfs_img,
            "dm_img": dm_img,
            "rms_history": loop_state.rms_history,
            "strehl_history": loop_state.strehl_history,
            "res_grid":      atm_grid,       # HO wavefront (tip-tilt removed)
            "atm_full_grid": atm_full_grid,  # Raw atmosphere phase
            "dm_grid_3d":    dm_grid_3d      # DM mirror surface shape
        })

if __name__ == '__main__':
    # Force loading FITS before starting server
    loop_state.reset()
    app.run(host='127.0.0.1', port=5000, debug=False)
