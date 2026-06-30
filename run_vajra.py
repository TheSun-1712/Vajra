import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure local package path is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vajra import config
from vajra.simulator import AOPipelineSimulator
from vajra.detector import DetectorProcessor
from vajra.reconstructor import WavefrontReconstructor
from vajra.controller import (
    ActuatorController,
    MechanicalNotchFilter,
    RegimeAwarePredictor,
    GLAODecomposer,
    TemporalMultiplexor,
    ChromaticOffsetModel
)
from vajra.calibration import LoopCalibrationMonitor, InfluenceFunctionCalibrator
from vajra.science import NCPATracker, PSFReconstructor

def run_loop(sim_mode="solar", condition="steady", use_vajra=True, n_frames=100,
             use_rl=False, use_spgd=False, load_telemetry=False, ra=0.0, dec=0.0,
             multi_rate=False):
    """Runs the closed-loop simulation under specified parameters and condition."""
    # Reset random seeds for fair comparison
    np.random.seed(42)
    
    # 1. Initialize VAJRA/SURYA pipeline components
    sim = AOPipelineSimulator(mode=sim_mode)
    detector = DetectorProcessor()
    reconstructor = WavefrontReconstructor(
        sim.pupil_grid, sim.pupil_mask, sim.subaps_pos, sim.subap_masks, target_mode=sim_mode
    )
    dm_controller = ActuatorController(
        reconstructor.G_full, reconstructor.pupil_grid, reconstructor.zernike_basis
    )
    notch_filter = MechanicalNotchFilter()
    regime_predictor = RegimeAwarePredictor()
    glao = GLAODecomposer()
    multiplexor = TemporalMultiplexor()
    chromatic_model = ChromaticOffsetModel()
    
    # Background calibration monitors
    calib_monitor = LoopCalibrationMonitor(dm_controller.num_actuators, reconstructor.num_zernikes)
    influence_calibrator = InfluenceFunctionCalibrator(dm_controller.num_actuators)
    ncpa_tracker = NCPATracker(num_zernikes=reconstructor.num_zernikes)
    psf_reconstructor = PSFReconstructor(sim.pupil_grid, reconstructor.zernike_basis)
    
    # 2. Configure condition parameters
    if condition == "steady":
        # Constant seeing and uniform slow wind
        wind_speeds = [10.0, 3.0]
        r0 = 0.08
    elif condition == "transition":
        # Changing wind profile
        wind_speeds = [25.0, 10.0]
        r0 = 0.07
    elif condition == "strong":
        # High D/r0 (bad seeing conditions)
        wind_speeds = [18.0, 6.0]
        r0 = 0.04
    else:
        wind_speeds = [15.0, 5.0]
        r0 = 0.08
        
    # Update simulator layer wind speeds
    for idx, layer in enumerate(sim.atmosphere_layers):
        v_spd = wind_speeds[idx]
        v_dir = config.WIND_DIRECTIONS[idx]
        if condition == "transition" and idx == 0:
            # Add sudden wind direction shear for transition testing
            v_dir += np.pi / 2
        layer.velocity = np.array([v_spd * np.cos(v_dir), v_spd * np.sin(v_dir)])
        
    # 3. Initialize advanced additions
    g_mag = None
    if sim_mode == "point" and (ra != 0.0 or dec != 0.0):
        from vajra.dataset_loader import StarCatalogQuery
        query = StarCatalogQuery()
        star_info = query.query_nearest_guide_star(ra, dec)
        g_mag = star_info["phot_g_mean_mag"]
        print(f"[Gaia Catalog] Target coordinates RA={ra}, Dec={dec} matched guide star G-mag: {g_mag:.2f}")

    if load_telemetry:
        from vajra.dataset_loader import TelemetryReplayer
        replayer = TelemetryReplayer()

    if use_spgd:
        from vajra.wfs_less import SPGDOptimizer
        spgd = SPGDOptimizer(perturbation_amplitude=1e-7)

    if use_rl:
        from vajra.rl_agent import SACAgent, SimpleReplayBuffer
        rl_agent = SACAgent(state_dim=config.MLA_GRID_SIZE**2 * 3 * 5, action_dim=3)
        rl_buffer = SimpleReplayBuffer(max_size=2000)
        rl_history = []
        
    # Initialize metrics vectors
    rms_open_loop = []
    rms_closed_loop = []
    strehl_history = []
    regime_steady_probs = []
    regime_trans_probs = []
    regime_broken_probs = []
    saturation_counts = []
    latencies = []
    
    current_commands = np.zeros(dm_controller.num_actuators)
    applied_commands = np.zeros(dm_controller.num_actuators)
    registration_drift_offset = np.zeros(2)
    telescope_temp = 20.0
    
    loop_time = 0.0
    dt_latency = config.NOMINAL_LATENCY
    
    # 4. Main Closed-Loop Run
    for frame in range(1, n_frames + 1):
        # Initialize default Zernikes for intermediate multi-rate loop steps
        z_wls = np.zeros(reconstructor.num_zernikes)
        uarc_vars = np.ones(reconstructor.num_zernikes)
        
        # Evolve atmosphere
        loop_time += 0.001
        sim.evolve_atmosphere(loop_time)
        
        # Site conditions temperature fluctuations (Problem 10 thermal NCPA)
        telescope_temp += np.random.normal(0.0, 0.005) # slow random walk
        
        # MLA to detector registration thermal drift (Problem 19)
        registration_drift_offset += np.random.normal(0.0, 0.0005, size=2)
        
        # Calculate instantaneous open-loop atmospheric phase error
        true_phase = sim.get_phase_screen()
        pupil_idx = (sim.pupil_mask > 0)
        rms_open = np.std(true_phase[pupil_idx])
        rms_open_loop.append(rms_open)
        
        # Simulate mechanical mount vibration acceleration feed-forward (Problem 14)
        mount_jitter = np.sin(frame * 0.25) * np.array([0.5, 0.5])
        
        # Wavelength photon flux scaling from Gaia magnitude query
        flux_level = config.PHOTON_FLUX_NOMINAL
        if g_mag is not None:
            flux_level = query.calculate_photon_flux(g_mag)
            
        t_start = time.perf_counter()
        
        # Multi-rate control loop partitioning (Optimizations 4)
        is_dm_frame = True
        if multi_rate and not use_spgd:
            # Tip-tilt mirror (TTM) updated every frame, Deformable Mirror (DM) updated every N frames
            is_dm_frame = (frame % getattr(config, 'MULTI_RATE_RATIO', 2) == 0)
            
        # Multi-modal branching
        if use_spgd:
            # WFS-less optimization step (sharpness maximization directly)
            applied_commands = spgd.run_optimization_step(sim, applied_commands)
            centroids = np.zeros((config.MLA_GRID_SIZE**2, 2))
            confidences = np.ones(config.MLA_GRID_SIZE**2)
            illum = np.ones(config.MLA_GRID_SIZE**2)
        elif load_telemetry:
            # Telemetry replay step
            centroids, current_commands, is_emulated = replayer.load_telemetry_frame()
            illum = np.ones(config.MLA_GRID_SIZE**2)
            confidences = np.ones(config.MLA_GRID_SIZE**2)
            applied_commands = current_commands.copy()
        else:
            # Generate raw WFS Hartmannogram image convolved with solar granulation
            raw_frame, illum = sim.generate_hartmannogram(
                dm_commands=applied_commands,
                registration_offset=registration_drift_offset,
                flux_nominal=flux_level
            )
            
            # WFS Centroid processing
            centroids, fluxes, fwhms, confidences = detector.process_frame(raw_frame, illum)

        if use_rl and not use_spgd:
            # --- RL META-CONTROLLER LOOP ---
            if not is_dm_frame:
                # Fast tip-tilt loop (1 kHz)
                z_tt, _ = reconstructor.reconstruct_wls(centroids, confidences, illum)
                z_tt_only = np.zeros(reconstructor.num_zernikes)
                z_tt_only[:2] = z_tt[:2]
                u_tt = dm_controller.zernike_to_command @ z_tt_only
                current_commands[:2] = current_commands[:2] - 0.5 * u_tt[:2]
                current_commands = dm_controller.clip_interactuator_shear(current_commands)
                applied_commands[:2] = current_commands[:2]
            else:
                # High-order loop updates (500 Hz)
                obs_frame = np.concatenate([centroids.flatten(), confidences])
                rl_history.append(obs_frame)
                if len(rl_history) > 5:
                    rl_history.pop(0)
                while len(rl_history) < 5:
                    rl_history.append(obs_frame)
                    
                obs = np.concatenate(rl_history)
                
                # Agent outputs gains & parameters: loop_gain, glao_weight, notch_r
                action = rl_agent.select_action(obs, evaluate=False)
                loop_gain, glao_weight, notch_r = action
                
                # Apply dynamic parameters
                dm_controller.set_loop_parameters(
                    modal_gains=np.ones(reconstructor.num_zernikes) * loop_gain,
                    glao_weight=glao_weight
                )
                # Update notch filter r coefficient
                notch_filter.fs = 1000.0 * (notch_r / 0.98)
                
                # Run reconstructor logic
                z_wls, uarc_vars = reconstructor.reconstruct_wls(centroids, confidences, illum, prev_r0=r0)
                regime_predictor.classify_regime(centroids.flatten())
                
                # Wind-shake vibration feed-forward compensation (Problem 14)
                predicted_z = regime_predictor.predict(z_wls, dt_latency, mount_acceleration=mount_jitter)
                
                controlled_slopes = glao.get_controlled_slopes(centroids, illum)
                z_glao, _ = reconstructor.reconstruct_wls(controlled_slopes, confidences, illum, prev_r0=r0)
                
                z_ctrl = 0.7 * predicted_z + 0.3 * z_glao
                
                current_commands, comp_commands = dm_controller.calculate_commands(z_ctrl, current_commands)
                comp_commands[:2] = notch_filter.filter_signal(comp_commands[:2])
                applied_commands = multiplexor.multiplex_command(comp_commands)
                
                # Log transition to train the SAC agent
                next_obs_frame = np.concatenate([centroids.flatten(), confidences])
                next_rl_history = list(rl_history[1:]) + [next_obs_frame]
                next_obs = np.concatenate(next_rl_history)
                
                dm_phase = sim.dm.phase_for(config.LAMBDA_SENSING)
                corrected_phase = (true_phase + dm_phase) * sim.pupil_mask
                rms_closed = np.std(corrected_phase[pupil_idx])
                reward = -float(rms_closed)
                
                rl_buffer.add(obs, action, reward, next_obs, False)
                rl_agent.train_step(rl_buffer, batch_size=4)
            
        elif use_vajra and not use_spgd:
            # --- STANDARD VAJRA PIPELINE ---
            if not is_dm_frame:
                # Fast tip-tilt loop (1 kHz)
                z_tt, _ = reconstructor.reconstruct_wls(centroids, confidences, illum)
                z_tt_only = np.zeros(reconstructor.num_zernikes)
                z_tt_only[:2] = z_tt[:2]
                u_tt = dm_controller.zernike_to_command @ z_tt_only
                current_commands[:2] = current_commands[:2] - 0.5 * u_tt[:2]
                current_commands = dm_controller.clip_interactuator_shear(current_commands)
                applied_commands[:2] = current_commands[:2]
            else:
                # High-order loop updates (500 Hz)
                # Wavefront Reconstruction
                z_wls, uarc_vars = reconstructor.reconstruct_wls(centroids, confidences, illum, prev_r0=r0)
                
                # Autocorrelation-based regime classification (Problem 8)
                regime_predictor.classify_regime(centroids.flatten())
                regime_steady_probs.append(regime_predictor.probabilities[0])
                regime_trans_probs.append(regime_predictor.probabilities[1])
                regime_broken_probs.append(regime_predictor.probabilities[2])
                
                # Dynamic Zernike prediction with mount vibration feed-forward (Problem 14)
                predicted_z = regime_predictor.predict(z_wls, dt_latency, mount_acceleration=mount_jitter)
                
                # Ground-layer / High-altitude separation (Problem 11 & 23)
                controlled_slopes = glao.get_controlled_slopes(centroids, illum)
                z_glao, _ = reconstructor.reconstruct_wls(controlled_slopes, confidences, illum, prev_r0=r0)
                
                # Blended control wavefront
                z_ctrl = 0.7 * predicted_z + 0.3 * z_glao
                
                # Command Generation (Saturation, Hysteresis, and Shear handling)
                current_commands, comp_commands = dm_controller.calculate_commands(z_ctrl, current_commands)
                
                # Suppress mechanical vibrations on tip-tilt (Problem 14)
                comp_commands[:2] = notch_filter.filter_signal(comp_commands[:2])
                
                # Temporal MCAO multiplexing (Problem 12)
                applied_commands = multiplexor.multiplex_command(comp_commands)
        elif not use_spgd:
            # --- CLASSICAL BASELINE ---
            if not is_dm_frame:
                # Fast tip-tilt loop (1 kHz)
                z_tt, _ = reconstructor.reconstruct_wls(centroids, confidences, illum)
                z_tt_only = np.zeros(reconstructor.num_zernikes)
                z_tt_only[:2] = z_tt[:2]
                u_tt = dm_controller.zernike_to_command @ z_tt_only
                current_commands[:2] = current_commands[:2] - 0.5 * u_tt[:2]
                current_commands = dm_controller.clip_interactuator_shear(current_commands)
                applied_commands[:2] = current_commands[:2]
            else:
                # High-order loop updates (500 Hz)
                z_wls, uarc_vars = reconstructor.reconstruct_wls(centroids, confidences, illum)
                u_target = dm_controller.zernike_to_command @ z_wls
                current_commands = current_commands - 0.4 * u_target
                current_commands = np.clip(current_commands, -dm_controller.stroke_limit, dm_controller.stroke_limit)
                
                # Apply shear protection check
                current_commands = dm_controller.clip_interactuator_shear(current_commands)
                
                current_commands[:2] = notch_filter.filter_signal(current_commands[:2])
                applied_commands = current_commands.copy()
            
        # Computational latency estimation with loop jitter (Problem 20)
        t_end = time.perf_counter()
        calc_latency = (t_end - t_start) + np.random.normal(config.NOMINAL_LATENCY, config.LATENCY_JITTER)
        calc_latency = np.clip(calc_latency, 1e-4, 5e-3)
        dt_latency = calc_latency
        latencies.append(calc_latency * 1000.0) # in ms
        
        # Track saturated actuators
        sat_count = np.sum(np.abs(current_commands) >= 0.85 * config.ACTUATOR_STROKE_LIMIT)
        saturation_counts.append(sat_count)
        
        # 5. Calibration Monitors (Layer 5 & 6)
        drifted_modes, needs_repoke = calib_monitor.monitor_drift(current_commands, z_wls, telescope_temp)
        if needs_repoke and use_vajra and not use_rl:
            dm_controller.zernike_to_command = calib_monitor.perform_targeted_repoke(
                drifted_modes, dm_controller.zernike_to_command, None
            )
            
        # Periodic actuator response calibrations (Problem 6)
        if frame % 20 == 0 and not load_telemetry:
            drift_detected, reconstructor.G_full = influence_calibrator.run_influence_check(
                sim, reconstructor.G_full
            )
            
        # 6. Science Diagnostics
        # Speckle NCPA tracking (Problem 10)
        if not load_telemetry:
            ncpa_bias = ncpa_tracker.estimate_ncpa(raw_frame[0:32, 0:32], telescope_temp)
        else:
            ncpa_bias = np.zeros(reconstructor.num_zernikes)
            
        # Record telemetry and update PSF (Problem 16)
        psf_reconstructor.record_telemetry(z_wls, current_commands, uarc_vars, ncpa_bias)
        
        # Calculate residual phase error
        dm_phase = sim.dm.phase_for(config.LAMBDA_SENSING)
        corrected_phase = (true_phase + dm_phase) * sim.pupil_mask
        rms_closed = np.std(corrected_phase[pupil_idx])
        rms_closed_loop.append(rms_closed)
        
        # Strehl ratio calculation
        strehl = np.exp(-rms_closed**2)
        strehl_history.append(strehl)

    # Compile final metrics dictionary
    metrics = {
        "open_rms": np.mean(rms_open_loop),
        "closed_rms": np.mean(rms_closed_loop),
        "strehl": np.mean(strehl_history),
        "latency": np.mean(latencies),
        "saturation": np.max(saturation_counts),
        "rms_history": rms_closed_loop,
        "open_history": rms_open_loop,
        "strehl_history": strehl_history,
        "regimes": (regime_steady_probs, regime_trans_probs, regime_broken_probs) if (use_vajra and not use_rl) else None,
        "psf": psf_reconstructor.reconstruct_psf()
    }
    
    return metrics

def quantize_model_dynamic(model_path, quantized_path):
    """Applies PyTorch post-training dynamic quantization to a saved model (Opt. 1)."""
    try:
        # Dynamically quantize the float32 model to int8 (for Linear/GRU layers)
        state_dict = torch.load(model_path, map_location='cpu')
        
        if "wfs_model" in model_path:
            from vajra.reconstructor import Layer0AttentionCNN
            model = Layer0AttentionCNN()
        else:
            from vajra.controller import SaturationNN
            model = SaturationNN()
            
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Apply PyTorch dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.GRU}, dtype=torch.qint8
        )
        
        torch.save(quantized_model.state_dict(), quantized_path)
        
        # Log size compression difference
        orig_size = os.path.getsize(model_path) / (1024 * 1024)
        quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"[Quantization] Quantized {os.path.basename(model_path)}:")
        print(f"  Original: {orig_size:.3f} MB")
        print(f"  Quantized: {quant_size:.3f} MB")
        print(f"  Compression: {orig_size / quant_size:.2f}x smaller")
        return True
    except Exception as e:
        print(f"[Quantization] Quantization failed for {os.path.basename(model_path)}: {e}")
        return False

def export_onnx_models(reconstructor, controller):
    """Exports Layer0AttentionCNN and SaturationNN models to ONNX and compares inference latencies."""
    print("\n[ONNX] Starting PyTorch to ONNX model export...")
    
    # 1. Export Layer 0 Attention CNN
    cnn_model = reconstructor.cnn_sensor
    dummy_input_cnn = torch.randn(1, 1, 256, 256, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    cnn_onnx_path = "data/wfs_model_point.onnx"
    
    try:
        torch.onnx.export(
            cnn_model, 
            dummy_input_cnn, 
            cnn_onnx_path, 
            export_params=True,
            opset_version=18, # Set opset version >= 18 for compatibility
            do_constant_folding=True,
            input_names=['input'], 
            output_names=['output']
        )
        print(f"[ONNX] Successfully exported WFS CNN to {cnn_onnx_path}")
    except Exception as e:
        print(f"[ONNX] Failed to export WFS CNN: {e}")
        
    # 2. Export Saturation NN
    sat_model = controller.saturation_nn
    dummy_target = torch.randn(1, config.ZERNIKE_MODES_MAX, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dummy_curr = torch.randn(1, config.DM_ACTUATORS_TOTAL, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    sat_onnx_path = "data/saturation_nn.onnx"
    
    try:
        torch.onnx.export(
            sat_model,
            (dummy_target, dummy_curr),
            sat_onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['target_zernikes', 'current_commands'],
            output_names=['corrections']
        )
        print(f"[ONNX] Successfully exported Saturation NN to {sat_onnx_path}")
    except Exception as e:
        print(f"[ONNX] Failed to export Saturation NN: {e}")

    # Latency comparison
    # PyTorch latency
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = sat_model(dummy_target, dummy_curr)
    t1 = time.perf_counter()
    pytorch_latency_us = ((t1 - t0) / 100) * 1e6
    
    # Try importing onnxruntime to verify ONNX speed
    try:
        import onnxruntime as ort
        ort_sess = ort.InferenceSession(sat_onnx_path, providers=['CPUExecutionProvider'])
        
        # Prepare inputs
        inputs = {
            'target_zernikes': dummy_target.cpu().numpy(),
            'current_commands': dummy_curr.cpu().numpy()
        }
        
        t0 = time.perf_counter()
        for _ in range(100):
            _ = ort_sess.run(None, inputs)
        t1 = time.perf_counter()
        onnx_latency_us = ((t1 - t0) / 100) * 1e6
        
        print(f"[ONNX Benchmark] Saturation NN inference latency:")
        print(f"  PyTorch: {pytorch_latency_us:.2f} us")
        print(f"  ONNX Runtime (CPU): {onnx_latency_us:.2f} us")
        print(f"  Speedup: {pytorch_latency_us / onnx_latency_us:.2f}x")
    except ImportError:
        print("[ONNX Benchmark] onnxruntime not installed. Skipping latency speedup comparison.")

def main():
    parser = argparse.ArgumentParser(description="VAJRA Closed-Loop Simulation Suite")
    parser.add_argument("--use-rl", action="store_true", help="Enable RL Meta-Controller gain tuning")
    parser.add_argument("--use-spgd", action="store_true", help="Run in Wavefront-Sensorless (SPGD) mode")
    parser.add_argument("--load-telemetry", action="store_true", help="Replay external telemetry FITS streams")
    parser.add_argument("--export-onnx", action="store_true", help="Export deep models to ONNX and profile speeds")
    parser.add_argument("--quantize", action="store_true", help="Applies dynamic dynamic quantization to .pth files")
    parser.add_argument("--multi-rate", action="store_true", help="Enable multi-rate loop (TTM 1kHz, DM 500Hz)")
    parser.add_argument("--ra", type=float, default=0.0, help="Gaia coordinate RA (degrees)")
    parser.add_argument("--dec", type=float, default=0.0, help="Gaia coordinate Dec (degrees)")
    parser.add_argument("--frames", type=int, default=100, help="Number of benchmark frames")
    parser.add_argument("--mode", type=str, default="solar", choices=["solar", "point"], help="Simulator mode")
    
    args = parser.parse_args()
    
    print("=============================================================")
    print("      VAJRA - Versatile Adaptive-optics Joint Real-time      ")
    print("               Actuation Benchmarking Run                    ")
    print("=============================================================")
    
    # 1. Run dynamic dynamic quantization if flag set
    if args.quantize:
        solar_pth = "data/wfs_model_solar.pth"
        point_pth = "data/wfs_model_point.pth"
        if os.path.exists(solar_pth):
            quantize_model_dynamic(solar_pth, "data/wfs_model_solar_quant.pth")
        if os.path.exists(point_pth):
            quantize_model_dynamic(point_pth, "data/wfs_model_point_quant.pth")
        if not os.path.exists(solar_pth) and not os.path.exists(point_pth):
            print("[Quantization] No model files (.pth) found in data/ for quantization.")
        return
        
    if args.export_onnx:
        sim = AOPipelineSimulator(mode=args.mode)
        reconstructor = WavefrontReconstructor(
            sim.pupil_grid, sim.pupil_mask, sim.subaps_pos, sim.subap_masks, target_mode=args.mode
        )
        dm_controller = ActuatorController(
            reconstructor.G_full, reconstructor.pupil_grid, reconstructor.zernike_basis
        )
        export_onnx_models(reconstructor, dm_controller)
        return

    conditions = ["steady", "transition", "strong"]
    results = {}
    
    for cond in conditions:
        print(f"\n[Benchmark] Evaluating condition: {cond.upper()} ...")
        
        print("  Running Classical SH-WFS Baseline...")
        metrics_classical = run_loop(sim_mode=args.mode, condition=cond, use_vajra=False, n_frames=args.frames, multi_rate=args.multi_rate)
        
        print("  Running VAJRA Wavefront Pipeline...")
        metrics_vajra = run_loop(
            sim_mode=args.mode, 
            condition=cond, 
            use_vajra=True, 
            n_frames=args.frames,
            use_rl=args.use_rl,
            use_spgd=args.use_spgd,
            load_telemetry=args.load_telemetry,
            ra=args.ra,
            dec=args.dec,
            multi_rate=args.multi_rate
        )
        
        results[cond] = {"classical": metrics_classical, "vajra": metrics_vajra}
        
        print(f"  Results ({cond}):")
        print(f"    Open Loop RMS       : {metrics_classical['open_rms']:.3f} rad")
        print(f"    Classical Closed RMS: {metrics_classical['closed_rms']:.3f} rad (Strehl: {metrics_classical['strehl']:.3f})")
        print(f"    VAJRA Closed RMS    : {metrics_vajra['closed_rms']:.3f} rad (Strehl: {metrics_vajra['strehl']:.3f})")
        print(f"    Max Actuator Sat    : Classical={metrics_classical['saturation']} | VAJRA={metrics_vajra['saturation']}")
        print(f"    Average Latency     : Classical={metrics_classical['latency']:.2f}ms | VAJRA={metrics_vajra['latency']:.2f}ms")

    # Generate diagnostic telemetry plots comparing VAJRA vs Classical
    print("\n[Plots] Generating final comparison figures...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Wavefront Residuals comparison in Strong Turbulence
    strong_res = results["strong"]
    axs[0, 0].plot(strong_res["classical"]["open_history"], label="Open Loop (Atmosphere)", color="tomato", linestyle="--")
    axs[0, 0].plot(strong_res["classical"]["rms_history"], label="Classical Baseline", color="orange", linewidth=1.5)
    axs[0, 0].plot(strong_res["vajra"]["rms_history"], label="VAJRA AO", color="forestgreen", linewidth=2.5)
    axs[0, 0].set_title("Wavefront Error (RMS rad) - Strong Turbulence")
    axs[0, 0].set_xlabel("Frame Number")
    axs[0, 0].set_ylabel("RMS (radians)")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Subplot 2: Strehl Ratio Comparison in Wind-Shear Transition
    trans_res = results["transition"]
    axs[0, 1].plot(trans_res["classical"]["strehl_history"], label="Classical Baseline", color="orange", linewidth=1.5)
    axs[0, 1].plot(trans_res["vajra"]["strehl_history"], label="VAJRA AO", color="dodgerblue", linewidth=2.5)
    axs[0, 1].set_title("Estimated Strehl Ratio - Wind Shear Transition")
    axs[0, 1].set_xlabel("Frame Number")
    axs[0, 1].set_ylabel("Strehl Ratio")
    axs[0, 1].set_ylim(0, 1.05)
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # Subplot 3: VAJRA Atmospheric Regime probabilities under Transition
    probs = trans_res["vajra"]["regimes"]
    if probs is not None:
        axs[1, 0].plot(probs[0], label="Steady", color="mediumseagreen")
        axs[1, 0].plot(probs[1], label="Transitioning", color="orange")
        axs[1, 0].plot(probs[2], label="Broken", color="crimson")
        axs[1, 0].set_title("VAJRA Autocorrelation Regime Classifier")
        axs[1, 0].set_ylabel("Probability")
    else:
        axs[1, 0].text(0.5, 0.5, "Regime Probability Plot\n(Only active in standard VAJRA mode)", 
                       ha='center', va='center', fontsize=12)
    axs[1, 0].set_xlabel("Frame Number")
    axs[1, 0].grid(True)
    
    # Subplot 4: Reconstructed PSF (VAJRA Strong seeing)
    psf_data = strong_res["vajra"]["psf"]
    im = axs[1, 1].imshow(psf_data, cmap='inferno')
    axs[1, 1].set_title("Reconstructed VAJRA Science PSF (H-Alpha)")
    fig.colorbar(im, ax=axs[1, 1])
    
    plt.tight_layout()
    plot_path = "data/vajra_telemetry.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[Plots] Telemetry comparison plot saved successfully in: {os.path.abspath(plot_path)}")
    
    # Print final validation metrics
    print("\n=============================================================")
    print("VAJRA Benchmarking Run Completed Successfully!")
    print("=============================================================")

if __name__ == "__main__":
    main()
