import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

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

def run_loop(sim_mode="solar", condition="steady", use_vajra=True, n_frames=100):
    """Runs the closed-loop simulation under specified parameters and condition."""
    # Reset random seeds for fair comparison
    np.random.seed(42)
    
    # 1. Initialize VAJRA/SURYA pipeline components
    sim = AOPipelineSimulator(mode=sim_mode)
    detector = DetectorProcessor()
    reconstructor = WavefrontReconstructor(
        sim.pupil_grid, sim.pupil_mask, sim.subaps_pos, sim.subap_masks
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
        
    # 3. Initialize metrics vectors
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
        
        # Generate raw WFS Hartmannogram image convolved with solar granulation
        # Ingests DM shape and registration drift
        raw_frame, illum = sim.generate_hartmannogram(
            dm_commands=applied_commands,
            registration_offset=registration_drift_offset
        )
        
        # Start timer for computational latency monitoring (Problem 20)
        t_start = time.perf_counter()
        
        # WFS Centroid processing
        centroids, fluxes, fwhms, confidences = detector.process_frame(raw_frame, illum)
        
        if use_vajra:
            # --- VAJRA PIPELINE ---
            # Wavefront Reconstruction
            z_wls, uarc_vars = reconstructor.reconstruct_wls(centroids, confidences, illum, prev_r0=r0)
            
            # Autocorrelation-based regime classification (Problem 8)
            regime_predictor.classify_regime(centroids.flatten())
            regime_steady_probs.append(regime_predictor.probabilities[0])
            regime_trans_probs.append(regime_predictor.probabilities[1])
            regime_broken_probs.append(regime_predictor.probabilities[2])
            
            # Dynamic Zernike prediction (Problem 8 & 20)
            predicted_z = regime_predictor.predict(z_wls, dt_latency)
            
            # Ground-layer / High-altitude separation (Problem 11 & 23)
            controlled_slopes = glao.get_controlled_slopes(centroids, illum)
            z_glao, _ = reconstructor.reconstruct_wls(controlled_slopes, confidences, illum, prev_r0=r0)
            
            # Blended control wavefront
            z_ctrl = 0.7 * predicted_z + 0.3 * z_glao
            
            # Command Generation (Saturation & Hysteresis handling)
            current_commands, comp_commands = dm_controller.calculate_commands(z_ctrl, current_commands)
            
            # Suppress mechanical vibrations on tip-tilt (Problem 14)
            comp_commands[:2] = notch_filter.filter_signal(comp_commands[:2])
            
            # Temporal MCAO multiplexing (Problem 12)
            applied_commands = multiplexor.multiplex_command(comp_commands)
        else:
            # --- CLASSICAL BASELINE ---
            # Wavefront Reconstruction (WLS without adaptive seeing checks)
            z_wls, uarc_vars = reconstructor.reconstruct_wls(centroids, confidences, illum)
            
            # Standard integrator control (no prediction, no GLAO)
            u_target = dm_controller.zernike_to_command @ z_wls
            current_commands = current_commands - 0.4 * u_target
            current_commands = np.clip(current_commands, -dm_controller.stroke_limit, dm_controller.stroke_limit)
            
            # Standard tip-tilt notch filtering
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
        if needs_repoke and use_vajra:
            dm_controller.zernike_to_command = calib_monitor.perform_targeted_repoke(
                drifted_modes, dm_controller.zernike_to_command, None
            )
            
        # Periodic actuator response calibrations (Problem 6)
        if frame % 20 == 0:
            drift_detected, reconstructor.G_full = influence_calibrator.run_influence_check(
                sim, reconstructor.G_full
            )
            
        # 6. Science Diagnostics
        # Speckle NCPA tracking (Problem 10)
        ncpa_bias = ncpa_tracker.estimate_ncpa(raw_frame[0:32, 0:32], telescope_temp)
        
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
        "regimes": (regime_steady_probs, regime_trans_probs, regime_broken_probs) if use_vajra else None,
        "psf": psf_reconstructor.reconstruct_psf()
    }
    
    return metrics

def main():
    print("=============================================================")
    print("      VAJRA - Versatile Adaptive-optics Joint Real-time      ")
    print("               Actuation Benchmarking Run                    ")
    print("=============================================================")
    
    conditions = ["steady", "transition", "strong"]
    results = {}
    
    for cond in conditions:
        print(f"\n[Benchmark] Evaluating condition: {cond.upper()} ...")
        
        print("  Running Classical SH-WFS Baseline...")
        metrics_classical = run_loop(sim_mode="solar", condition=cond, use_vajra=False)
        
        print("  Running VAJRA Wavefront Pipeline...")
        metrics_vajra = run_loop(sim_mode="solar", condition=cond, use_vajra=True)
        
        results[cond] = {"classical": metrics_classical, "vajra": metrics_vajra}
        
        print(f"  Results ({cond}):")
        print(f"    Open Loop RMS       : {metrics_classical['open_rms']:.3f} rad")
        print(f"    Classical Closed RMS: {metrics_classical['closed_rms']:.3f} rad (Strehl: {metrics_classical['strehl']:.3f})")
        print(f"    VAJRA Closed RMS    : {metrics_vajra['closed_rms']:.3f} rad (Strehl: {metrics_vajra['strehl']:.3f})")
        print(f"    Max Actuator Sat    : Classical={metrics_classical['saturation']} | VAJRA={metrics_vajra['saturation']}")
        print(f"    Average Latency     : Classical={metrics_classical['latency']:.2f}ms | VAJRA={metrics_vajra['latency']:.2f}ms")

    # 7. Generate diagnostic telemetry plots comparing VAJRA vs Classical
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
    axs[1, 0].set_xlabel("Frame Number")
    axs[1, 0].set_ylabel("Probability")
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
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
