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
from vajra.reconstructor import WavefrontReconstructor, TomographicReconstructor
from vajra.controller import (
    ActuatorController,
    MechanicalNotchFilter,
    RegimeAwarePredictor,
    GLAODecomposer,
    TemporalMultiplexor,
    LQGStructuralTracker
)
from vajra.calibration import LoopCalibrationMonitor, InfluenceFunctionCalibrator
from vajra.science import NCPATracker, PSFReconstructor, SpeckleNullingController

def get_science_image(phase, mask):
    """Problem 26: Physical optics propagation to compute science focal plane image (PSF)."""
    pupil_size = 256
    padded_size = 512
    complex_pupil = np.zeros((padded_size, padded_size), dtype=complex)
    
    phase_2d = phase.reshape((pupil_size, pupil_size))
    mask_2d = mask.reshape((pupil_size, pupil_size))
    
    start = (padded_size - pupil_size) // 2
    complex_pupil[start:start+pupil_size, start:start+pupil_size] = np.exp(1j * phase_2d) * mask_2d
    
    focal_field = np.fft.fft2(complex_pupil)
    focal_field = np.fft.fftshift(focal_field)
    psf = np.abs(focal_field) ** 2
    return psf / (psf.sum() + 1e-12)

def run_stellar_loop(n_frames=50, use_tomography=False, use_nulling=False, use_lqg=False):
    """Runs the closed-loop stellar simulation incorporating the 5 unexplored challenges."""
    np.random.seed(42)
    
    # 1. Initialize Simulator in Point-Source Mode
    sim = AOPipelineSimulator(mode="point")
    detector = DetectorProcessor()
    
    # Pre-calculate diffraction-limited peak
    psf_dl = get_science_image(np.zeros(sim.pupil_grid.size), sim.pupil_mask)
    dl_peak = np.max(psf_dl)
    
    # 2. Setup Tomographic Reconstructors (4 LGS constellation)
    num_lgs = 4
    reconstructors = []
    for d in range(num_lgs):
        reconstructors.append(
            WavefrontReconstructor(
                sim.pupil_grid, sim.pupil_mask, sim.subaps_pos, sim.subap_masks, target_mode="point"
            )
        )
    tomo_reconstructor = TomographicReconstructor(reconstructors)
    
    reconstructor_scao = WavefrontReconstructor(
        sim.pupil_grid, sim.pupil_mask, sim.subaps_pos, sim.subap_masks, target_mode="point"
    )
    
    # 3. Setup Controllers and Jitter Trackers
    dm_controller = ActuatorController(
        reconstructor_scao.G_full, reconstructor_scao.pupil_grid, reconstructor_scao.zernike_basis
    )
    notch_filter = MechanicalNotchFilter()
    lqg_tracker = LQGStructuralTracker(dt=0.001)
    
    # 4. Setup Science loops
    speckle_nuller = SpeckleNullingController(num_actuators=dm_controller.num_actuators)
    
    # Tracking logs
    rms_history = []
    strehl_history = []
    tracked_frequencies = []
    contrast_history = []
    
    current_commands = np.zeros(dm_controller.num_actuators)
    applied_commands = np.zeros(dm_controller.num_actuators)
    applied_commands_alt = np.zeros(dm_controller.num_actuators)
    nulling_offsets = np.zeros(dm_controller.num_actuators)
    
    vibration_frequency = 48.0
    
    # 5. Main Closed-Loop Run
    for frame in range(1, n_frames + 1):
        loop_time = frame * 0.001
        sim.evolve_atmosphere(loop_time)
        
        vibration_frequency += 0.08
        mount_jitter = np.sin(2.0 * np.pi * vibration_frequency * loop_time) * 0.2
        
        # WFS slope measurement
        if use_tomography:
            centroids_list = []
            confidences_list = []
            illumination_list = []
            
            for d in range(num_lgs):
                direction_offset = np.array([np.sin(d * np.pi/2), np.cos(d * np.pi/2)]) * 0.2
                # Add mechanical jitter and directional pointing offsets
                raw_frame, illum = sim.generate_hartmannogram(
                    dm_commands=applied_commands,
                    dm_commands_altitude=applied_commands_alt,
                    registration_offset=direction_offset + mount_jitter
                )
                centroids, _, _, confidences = detector.process_frame(raw_frame, illum)
                centroids_list.append(centroids)
                confidences_list.append(confidences)
                illumination_list.append(illum)
                
            z_ctrl = tomo_reconstructor.reconstruct_tomography(
                centroids_list, confidences_list, illumination_list
            )
        else:
            raw_frame, illum = sim.generate_hartmannogram(
                dm_commands=applied_commands,
                dm_commands_altitude=applied_commands_alt,
                registration_offset=(mount_jitter, mount_jitter)
            )
            centroids, _, _, confidences = detector.process_frame(raw_frame, illum)
            z_ctrl, _ = reconstructor_scao.reconstruct_wls(centroids, confidences, illum)

        # Calculate residual phase error
        atmosphere_phase = sim.get_phase_screen()
        # Evaluate dual-DM correction (applied commands + altitude commands)
        dm_phase = (sim.dm.phase_for(config.LAMBDA_SENSING) + sim.dm_altitude.phase_for(config.LAMBDA_SENSING))
        
        # Test loop correction sign alignment
        corrected_phase = (atmosphere_phase - dm_phase) * sim.pupil_mask
        rms_closed = np.std(corrected_phase[sim.pupil_mask > 0])
        rms_history.append(rms_closed)
        
        # Compute true physical Strehl ratio from focal plane PSF
        psf_actual = get_science_image(corrected_phase, sim.pupil_mask)
        strehl = np.max(psf_actual) / dl_peak
        strehl_history.append(strehl)

        # Active Speckle Nulling Probe Loop (runs every 10 frames)
        if use_nulling and frame % 10 == 0:
            intensity_measurements = []
            for probe_idx in range(4):
                probe_shape = speckle_nuller.generate_probe(probe_idx)
                # Compute science camera PSF under the probe phase shift by temporarily deforming the DM
                sim.dm.actuators = applied_commands + probe_shape
                probe_dm_phase = sim.dm.phase_for(config.LAMBDA_SENSING) + sim.dm_altitude.phase_for(config.LAMBDA_SENSING)
                probe_phase = (atmosphere_phase - probe_dm_phase) * sim.pupil_mask
                science_img = get_science_image(probe_phase, sim.pupil_mask)
                # Crop a 32x32 patch centered on the dark hole region
                science_img_cropped = science_img[240:272, 240:272]
                intensity_measurements.append(science_img_cropped)
                
            # Reset actuators back to loop state
            sim.dm.actuators = applied_commands
            nulling_offsets = speckle_nuller.compute_nulling_commands(intensity_measurements)

        # Adaptive LQG vibration tracking (Problem 30)
        measured_tip = z_ctrl[0]
        if use_lqg:
            tracked_f = lqg_tracker.update_and_track(measured_tip)
            tracked_frequencies.append(tracked_f)
            notch_filter.fs = 10.0 * tracked_f
        else:
            tracked_frequencies.append(48.0)
            
        # Command updates
        current_commands, comp_commands = dm_controller.calculate_commands(z_ctrl, current_commands)
        
        # Apply mechanical vibration suppression
        comp_commands[:2] = notch_filter.filter_signal(comp_commands[:2])
        
        # Merge nulling offsets into high-order commands
        comp_commands = comp_commands + nulling_offsets
        
        # Actuator Hysteresis pre-shaping (Problem 29 & 21)
        applied_commands = dm_controller.compensate_hysteresis(comp_commands)
        applied_commands = dm_controller.clip_interactuator_shear(applied_commands)
        
        # Dual-DM command update
        applied_commands_alt = 0.3 * comp_commands
        
        # Calculate dark hole region contrast (mean speckle intensity in off-axis region)
        contrast = np.mean(psf_actual[260:280, 260:280])
        contrast_history.append(contrast)
        
        if frame % 10 == 0:
            print(f"  Frame {frame:2d}/{n_frames} | RMS Error: {rms_closed:.3f} rad | Strehl: {strehl:.4f} | Tracked Res: {tracked_frequencies[-1]:.2f} Hz | Contrast: {contrast:.6f}")

    metrics = {
        "rms_history": rms_history,
        "strehl_history": strehl_history,
        "tracked_frequencies": tracked_frequencies,
        "contrast_history": contrast_history,
        "final_strehl": np.mean(strehl_history[-5:])
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="VAJRA Stellar AO Closed-Loop Benchmark Suite")
    parser.add_argument("--frames", type=int, default=50, help="Number of benchmark frames")
    parser.add_argument("--use-tomography", action="store_true", help="Enable LGS Volumetric Tomography")
    parser.add_argument("--use-nulling", action="store_true", help="Enable Active Speckle Nulling")
    parser.add_argument("--use-lqg", action="store_true", help="Enable LQG Structural Jitter Tracker")
    
    args = parser.parse_args()
    
    print("=============================================================")
    print("   VAJRA Stellar AO - Unexplored Challenges Benchmark        ")
    print("=============================================================")
    
    # 1. Classical Single Guide Star Baseline
    print("\n[Benchmark] Evaluating SCAO Baseline...")
    metrics_baseline = run_stellar_loop(
        n_frames=args.frames, use_tomography=False, use_nulling=False, use_lqg=False
    )
    
    # 2. Advanced Stellar AO Suite
    print(f"\n[Benchmark] Evaluating Advanced Stellar AO (Tomography={args.use_tomography}, Nulling={args.use_nulling}, LQG={args.use_lqg})...")
    metrics_advanced = run_stellar_loop(
        n_frames=args.frames, 
        use_tomography=args.use_tomography, 
        use_nulling=args.use_nulling, 
        use_lqg=args.use_lqg
    )
    
    print("\n======================= RESULTS ===============================")
    print(f"SCAO Strehl Ratio     : {metrics_baseline['final_strehl']:.4f}")
    print(f"Stellar AO Strehl Ratio: {metrics_advanced['final_strehl']:.4f}")
    print(f"SCAO Speckle Contrast : {np.mean(metrics_baseline['contrast_history'][-5:]):.6f}")
    print(f"Stellar AO Contrast   : {np.mean(metrics_advanced['contrast_history'][-5:]):.6f}")
    print("===============================================================")

    # Plot final telemetry
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    axs[0, 0].plot(metrics_baseline["rms_history"], label="SCAO Baseline", color="orange")
    axs[0, 0].plot(metrics_advanced["rms_history"], label="Stellar Tomography AO", color="forestgreen")
    axs[0, 0].set_title("Wavefront Error (RMS rad)")
    axs[0, 0].set_xlabel("Frame")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    axs[0, 1].plot(metrics_baseline["strehl_history"], label="SCAO Baseline", color="orange")
    axs[0, 1].plot(metrics_advanced["strehl_history"], label="Stellar Tomography AO", color="dodgerblue")
    axs[0, 1].set_title("Strehl Ratio")
    axs[0, 1].set_xlabel("Frame")
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # tracked frequency plot
    axs[1, 0].plot(metrics_advanced["tracked_frequencies"], label="Tracked Frequency (Kalman)", color="crimson")
    # true frequency drift reference
    true_drift = [48.0 + 0.08 * i for i in range(args.frames)]
    axs[1, 0].plot(true_drift, label="True Drifting Resonance", color="black", linestyle="--")
    axs[1, 0].set_title("LQG Vibration Tracker (Hz)")
    axs[1, 0].set_xlabel("Frame")
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # contrast plot
    axs[1, 1].plot(metrics_baseline["contrast_history"], label="SCAO Baseline", color="orange")
    axs[1, 1].plot(metrics_advanced["contrast_history"], label="Active Speckle Nulling", color="purple")
    axs[1, 1].set_title("Coronagraph Speckle Intensity")
    axs[1, 1].set_xlabel("Frame")
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/stellar_ao_telemetry.png", dpi=150)
    print("\n[Plots] Telemetry comparison plot saved successfully in: data/stellar_ao_telemetry.png")

if __name__ == "__main__":
    main()
