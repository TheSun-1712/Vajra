# Presentation Slides Content: VAJRA Adaptive Optics System

This document contains the complete and detailed slide-by-slide content for the **VAJRA** project presentation, structured according to the specified 10-slide skeleton.

---

## Slide 1 — Title

*   **Project Name**: **VAJRA**
*   **Expanded Acronym/Tagline**: **Versatile Adaptive-optics Joint Real-time Actuation** — *Real-time Wavefront Reconstruction and Control for Solar and Stellar Telescopes*
*   **Team Name**: **SUNSHINE-ISRO**
*   **Problem Statement Reference**:
    *   **Problem Statement Number**: ISRO-02 (SIH-2024 / Space Applications Centre)
    *   **One-Line Topic**: Real-Time Wavefront Sensing, Reconstruction, and Deformable Mirror Control under Intense Atmospheric Turbulence and Mechanical Mount Vibrations.
*   **Team Leader Name**: **Aditya** (Lead Controls Engineer & System Architect)

---

## Slide 2 — Team Members

*   **Member 1 (Team Leader)**:
    *   **Name**: Aditya
    *   **Role/Specialization**: Lead Controls & Embedded Systems Engineer. Responsible for developing the adaptive digital notch filters, the multi-rate loop control architecture, and the GRU-based actuator hysteresis compensator.
    *   **College/Branch**: Indian Institute of Space Science and Technology (IIST) / B.Tech in Avionics (Control Systems)
*   **Member 2**:
    *   **Name**: Rohan Sharma
    *   **Role/Specialization**: Deep Learning & Computer Vision Specialist. Engineered the image-to-Zernike `Layer0AttentionCNN` model, implemented the self-supervised SimCLR contrastive pre-training pipeline, and designed the INT8 model quantization routines.
    *   **College/Branch**: Indian Institute of Space Science and Technology (IIST) / B.Tech in Avionics (AI & ML)
*   **Member 3**:
    *   **Name**: Sneha Patel
    *   **Role/Specialization**: Physical Optics & Atmospheric Simulation Specialist. Programmed the physical optics Wavefront Propagator, the multi-layer Kolmogorov atmospheric turbulence models, and the Shack-Hartmann spot-field generators.
    *   **College/Branch**: Indian Institute of Space Science and Technology (IIST) / B.Tech in Engineering Physics (Optical Engineering)
*   **Member 4**:
    *   **Name**: Vikram Singh
    *   **Role/Specialization**: Full-Stack UI & Telemetry Software Developer. Developed the React, Vite, and TanStack Start dashboard frontend, engineered the Flask-based real-time telemetry broker, and optimized the client-server data streaming logic.
    *   **College/Branch**: Indian Institute of Space Science and Technology (IIST) / B.Tech in Computer Science & Engineering

---

## Slide 3 — Opportunity / Proposed Solution

*   **Block 1: How this differs from existing solutions**
    *   Traditional Adaptive Optics (AO) loops rely on Center of Gravity (CoG) centroiding, which fails completely on the Sun because solar granulation acts as an extended, low-contrast, constantly evolving texture instead of a point source. Furthermore, classical loops are vulnerable to a 1.5 ms measurement-to-actuation latency and narrow-band mechanical mount vibrations. VAJRA solves these limitations by:
        1.  Replacing standard centroiding with an attention-based neural network (`Layer0AttentionCNN`) that maps raw subaperture images directly to Zernike coefficients.
        2.  Using a `RegimeAwarePredictor` paired with a GRU network to forecast atmospheric wavefront transitions ahead of the loop delay.
        3.  Deploying an active vibration tracking algorithm that dynamically updates digital notch filters to cancel sharp telescope resonance peaks.
*   **Block 2: How it solves the problem (Pipeline Name & Key Challenge-to-Solution Pairs)**
    *   **Pipeline Name**: *VAJRA Closed-Loop Wavefront Reconstruction & Actuation Pipeline*
    *   *Challenge-to-Solution Pairs*:
        *   **Challenge**: Solar granulation makes subaperture spot centroiding impossible.
            *   **Solution**: **Direct Image-to-Zernike Attention CNN** (`Layer0AttentionCNN`) trained end-to-end to regress wavefront Zernike coefficients directly from raw frames.
        *   **Challenge**: 1.5 ms loop latency causes the correction to lag behind the actual atmosphere.
            *   **Solution**: **Regime-Aware Predictor + Recurrent GRU Forecast** to classify atmospheric states and predict Zernike transitions ahead of time.
        *   **Challenge**: Mount vibrations and cooling pumps inject narrow-band resonance peaks (e.g., 48Hz and 85Hz).
            *   **Solution**: **Mechanical Notch Filter** that dynamically tracks dominant frequencies via real-time FFT peak monitoring and dampens them.
        *   **Challenge**: DM actuator stroke saturation and mechanical limits.
            *   **Solution**: **MLP-based Saturation Command Mapping** (`SaturationNN`) ensuring stroke constraints are resolved in microseconds.
        *   **Challenge**: Non-Common-Path Aberration (NCPA) downstream of the WFS beam splitter.
            *   **Solution**: **Focal Plane Speckle Nulling Controller** that applies phase-shifting DM probes to cancel static speckles.
*   **Block 3: USP (Unique Selling Proposition)**
    *   **A fully-integrated, end-to-end differentiable physical-optics closed-loop controller that achieves sub-millisecond, sub-aperture-level correction below the atmospheric coherence limit using deep sequence modeling and adaptive vibration suppression.**

---

## Slide 4 — Features List

1.  **Direct Image-to-Zernike Attention CNN (`Layer0AttentionCNN`)**
    *   *Benefit*: Bypasses standard centroiding, resolving centroid-gain errors and spot-truncation issues on extended solar targets.
2.  **Predictive Recurrent Zernike Forecaster (GRU)**
    *   *Benefit*: Compensates for 1.5 ms loop latency by anticipating wavefront changes based on wind-shear layers.
3.  **Adaptive Cascaded Digital Notch Filter**
    *   *Benefit*: Dynamically tracks and cancels narrow-band mechanical vibrations (cooling pumps, motors) at 48Hz and 85Hz.
4.  **Tomographic Wavefront Reconstruction**
    *   *Benefit*: Computes 3D volumetric turbulence profiles from multi-LGS constellations to correct for focus anisoplanatism.
5.  **Microsecond Actuator Saturation Solver (`SaturationNN`)**
    *   *Benefit*: Runs inline to map conjugate wavefronts to Deformable Mirror commands while satisfying physical stroke limits.
6.  **Science Path NCPA Speckle Nuller**
    *   *Benefit*: Continuously cleans static aberrations downstream of the wavefront sensor to maximize science image contrast.

*   **Supporting Image Slot**: `[Atmospheric vs. Residual Wavefront Phase Screen (from VAJRA Dashboard)]`

---

## Slide 5 — Process Flow Diagram

*   **Flowchart Stage Sequence**:
    1.  **Wavefront Distortion**: Plane-parallel wavefront passes through multi-layer Kolmogorov atmospheric turbulence.
    2.  **Subaperture Grid Capture**: Microlens Array (MLA) splits the wavefront and projects a spot field onto the WFS detector.
    3.  **Intensity to Wavefront Mapping**: `Layer0AttentionCNN` directly infers Zernike coefficients from raw intensity frames.
    4.  **Confidence-Weighted Least Squares (WLS)**: `DetectorProcessor` estimates spot FWHMs and photon noise to weight wavefront reconstructors.
    5.  **Temporal Latency Forecasting**: `RegimeAwarePredictor` feeds history to GRU networks to predict the upcoming phase state.
    6.  **Vibration Suppression**: `MechanicalNotchFilter` strips out sharp mechanical vibrations from the control stream.
    7.  **Actuator Mapping & Bounds Check**: `ActuatorController` and `SaturationNN` map phase offsets to physical commands.
    8.  **Physical DM Correction**: Deformable Mirror (DM) deforms to conjugate shape, producing a flat wavefront for the science camera.

*   **Image Placeholder**: `[End-to-End VAJRA Simulation Pipeline Block Diagram]`

---

## Slide 6 — Dashboard Mockup

*   **Image Slot**: `[Sky Weaver Dashboard UI Mockup Screenshot]`
*   **Panel Labels**:
    *   **Top Bar Header**: Displays system diagnostic telemetry, loop frequency (1000 Hz), site details (IAO Hanle), and the active regime classifier output.
    *   **Atmospheric Phase Map (Left)**: Represents the simulated turbulent Kolmogorov atmosphere in real-time.
    *   **Residual Phase Map (Center)**: Displays the corrected wavefront phase, illustrating the loop's effectiveness.
    *   **WFS Hartmannogram (Right)**: Shows the current spot grid imaged by the Shack-Hartmann microlens array.
    *   **DM Actuator Map (Far Right)**: Shows the 18x18 grid of physical actuator deformations.
    *   **Telemetry History Chart (Bottom)**: Tracks the time-series of the RMS wavefront error (in nm) and the Strehl ratio.

---

## Slide 7 — Architecture Diagram

*   **Architecture Layers**:
    *   **Layer 1: Optical Propagation & Physics Simulation**: Simulates the telescope aperture, atmosphere layers, MLA grid, and Deformable Mirror.
    *   **Layer 2: Real-time Inference Layer**: PyTorch-based networks (`Layer0AttentionCNN`, `RecurrentWavefrontPredictor`, `RegimeClassifier`, `SaturationNN`).
    *   **Layer 3: Control & Signal Processing Layer**: Houses the `ActuatorController`, `MechanicalNotchFilter`, and `GLAODecomposer`.
    *   **Layer 4: Telemetry Broker**: Flask API server running at `127.0.0.1:5000` exposing `/api/reset` and `/api/run_step`.
    *   **Layer 5: Visualization & UI Layer**: The React/TanStack Start dashboard running on a separate node port.

*   **Image Placeholder**: `[VAJRA Layered Software & Hardware Architecture Diagram]`

---

## Slide 8 — Technologies Used

*   **PyTorch**: Drives deep neural network training, inference, and end-to-end differentiable backpropagation.
*   **Flask**: Acts as the high-throughput REST API backend hosting the physical loop simulation.
*   **React & TanStack Start / Router**: Powers the modular, high-performance user interface.
*   **Vite / TypeScript**: Bundles the frontend code and ensures static type-safety.
*   **Recharts**: Renders smooth, real-time charts for monitoring RMS error and Strehl ratio.
*   **TailwindCSS**: Provides responsive layout designs and modern styling.
*   **Numpy & SciPy**: Handles high-performance matrix algebra, FFT calculations, and Zernike expansions.
*   **Gymnasium**: Wraps the AO simulation as an RL environment to train loop parameters using SAC.

---

## Slide 9 — Implementation Cost & Roadmap

*   **Project Roadmap**:
    *   **Phase 1: Physical Simulator & WFS Modeling (Months 1-2)**: Build the optical propagator, Shack-Hartmann generator, and Kolmogorov profiles.
    *   **Phase 2: Deep Learning Models (Months 3-4)**: Train `Layer0AttentionCNN` on solar granulation FITS datasets and configure Zernike forecasters.
    *   **Phase 3: Control & Calibration (Months 5-6)**: Integrate notch filters, GLAO decomposer, and influence function calibration.
    *   **Phase 4: Laboratory Hardware Integration (Months 7-8)**: Port Python controllers to real-time hardware-in-the-loop (RT-HIL) targets.
    *   **Phase 5: Telescope Commissioning (Months 9-12)**: Deploy at NLST IAO Hanle for on-sky validation.

*   **Budget Table**:

    | Line Item | Description | Cost (INR) | Cost (USD) |
    | :--- | :--- | :--- | :--- |
    | **Deformable Mirror (DM)** | Boston Micromachines 276-actuator DM with driver electronics | ₹45,00,000 | $55,000 |
    | **Wavefront Sensor Camera** | High-speed (1 kHz), low read-noise sCMOS detector | ₹12,50,000 | $15,000 |
    | **Real-time Controller Unit** | Xeon-based server + Dual NVIDIA RTX A6000 GPUs | ₹10,00,000 | $12,000 |
    | **Optomechanical Stage & MLA** | Microlens arrays, optical mounts, and alignment stages | ₹6,00,000 | $7,200 |
    | **Software Development & Compute** | Cloud GPU training instances for deep models | ₹4,00,000 | $4,800 |
    | **Site Integration & Commissioning** | Field integration expenses at IAO Hanle | ₹8,00,000 | $9,600 |
    | **Total Estimated Budget** | **Comprehensive Hardware & Software Integration** | **₹85,50,000** | **$103,600** |

---

## Slide 10 — Closing

*   **Fixed Image Placeholder**: `[NLST IAO Hanle Observatory Night View with Laser Guide Star]`
