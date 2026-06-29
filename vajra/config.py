import numpy as np

# Physical constants and scaling laws
LAMBDA_SENSING = 500e-9      # Wavelength for wavefront sensing (500 nm)
LAMBDA_SCIENCE = 656.3e-9    # Wavelength for science (H-alpha, 656.3 nm)
D_APERTURE = 2.0             # Aperture diameter of telescope (2.0 meters)

# Atmosphere at Hanle (typical/worst-case values)
R0_500 = 0.08                # Fried parameter at 500nm (8 cm under moderate seeing)
OUTER_SCALE = 20.0           # Outer scale of turbulence L0 (20 meters)
WIND_SPEEDS = [15.0, 5.0]    # Wind speeds for a 2-layer model (m/s)
WIND_DIRECTIONS = [0.0, np.pi/4] # Wind directions (rad)
LAYER_HEIGHTS = [0.0, 5000.0] # Turbulence layer heights (meters)
LAYER_WEIGHTS = [0.7, 0.3]   # Cn2 profile weights (mostly ground layer at Hanle)

# Microlens Array (MLA) and Detector
MLA_GRID_SIZE = 16           # 16x16 subapertures
MLA_PITCH = D_APERTURE / MLA_GRID_SIZE # Subaperture width d = 12.5 cm
MLA_FOCAL_LENGTH = 0.025     # MLA focal length (25 mm)
SUBAP_PIXELS = 16            # Pixels per subaperture (16x16)
PIXEL_SCALE = 0.15           # Arcsec per pixel
READ_NOISE = 2.0             # Electrons read noise (e-)
PHOTON_FLUX_NOMINAL = 5000   # Photons per subaperture per frame (nominal high SNR)
PHOTON_FLUX_LOW = 150        # Faint guide star/limb threshold (low SNR)
SKY_BACKGROUND_PEDESTAL = 10.0 # Mean sky background photoelectrons/pixel

# Deformable Mirror (DM) - Aligned with Boston Micromachines Standard 276-3.5
DM_ACTUATORS_SIDE = 18       # 18x18 grid (324 total, ~276 inside circular pupil)
DM_ACTUATORS_TOTAL = DM_ACTUATORS_SIDE ** 2
ACTUATOR_STROKE_LIMIT = 3.5e-6 # Max physical stroke from specs (3.5 microns)
INTERACTUATOR_COUPLING = 0.13  # Interactuator coupling factor (13%)
INFLUENCE_FWHM = 0.15        # FWHM of actuator influence function (meters)
DM_HYSTERESIS_ALPHA = 0.10   # 10% Hysteresis coefficient
DM_CREEP_BETA = 0.02         # Hysteresis creep rate

# Controller & Loop Settings
LOOP_FREQUENCY = 1000.0      # Loop speed (1000 Hz / 1 ms)
NOMINAL_LATENCY = 0.0015     # Average latency (1.5 ms)
LATENCY_JITTER = 0.0003      # Latency Jitter (300 us RMS)
ZERNIKE_MODES_MAX = 66       # Max Zernike modes to track (Noll indices 2 to 67)

# NCPA & Calibrations
THERMAL_DRIFT_RATE = 1e-9    # Thermal expansion drift scale per degree C (meters)
NCPA_REF_DRIFT_DEGREE = 0.5  # Typical temperature rate of change (deg/hour)
MLA_REGISTRATION_DRIFT = 0.02 # Alignment drift rate (pixels per minute)

# SST/DKIST Granulation Dataset path
SST_DKIST_DATA_DIR = 'data/pid_2_13/YNCQFH/'
