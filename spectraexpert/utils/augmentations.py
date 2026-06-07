"""
Spectral Augmentations for Positive Pair Construction in Contrastive Learning.

All augmentations are designed to be **chemically identity-preserving**:
they simulate realistic experimental variations (temperature, pressure,
humidity, detector noise) without altering the underlying molecular identity.

Reference: MatExpert (ICLR 2025) — SimCLR contrastive learning framework,
adapted from material property↔structure alignment to spectrum↔spectrum alignment.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# 1. Wavenumber shift (temperature / pressure fluctuations)
# ---------------------------------------------------------------------------

def wavenumber_shift(spectrum: np.ndarray, max_shift: float = 2.0,
                     dx: float = 1.0) -> np.ndarray:
    """
    Apply a random global shift along the wavenumber axis.

    Parameters
    ----------
    spectrum : (N,) array — intensity values on a uniform grid.
    max_shift : max absolute shift in cm⁻¹ (default 2.0).
    dx : grid spacing in cm⁻¹ (default 1.0).

    Returns
    -------
    Shifted spectrum (same shape) via sub-pixel interpolation.
    """
    shift_pixels = np.random.uniform(-max_shift, max_shift) / dx
    x = np.arange(len(spectrum), dtype=np.float32)
    interpolator = interp1d(x, spectrum, kind="cubic", bounds_error=False,
                            fill_value=0.0)
    shifted = interpolator(x - shift_pixels)
    return shifted.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Peak broadening (temperature / disorder)
# ---------------------------------------------------------------------------

def peak_broadening(spectrum: np.ndarray, wavenumber_axis: np.ndarray,
                    sigma_range=(0.5, 3.0)) -> np.ndarray:
    """
    Convolve with a Gaussian kernel whose width varies along the axis.

    Parameters
    ----------
    spectrum : (N,) array.
    wavenumber_axis : (N,) array of wavenumber values (cm⁻¹).
    sigma_range : (min, max) of kernel sigma in cm⁻¹.

    Returns
    -------
    Broadened spectrum.
    """
    sigma = np.random.uniform(*sigma_range)
    # Frequency-dependent broadening: higher wavenumber → wider kernel
    normalized_wavenumber = wavenumber_axis / wavenumber_axis.max()
    kernel_width = sigma * (1.0 + 0.5 * normalized_wavenumber)
    # Use Gaussian filter with position-dependent sigma (approximated as
    # global average — true position-dependent convolution is expensive)
    mean_sigma = float(np.mean(kernel_width) / (wavenumber_axis[1] - wavenumber_axis[0]))
    broadened = gaussian_filter1d(spectrum, sigma=mean_sigma, mode="constant")
    return broadened.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Baseline drift (thermal radiation / fluorescence)
# ---------------------------------------------------------------------------

def baseline_addition(spectrum: np.ndarray, wavenumber_axis: np.ndarray,
                      max_amplitude: float = 0.1) -> np.ndarray:
    """
    Add a low-order polynomial baseline to simulate fluorescence / thermal
    background.

    Parameters
    ----------
    spectrum : (N,) array.
    wavenumber_axis : (N,) array.
    max_amplitude : max baseline amplitude relative to spectrum's max.

    Returns
    -------
    Spectrum with added baseline.
    """
    x_norm = wavenumber_axis / wavenumber_axis.max()
    order = np.random.choice([1, 2, 3])
    coeffs = np.random.uniform(-1.0, 1.0, size=order + 1)
    baseline = np.polyval(coeffs, x_norm)
    # Scale baseline amplitude relative to spectrum range
    spec_range = spectrum.max() - spectrum.min()
    if spec_range > 0:
        baseline = baseline / baseline.std() * spec_range * max_amplitude
    return (spectrum + baseline).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Intensity fluctuations (laser power / detector noise)
# ---------------------------------------------------------------------------

def intensity_fluctuation(spectrum: np.ndarray,
                          global_std: float = 0.05,
                          local_sigma_pixels: float = 20.0,
                          local_std: float = 0.05) -> np.ndarray:
    """
    Simulate laser power instability and spatially-correlated detector noise.

    Parameters
    ----------
    spectrum : (N,) array.
    global_std : std of global multiplicative factor.
    local_sigma_pixels : correlation length of local fluctuation field.
    local_std : std of local multiplicative factor.

    Returns
    -------
    Spectrum with intensity fluctuations.
    """
    # Global multiplicative fluctuation (laser power)
    global_factor = np.random.normal(1.0, global_std)

    # Locally-correlated multiplicative fluctuation (detector inhomogeneity)
    noise = np.random.normal(0, local_std, size=len(spectrum))
    local_factor = gaussian_filter1d(noise, sigma=local_sigma_pixels, mode="reflect")
    local_factor = 1.0 + local_factor / np.maximum(local_factor.std(), 1e-8) * local_std

    return (spectrum * global_factor * local_factor).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Water peak / contaminant peak addition (humidity)
# ---------------------------------------------------------------------------

def _gaussian_peak(x: np.ndarray, center: float, width: float,
                   amplitude: float) -> np.ndarray:
    """Generate a single Gaussian peak."""
    return amplitude * np.exp(-((x - center) / width) ** 2)


# Pre-computed water template centres (common Raman water bands, cm⁻¹)
_WATER_BANDS = [
    (3400, 250),   # O-H stretching
    (1640, 60),    # H-O-H bending
    (700, 100),    # librational
]


def add_contaminant_peaks(spectrum: np.ndarray, wavenumber_axis: np.ndarray,
                          max_relative_intensity: float = 0.2) -> np.ndarray:
    """
    Add random contaminant peaks to simulate humidity or impurities.

    Parameters
    ----------
    spectrum : (N,) array.
    wavenumber_axis : (N,) array.
    max_relative_intensity : max peak height relative to spectrum's max.

    Returns
    -------
    Spectrum with added peaks.
    """
    spec_max = spectrum.max()
    if spec_max <= 0:
        return spectrum.astype(np.float32)
    amp_scale = spec_max * max_relative_intensity

    # Randomly add water-like peaks
    for center, width in _WATER_BANDS:
        if np.random.random() > 0.5:
            amplitude = np.random.uniform(0.02, 0.15) * amp_scale
            spectrum = spectrum + _gaussian_peak(wavenumber_axis, center, width, amplitude)

    # Optionally add a random narrow spike (simulating cosmic ray / impurity)
    if np.random.random() > 0.7:
        rand_center = np.random.uniform(wavenumber_axis[0], wavenumber_axis[-1])
        rand_width = np.random.uniform(2.0, 10.0)
        rand_amp = np.random.uniform(0.05, 0.2) * amp_scale
        spectrum = spectrum + _gaussian_peak(wavenumber_axis, rand_center, rand_width, rand_amp)

    return spectrum.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Poisson-like shot noise
# ---------------------------------------------------------------------------

def add_shot_noise(spectrum: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """
    Add Poisson-like signal-dependent noise.

    Parameters
    ----------
    spectrum : (N,) array.
    noise_level : relative noise std (fraction of sqrt(signal)).

    Returns
    -------
    Noisy spectrum.
    """
    # Ensure non-negative
    safe = np.maximum(spectrum, 0)
    shot_noise = np.random.normal(0, 1, size=len(spectrum)) * np.sqrt(safe) * noise_level
    return (spectrum + shot_noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Composition: create a positive pair from one spectrum
# ---------------------------------------------------------------------------

def create_positive_pair(spectrum: np.ndarray, wavenumber_axis: np.ndarray,
                         rng: np.random.Generator = None) -> tuple:
    """
    Create a positive pair (anchor, positive) from a single spectrum
    by applying a randomised sequence of identity-preserving augmentations.

    The anchor is the original spectrum (with minor normalisation).
    The positive is a heavily augmented version.

    Parameters
    ----------
    spectrum : (N,) array — original Raman spectrum intensities.
    wavenumber_axis : (N,) array — wavenumber grid (cm⁻¹).
    rng : optional numpy random Generator.

    Returns
    -------
    (anchor, positive) — both (N,) float32 arrays.
    """
    # Ensure float32
    spectrum = spectrum.astype(np.float32)
    wavenumber_axis = wavenumber_axis.astype(np.float32)

    # --- Anchor: light normalisation only ---
    anchor = spectrum.copy()

    # --- Positive: apply 3-5 randomised augmentations ---
    positive = spectrum.copy()

    # List of (aug_fn, probability) pairs
    augmentations = [
        (lambda s: wavenumber_shift(s, max_shift=2.0), 0.6),
        (lambda s: peak_broadening(s, wavenumber_axis, sigma_range=(0.5, 3.0)), 0.5),
        (lambda s: baseline_addition(s, wavenumber_axis, max_amplitude=0.1), 0.5),
        (lambda s: intensity_fluctuation(s), 0.6),
        (lambda s: add_contaminant_peaks(s, wavenumber_axis, max_relative_intensity=0.2), 0.4),
    ]

    for aug_fn, prob in augmentations:
        if np.random.random() < prob:
            positive = aug_fn(positive)

    # Always add a small amount of shot noise to the positive
    positive = add_shot_noise(positive, noise_level=0.005)

    return anchor, positive


# ---------------------------------------------------------------------------
# Convenience: augment a batch of spectra into positive pairs
# ---------------------------------------------------------------------------

def create_batch_positive_pairs(spectra: np.ndarray,
                                wavenumber_axis: np.ndarray) -> tuple:
    """
    Create positive pairs for an entire batch at once.

    Parameters
    ----------
    spectra : (B, N) array — batch of spectra.
    wavenumber_axis : (N,) array.

    Returns
    -------
    (anchors, positives) — both (B, N) float32 arrays.
    """
    anchors = []
    positives = []
    for i in range(len(spectra)):
        a, p = create_positive_pair(spectra[i], wavenumber_axis)
        anchors.append(a)
        positives.append(p)
    return np.stack(anchors), np.stack(positives)
