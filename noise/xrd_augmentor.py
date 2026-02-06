"""
XRD Spectrum Augmentor - Physics-Informed Data Augmentation
============================================================
A unified, easy-to-use interface for XRD data augmentation.

Based on: Ceder Group XRD-AutoAnalyzer
Reference: Szymanski et al., Chemistry of Materials (2021)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Callable
import random


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class AugmentConfig:
    """Configuration for XRD augmentation pipeline."""

    # Noise parameters
    noise_std: float = 0.25
    use_poisson: bool = False

    # Peak shift parameters
    max_shift: float = 0.5  # degrees, uniform shift
    max_strain: float = 0.04  # fractional, non-uniform

    # Peak broadening (Scherrer)
    min_domain_size: float = 5.0  # nm
    max_domain_size: float = 100.0  # nm

    # Texture/preferred orientation
    max_texture: float = 0.6  # intensity scaling factor

    # Background
    add_background: bool = False
    background_scale: float = 5.0
    background_order: int = 3

    # Spectrum parameters
    min_angle: float = 10.0
    max_angle: float = 80.0
    num_points: int = 4501
    wavelength: float = 1.5406  # Cu Ka


# =============================================================================
# Core Physics Functions
# =============================================================================

def scherrer_fwhm(two_theta: float, domain_size: float, wavelength: float = 1.5406) -> float:
    """
    Calculate FWHM using Scherrer equation.
    FWHM = K * λ / (D * cos(θ))
    """
    K = 0.9
    wavelength_nm = wavelength * 0.1
    theta = np.radians(two_theta / 2.0)
    beta = (K * wavelength_nm) / (np.cos(theta) * domain_size)
    sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * np.degrees(beta)
    return sigma ** 2


def peaks_to_spectrum(angles, intensities, domain_size, config: AugmentConfig):
    """Convert discrete peaks to continuous spectrum with broadening."""
    steps = np.linspace(config.min_angle, config.max_angle, config.num_points)
    signals = np.zeros([len(angles), config.num_points])

    for i, ang in enumerate(angles):
        idx = np.argmin(np.abs(ang - steps))
        if 0 <= idx < config.num_points:
            signals[i, idx] = intensities[i]

    step_size = (config.max_angle - config.min_angle) / config.num_points
    for i in range(signals.shape[0]):
        row = signals[i, :]
        ang = steps[np.argmax(row)]
        std_dev = scherrer_fwhm(ang, domain_size, config.wavelength)
        signals[i, :] = gaussian_filter1d(row, np.sqrt(std_dev) / step_size, mode='constant')

    signal = np.sum(signals, axis=0)
    if signal.max() > 0:
        signal = 100 * signal / signal.max()
    return signal


# =============================================================================
# Individual Augmentation Transforms
# =============================================================================

def add_gaussian_noise(spectrum: np.ndarray, std: float = 0.25) -> np.ndarray:
    """Add Gaussian noise."""
    return spectrum + np.random.normal(0, std, len(spectrum))


def add_poisson_noise(spectrum: np.ndarray, scale: float = 100) -> np.ndarray:
    """Add Poisson noise (physically realistic for photon counting)."""
    scaled = np.maximum(spectrum * scale, 0)
    noisy = np.random.poisson(scaled.astype(int))
    return noisy / scale


def apply_uniform_shift(angles: np.ndarray, max_shift: float) -> np.ndarray:
    """Apply uniform peak shift (sample height error)."""
    shift = np.random.uniform(-max_shift, max_shift)
    return np.array(angles) + shift


def apply_texture(intensities, hkls, max_texture: float, is_hexagonal: bool = False):
    """Apply texture/preferred orientation effect."""
    n_idx = 4 if is_hexagonal else 3
    while True:
        pref_dir = [random.choice([0, 1]) for _ in range(n_idx)]
        if np.dot(pref_dir, pref_dir) > 0:
            break

    bound = 1.0 - max_texture
    scaled = []
    pref_dir = np.array(pref_dir)

    for hkl, intensity in zip(hkls, intensities):
        hkl = np.array(hkl)
        norm1 = np.sqrt(np.dot(hkl, hkl))
        norm2 = np.sqrt(np.dot(pref_dir, pref_dir))
        if norm1 * norm2 > 0:
            factor = abs(np.dot(hkl, pref_dir) / (norm1 * norm2))
        else:
            factor = 0.5
        factor = bound + ((1.0 - bound) * factor)
        scaled.append(intensity * factor)
    return scaled


def generate_background(num_points: int, scale: float, order: int = 3) -> np.ndarray:
    """Generate polynomial background."""
    x = np.linspace(-1, 1, num_points)
    coeffs = np.random.uniform(-1, 1, order + 1) * scale
    bg = np.polyval(coeffs, x)
    bg = bg - bg.min()
    return bg


# =============================================================================
# Main Augmentor Class
# =============================================================================

class XRDAugmentor:
    """
    Unified XRD spectrum augmentor with configurable pipeline.

    Example:
        aug = XRDAugmentor(config)
        spectrum = aug.augment(angles, intensities)
    """

    def __init__(self, config: AugmentConfig = None):
        self.config = config or AugmentConfig()
        self._two_theta = np.linspace(
            self.config.min_angle,
            self.config.max_angle,
            self.config.num_points
        )

    @property
    def two_theta(self):
        return self._two_theta

    def augment(self, angles, intensities, hkls=None, is_hexagonal=False,
                enable_shift=True, enable_broadening=True,
                enable_texture=True, enable_noise=True,
                enable_background=None):
        """
        Apply augmentation pipeline.

        Args:
            angles: peak positions (2theta)
            intensities: peak intensities
            hkls: Miller indices for texture
            is_hexagonal: hexagonal crystal system
            enable_*: toggle individual augmentations
        """
        cfg = self.config
        enable_background = enable_background if enable_background is not None else cfg.add_background

        # 1. Texture
        if enable_texture and hkls is not None:
            intensities = apply_texture(intensities, hkls, cfg.max_texture, is_hexagonal)

        # 2. Shift
        if enable_shift:
            angles = apply_uniform_shift(angles, cfg.max_shift)

        # 3. Broadening
        if enable_broadening:
            domain = np.random.uniform(cfg.min_domain_size, cfg.max_domain_size)
        else:
            domain = 25.0
        spectrum = peaks_to_spectrum(angles, intensities, domain, cfg)

        # 4. Background
        if enable_background:
            bg = generate_background(cfg.num_points, cfg.background_scale, cfg.background_order)
            spectrum = spectrum + bg
            spectrum = 100 * spectrum / spectrum.max()

        # 5. Noise
        if enable_noise:
            if cfg.use_poisson:
                spectrum = add_poisson_noise(spectrum)
            else:
                spectrum = add_gaussian_noise(spectrum, cfg.noise_std)

        return spectrum

    def augment_batch(self, angles, intensities, n_samples: int,
                      hkls=None, is_hexagonal=False, **kwargs):
        """Generate multiple augmented spectra."""
        return np.array([
            self.augment(angles, intensities, hkls, is_hexagonal, **kwargs)
            for _ in range(n_samples)
        ])


# =============================================================================
# Preset Configurations
# =============================================================================

def get_preset_config(preset: str) -> AugmentConfig:
    """Get preset augmentation configurations."""
    presets = {
        'mild': AugmentConfig(
            noise_std=0.15, max_shift=0.3, max_texture=0.3,
            min_domain_size=20, max_domain_size=100
        ),
        'moderate': AugmentConfig(
            noise_std=0.25, max_shift=0.5, max_texture=0.5,
            min_domain_size=10, max_domain_size=80
        ),
        'strong': AugmentConfig(
            noise_std=0.5, max_shift=1.0, max_texture=0.7,
            min_domain_size=5, max_domain_size=50, add_background=True
        ),
        'experimental': AugmentConfig(
            noise_std=0.4, max_shift=0.8, max_texture=0.6,
            min_domain_size=5, max_domain_size=60,
            add_background=True, use_poisson=True
        )
    }
    return presets.get(preset, AugmentConfig())


# =============================================================================
# Convenience Functions
# =============================================================================

def augment_xrd(angles, intensities, preset='moderate', n_samples=1, **kwargs):
    """
    One-line augmentation function.

    Args:
        angles: peak positions
        intensities: peak intensities
        preset: 'mild', 'moderate', 'strong', 'experimental'
        n_samples: number of augmented spectra
    """
    config = get_preset_config(preset)
    aug = XRDAugmentor(config)
    if n_samples == 1:
        return aug.augment(angles, intensities, **kwargs)
    return aug.augment_batch(angles, intensities, n_samples, **kwargs)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example peaks
    angles = [25.3, 37.8, 48.1, 54.2, 65.5]
    intensities = [100, 45, 30, 25, 15]

    # Method 1: One-line
    spec = augment_xrd(angles, intensities, preset='moderate')
    print(f"Single spectrum: {spec.shape}")

    # Method 2: Batch
    specs = augment_xrd(angles, intensities, preset='strong', n_samples=10)
    print(f"Batch spectra: {specs.shape}")

    # Method 3: Custom config
    config = AugmentConfig(noise_std=0.3, max_shift=0.8)
    aug = XRDAugmentor(config)
    spec = aug.augment(angles, intensities)
    print(f"Custom config: {spec.shape}")
