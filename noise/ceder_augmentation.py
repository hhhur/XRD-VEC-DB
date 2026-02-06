"""
Ceder Group XRD Data Augmentation Methods
=========================================
Source: https://github.com/njszym/XRD-AutoAnalyzer

This module implements physics-informed data augmentation for XRD patterns,
including:
1. Strain shifts (non-uniform peak shifts)
2. Uniform shifts (sample height error)
3. Peak broadening (domain size effects)
4. Intensity changes (texture/preferred orientation)
5. Impurity peaks
6. Mixed augmentation (all effects combined)

Reference:
Szymanski et al., "Probabilistic Deep Learning Approach to Automate
the Interpretation of Multi-phase Diffraction Spectra"
"""

import numpy as np
import random
import math
from scipy.ndimage import gaussian_filter1d

# Optional imports - will be checked at runtime
try:
    import pymatgen as mg
    from pymatgen.analysis.diffraction import xrd
    from pymatgen.core import Structure, Lattice
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

try:
    from pyxtal import pyxtal
    HAS_PYXTAL = True
except ImportError:
    HAS_PYXTAL = False


class XRDCalculatorWrapper:
    """Wrapper for pymatgen XRD calculator"""

    def __init__(self, wavelength='CuKa'):
        if not HAS_PYMATGEN:
            raise ImportError("pymatgen is required. Install with: pip install pymatgen")
        self.calculator = xrd.XRDCalculator(wavelength=wavelength)
        self.wavelength = self.calculator.wavelength

    def get_pattern(self, struc, two_theta_range):
        return self.calculator.get_pattern(struc, two_theta_range=two_theta_range)


def calc_std_dev(two_theta, tau, wavelength=1.5406):
    """
    Calculate standard deviation based on angle and domain size using Scherrer equation.

    Args:
        two_theta: angle in two theta space (degrees)
        tau: domain size in nm
        wavelength: X-ray wavelength in Angstrom (default: Cu Ka = 1.5406)

    Returns:
        variance for gaussian kernel
    """
    K = 0.9  # shape factor
    wavelength_nm = wavelength * 0.1  # angstrom to nm
    theta = np.radians(two_theta / 2.0)  # Bragg angle in radians
    beta = (K * wavelength_nm) / (np.cos(theta) * tau)  # FWHM in radians

    # Convert FWHM to std deviation of gaussian
    sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * np.degrees(beta)
    return sigma ** 2


def apply_peak_broadening(angles, intensities, domain_size, min_angle, max_angle,
                          num_points=4501, wavelength=1.5406):
    """
    Apply peak broadening based on domain size (Scherrer equation).

    Args:
        angles: peak positions in 2theta
        intensities: peak intensities
        domain_size: crystallite domain size in nm
        min_angle, max_angle: 2theta range
        num_points: number of points in output spectrum
        wavelength: X-ray wavelength in Angstrom

    Returns:
        Broadened spectrum as numpy array
    """
    steps = np.linspace(min_angle, max_angle, num_points)
    signals = np.zeros([len(angles), num_points])

    # Place delta peaks
    for i, ang in enumerate(angles):
        idx = np.argmin(np.abs(ang - steps))
        if 0 <= idx < num_points:
            signals[i, idx] = intensities[i]

    # Convolve each peak with angle-dependent Gaussian
    step_size = (max_angle - min_angle) / num_points
    for i in range(signals.shape[0]):
        row = signals[i, :]
        ang = steps[np.argmax(row)]
        std_dev = calc_std_dev(ang, domain_size, wavelength)
        signals[i, :] = gaussian_filter1d(row, np.sqrt(std_dev) / step_size, mode='constant')

    # Combine and normalize
    signal = np.sum(signals, axis=0)
    if max(signal) > 0:
        signal = 100 * signal / max(signal)

    return signal


def add_gaussian_noise(spectrum, noise_std=0.25):
    """
    Add Gaussian noise to spectrum.

    Args:
        spectrum: input spectrum array
        noise_std: standard deviation of noise (default 0.25)

    Returns:
        Noisy spectrum
    """
    noise = np.random.normal(0, noise_std, len(spectrum))
    return spectrum + noise


class UniformShiftAugmenter:
    """
    Apply uniform peak shifts to simulate sample height error.
    Shifts all peaks by the same amount in 2theta.
    """

    def __init__(self, max_shift=0.5, min_angle=10.0, max_angle=80.0,
                 num_points=4501, noise_std=0.25):
        self.max_shift = max_shift
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.num_points = num_points
        self.noise_std = noise_std

    def augment(self, angles, intensities, domain_size=25.0):
        """
        Apply uniform shift augmentation.

        Args:
            angles: peak positions in 2theta
            intensities: peak intensities
            domain_size: domain size for broadening (nm)

        Returns:
            Augmented spectrum
        """
        shift = np.random.uniform(-self.max_shift, self.max_shift)
        shifted_angles = np.array(angles) + shift

        spectrum = apply_peak_broadening(
            shifted_angles, intensities, domain_size,
            self.min_angle, self.max_angle, self.num_points
        )
        return add_gaussian_noise(spectrum, self.noise_std)


class PeakBroadeningAugmenter:
    """
    Apply peak broadening to simulate domain size effects.
    Uses Scherrer equation for angle-dependent broadening.
    """

    def __init__(self, min_domain_size=1.0, max_domain_size=100.0,
                 min_angle=10.0, max_angle=80.0, num_points=4501, noise_std=0.25):
        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.num_points = num_points
        self.noise_std = noise_std

    def augment(self, angles, intensities):
        """Apply random domain size broadening."""
        domain_size = np.random.uniform(self.min_domain_size, self.max_domain_size)
        spectrum = apply_peak_broadening(
            angles, intensities, domain_size,
            self.min_angle, self.max_angle, self.num_points
        )
        return add_gaussian_noise(spectrum, self.noise_std)


class TextureAugmenter:
    """
    Apply texture (preferred orientation) effects.
    Scales peak intensities based on crystallographic direction.
    """

    def __init__(self, max_texture=0.6, min_angle=10.0, max_angle=80.0,
                 num_points=4501, noise_std=0.25):
        self.max_texture = max_texture
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.num_points = num_points
        self.noise_std = noise_std

    def _map_interval(self, v):
        """Map value from [0,1] to [1-max_texture, 1]"""
        bound = 1.0 - self.max_texture
        return bound + ((1.0 - bound) * v)

    def apply_texture(self, intensities, hkls, is_hexagonal=False):
        """
        Apply texture scaling to intensities.

        Args:
            intensities: original peak intensities
            hkls: list of (h,k,l) Miller indices
            is_hexagonal: True if hexagonal (4 indices)
        """
        n_idx = 4 if is_hexagonal else 3
        while True:
            pref_dir = [random.choice([0, 1]) for _ in range(n_idx)]
            if np.dot(pref_dir, pref_dir) > 0:
                break

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
            factor = self._map_interval(factor)
            scaled.append(intensity * factor)
        return scaled

    def augment(self, angles, intensities, hkls, is_hexagonal=False, domain_size=25.0):
        """Apply texture augmentation."""
        textured = self.apply_texture(intensities, hkls, is_hexagonal)
        spectrum = apply_peak_broadening(
            angles, textured, domain_size,
            self.min_angle, self.max_angle, self.num_points
        )
        return add_gaussian_noise(spectrum, self.noise_std)


class MixedAugmenter:
    """
    Apply all augmentation effects simultaneously:
    - Uniform shift (sample height error)
    - Peak broadening (domain size)
    - Texture (preferred orientation)
    - Gaussian noise
    """

    def __init__(self, max_shift=0.5, min_domain_size=5.0, max_domain_size=100.0,
                 max_texture=0.6, min_angle=10.0, max_angle=80.0,
                 num_points=4501, noise_std=0.25):
        self.max_shift = max_shift
        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size
        self.max_texture = max_texture
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.num_points = num_points
        self.noise_std = noise_std
        self._texture = TextureAugmenter(max_texture, min_angle, max_angle,
                                         num_points, noise_std=0)

    def augment(self, angles, intensities, hkls=None, is_hexagonal=False):
        """Apply all augmentations."""
        if hkls is not None:
            intensities = self._texture.apply_texture(intensities, hkls, is_hexagonal)

        shift = np.random.uniform(-self.max_shift, self.max_shift)
        shifted_angles = np.array(angles) + shift

        domain_size = np.random.uniform(self.min_domain_size, self.max_domain_size)
        spectrum = apply_peak_broadening(
            shifted_angles, intensities, domain_size,
            self.min_angle, self.max_angle, self.num_points
        )
        return add_gaussian_noise(spectrum, self.noise_std)


# =============================================================================
# Convenience function for simple spectrum augmentation
# =============================================================================

def augment_spectrum_simple(spectrum, noise_std=0.25, shift_points=5,
                            broaden_sigma=None):
    """
    Simple augmentation for pre-computed spectrum array.

    Args:
        spectrum: 1D intensity array
        noise_std: Gaussian noise std
        shift_points: max shift in array indices
        broaden_sigma: optional Gaussian blur sigma
    """
    result = np.array(spectrum).copy()

    if shift_points > 0:
        shift = int(np.random.uniform(-shift_points, shift_points))
        result = np.roll(result, shift)

    if broaden_sigma and broaden_sigma > 0:
        result = gaussian_filter1d(result, broaden_sigma)

    if noise_std > 0:
        result = add_gaussian_noise(result, noise_std)

    return result


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Example: augment a simple synthetic pattern
    two_theta = np.linspace(10, 80, 4501)

    # Synthetic peaks at specific angles
    peak_angles = [25.3, 37.8, 48.1, 54.2, 65.5]
    peak_intensities = [100, 45, 30, 25, 15]

    # Create augmenters
    shift_aug = UniformShiftAugmenter(max_shift=0.5)
    broad_aug = PeakBroadeningAugmenter(min_domain_size=5, max_domain_size=50)
    mixed_aug = MixedAugmenter()

    # Generate augmented spectra
    spec1 = shift_aug.augment(peak_angles, peak_intensities)
    spec2 = broad_aug.augment(peak_angles, peak_intensities)
    spec3 = mixed_aug.augment(peak_angles, peak_intensities)

    print(f"Generated spectra shape: {spec1.shape}")
    print("Augmentation complete!")
