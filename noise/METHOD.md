# Physics-Informed Data Augmentation for XRD Spectrum Analysis

## Method

### 3.1 Overview

We propose a physics-informed data augmentation pipeline for X-ray Diffraction (XRD) spectrum analysis in deep geometric learning frameworks. Unlike generic image augmentation techniques (e.g., random cropping, flipping), our approach incorporates domain-specific physical transformations that simulate realistic experimental variations encountered in powder diffraction measurements.

The augmentation pipeline consists of five physically-motivated transformations: (1) peak position shifts, (2) peak broadening, (3) preferred orientation effects, (4) background modeling, and (5) noise injection. Each transformation is parameterized by physically meaningful quantities derived from crystallographic theory.

### 3.2 Peak Position Shifts

**Physical Basis.** In experimental XRD measurements, peak positions can deviate from theoretical values due to two primary mechanisms:

1. **Uniform Shift (Instrumental)**: Sample height displacement causes all peaks to shift uniformly along the 2θ axis. We model this as:
   ```
   θ'_i = θ_i + Δθ,  where Δθ ~ U(-δ_max, δ_max)
   ```
   Typical values: δ_max = 0.5° for standard laboratory diffractometers.

2. **Non-uniform Shift (Lattice Strain)**: Internal stress causes anisotropic lattice distortion, resulting in peak-dependent shifts governed by Bragg's law:
   ```
   d'_hkl = d_hkl(1 + ε_hkl),  where ε ~ U(-ε_max, ε_max)
   ```
   We apply symmetry-preserving strain tensors based on crystal system (cubic, hexagonal, etc.) with ε_max ≤ 4%.

**Relevance to Deep Learning.** Peak shift augmentation teaches the model translation invariance along the 2θ axis, preventing overfitting to exact peak positions in training data.

### 3.3 Peak Broadening (Scherrer Effect)

**Physical Basis.** Finite crystallite size causes diffraction peak broadening according to the Scherrer equation:

```
β = Kλ / (D·cos θ)
```

where β is the full-width at half-maximum (FWHM), K ≈ 0.9 is the shape factor, λ is the X-ray wavelength, D is the crystallite domain size, and θ is the Bragg angle.

**Implementation.** We convolve each Bragg peak with an angle-dependent Gaussian kernel:

```
σ(θ) = β(θ) / (2√(2ln2))
```

Domain sizes are sampled uniformly: D ~ U(D_min, D_max), typically 5-100 nm.

**Relevance to Deep Learning.** This augmentation exposes the model to varying peak widths, enabling robust feature extraction regardless of sample crystallinity.

### 3.4 Preferred Orientation (Texture Effect)

**Physical Basis.** In polycrystalline samples, crystallites may exhibit non-random orientation distributions, causing systematic intensity deviations from powder-average values.

**Implementation.** We simulate texture by scaling peak intensities based on the alignment between Miller indices (hkl) and a randomly sampled preferred direction **p**:

```
I'_hkl = I_hkl · f(hkl, p)
f(hkl, p) = (1 - τ_max) + τ_max · |cos(hkl, p)|
```

where τ_max ∈ [0, 1] controls texture strength (typically 0.6).

**Relevance to Deep Learning.** Texture augmentation prevents the model from memorizing exact intensity ratios, forcing it to learn relative peak patterns.

### 3.5 Background Modeling

**Physical Basis.** Experimental XRD patterns contain background contributions from:
- Amorphous sample components
- Air scattering
- Sample holder scattering
- Incoherent (Compton) scattering

**Implementation.** We model background using low-order polynomials:

```
B(θ) = Σ_i c_i · P_i(θ),  c_i ~ U(-s, s)
```

where P_i are Chebyshev polynomials (order 3-6) and s controls amplitude.

### 3.6 Noise Injection

**Physical Basis.** XRD measurements are subject to two noise sources:

1. **Poisson Noise**: Photon counting statistics (dominant at low intensities)
2. **Gaussian Noise**: Electronic/thermal detector noise

**Implementation.**
```
I_noisy = I + N(0, σ)      # Gaussian
I_noisy = Poisson(I·k)/k   # Poisson
```
Default: σ = 0.25 (normalized intensity scale 0-100).

### 3.7 Augmentation Pipeline

The complete augmentation pipeline applies transformations sequentially:

```
Input: Peak positions {θ_i}, Intensities {I_i}, Miller indices {hkl_i}

1. Texture:     I'_i ← apply_texture(I_i, hkl_i, p)
2. Shift:       θ'_i ← θ_i + Δθ
3. Broadening:  S(θ) ← convolve(peaks, G(σ(θ, D)))
4. Background:  S'(θ) ← S(θ) + B(θ)
5. Noise:       S''(θ) ← S'(θ) + noise

Output: Augmented spectrum S''(θ)
```

### 3.8 Hyperparameters

| Parameter | Symbol | Range | Default | Physical Meaning |
|-----------|--------|-------|---------|------------------|
| Noise std | σ | 0.1-1.0 | 0.25 | Detector noise level |
| Max shift | δ_max | 0.1-1.0° | 0.5° | Sample displacement |
| Max strain | ε_max | 0.01-0.05 | 0.04 | Lattice distortion |
| Domain size | D | 1-100 nm | 5-100 | Crystallite size |
| Texture | τ_max | 0.3-0.8 | 0.6 | Orientation preference |

### 3.9 Integration with Deep Geometric Learning

**Motivation.** XRD spectra encode geometric information about crystal structures through Bragg's law. Deep geometric learning models must learn representations invariant to experimental artifacts while preserving structural information.

**Benefits for Representation Learning:**

1. **Translation Equivariance**: Peak shifts teach models to recognize patterns regardless of absolute position

2. **Scale Invariance**: Broadening variations ensure robustness to peak width changes

3. **Intensity Invariance**: Texture augmentation prevents overfitting to exact intensity ratios

4. **Noise Robustness**: Noise injection improves generalization to low-quality experimental data

5. **Domain Gap Bridging**: Physics-informed augmentation bridges the gap between simulated and experimental spectra

### 3.10 Usage Example

```python
from xrd_augmentor import XRDAugmentor, AugmentConfig, augment_xrd

# Method 1: One-line API
spectrum = augment_xrd(angles, intensities, preset='moderate')

# Method 2: Custom configuration
config = AugmentConfig(
    noise_std=0.3,
    max_shift=0.5,
    min_domain_size=5,
    max_domain_size=50,
    max_texture=0.6
)
aug = XRDAugmentor(config)
spectra = aug.augment_batch(angles, intensities, n_samples=100)
```

### References

[1] Szymanski et al., "Probabilistic Deep Learning Approach to Automate the Interpretation of Multi-phase Diffraction Spectra," Chemistry of Materials, 2021.

[2] Scherrer, P. "Bestimmung der Größe und der inneren Struktur von Kolloidteilchen mittels Röntgenstrahlen," Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen, 1918.
