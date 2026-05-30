# Si XRD Augmentation Analysis

This directory contains XRD augmentation examples for diamond-structure Si.

## Structure Information

| Parameter | Value |
| --- | --- |
| Material | Si |
| Structure type | Diamond |
| Space group | Fd-3m (No. 227) |
| Lattice constant | a = 5.43 Angstrom |
| X-ray wavelength | Cu K-alpha (1.5406 Angstrom) |
| 2theta range | 10-80 degrees |

## Directory Layout

```text
analysis/
├── README.md
├── plots/
│   ├── augmented_xrd.png/pdf
│   ├── si_xrd_md_comparison.png/pdf
│   └── si_xrd_overlay.png/pdf
├── structures/
│   ├── Si_original.cif
│   ├── Si_original_structure.png
│   ├── Si_parametric_original.cif
│   ├── Si_NVT-Langevin_snapshot_*.cif
│   ├── Si_NVT-Langevin_snapshot_*.png
│   ├── Si_NVT-Berendsen_snapshot_*.cif
│   ├── Si_NVT-Berendsen_snapshot_*.png
│   ├── Si_NPT-Berendsen_snapshot_*.cif
│   └── Si_NPT-Berendsen_snapshot_*.png
└── data/
    ├── two_theta.npy
    ├── spectra_clean.npy
    ├── spectra_shifted.npy
    ├── spectra_broadened.npy
    ├── spectra_textured.npy
    ├── spectra_mixed.npy
    ├── augmented_xrd.csv
    └── si_xrd_md_data.csv
```

## Augmentation Methods

### Parametric augmentation

These examples use fast signal-level transformations.

| Type | Interpretation | Parameter range |
| --- | --- | --- |
| Clean | Gaussian noise only | sigma = 0.25 |
| Shifted | Uniform peak-position shift | +/-0.5 degrees |
| Broadened | Scherrer-type peak broadening | D = 5-50 nm |
| Textured | Preferred-orientation effect | tau = 0.6 |
| Mixed | Combined perturbations | - |

### Molecular-dynamics augmentation

These examples use thermally perturbed structures sampled with CHGNet.

| Ensemble | Temperature | Pressure | Description |
| --- | --- | --- | --- |
| Original | 0 K | - | Ideal crystal structure |
| NVT-Langevin | 300 K | - | Stochastic heat-bath coupling |
| NVT-Berendsen | 300 K | - | Weak-coupling thermostat |
| NPT-Berendsen | 300 K | 1 atm | Thermostat and barostat with thermal expansion |

MD parameters:

- Time step: 1.0 fs
- Total steps: 400
- Sampling interval: 50 steps
- Equilibration period: 100 steps
- Samples per ensemble: 5 structures

## Plot Descriptions

`augmented_xrd.png` compares five parametric augmentation methods for Si XRD.

`si_xrd_md_comparison.png` compares the original Si structure with three MD-derived ensembles.

`si_xrd_overlay.png` overlays the four MD-derived spectra to compare peak-position and intensity changes.

## Data Loading Example

```python
import numpy as np

two_theta = np.load("data/two_theta.npy")
spectra_mixed = np.load("data/spectra_mixed.npy")
print(f"Shape: {spectra_mixed.shape}")
```

## References

1. Szymanski et al., "Probabilistic Deep Learning Approach to Automate the Interpretation of Multi-phase Diffraction Spectra", Chemistry of Materials, 2021.
2. Deng et al., "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling", Nature Machine Intelligence, 2023.
