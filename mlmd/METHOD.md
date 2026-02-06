# Machine Learning Molecular Dynamics for XRD Data Augmentation

## Method

### 3.1 Overview

We propose a physics-based data augmentation framework that leverages Machine Learning Interatomic Potentials (MLIPs) to generate thermally perturbed crystal structures through molecular dynamics (MD) simulations. Unlike parametric augmentation methods that apply empirical transformations to XRD spectra, our approach samples physically realistic atomic configurations from thermodynamic ensembles, producing augmented XRD patterns that capture genuine thermal effects observed in experimental measurements.

The framework employs CHGNet, a universal neural network potential trained on the Materials Project database, to drive MD simulations across multiple statistical ensembles: NVT-Langevin, NVT-Berendsen, and NPT-Berendsen. Each ensemble samples distinct thermodynamic conditions, generating diverse structural perturbations that manifest as characteristic modifications in the resulting XRD patterns.

### 3.2 Machine Learning Interatomic Potentials

**Physical Basis.** Traditional ab initio molecular dynamics (AIMD) computes interatomic forces from density functional theory (DFT) at each timestep, achieving high accuracy but at prohibitive computational cost with O(N³) scaling where N is the number of electrons. MLIPs approximate the potential energy surface learned from DFT training data, enabling near-DFT accuracy at classical MD computational speeds.

**Energy Model.** The total potential energy E of a crystal configuration is modeled as:

```
E = f_θ(G(r, Z))
```

where:
- E: total potential energy of the system (eV)
- f_θ: neural network function with learnable parameters θ
- G: crystal graph representation
- r = {r₁, r₂, ..., rₙ}: atomic position vectors (Å)
- Z = {Z₁, Z₂, ..., Zₙ}: atomic numbers

The interatomic forces are obtained via automatic differentiation:

```
F_i = -∂E/∂r_i
```

where F_i is the force vector acting on atom i.

**Relevance to Deep Learning.** Using MLIPs ensures that augmented structures remain on the physical potential energy surface, avoiding unphysical configurations that could introduce artifacts in downstream learning tasks.

### 3.3 Thermodynamic Ensembles

**Physical Basis.** Statistical mechanics defines ensembles as collections of microstates consistent with macroscopic thermodynamic constraints. Different ensembles sample distinct regions of phase space, each characterized by conserved quantities:

| Ensemble | Conserved Quantities | Fluctuating Quantities |
|----------|---------------------|------------------------|
| NVT (Canonical) | N, V, T | E, P |
| NPT (Isothermal-Isobaric) | N, P, T | E, V |

where:
- N: number of particles (dimensionless)
- V: system volume (Å³)
- T: temperature (K)
- E: total energy (eV)
- P: pressure (bar or GPa)

The probability of observing a microstate with energy E in the canonical ensemble follows the Boltzmann distribution:

```
P(E) ∝ exp(-E / k_B T)
```

where k_B = 8.617 × 10⁻⁵ eV/K is the Boltzmann constant.

### 3.4 NVT-Langevin Dynamics

**Physical Basis.** The Langevin equation augments Newton's equations of motion with dissipative and stochastic terms that model coupling to a heat bath:

```
m_i (d²r_i/dt²) = F_i - γ m_i (dr_i/dt) + R_i(t)
```

where:
- m_i: mass of atom i (amu)
- r_i: position vector of atom i (Å)
- t: time (fs)
- F_i: conservative force from the interatomic potential (eV/Å)
- γ: friction coefficient (fs⁻¹)
- R_i(t): random force vector (eV/Å)

The random force satisfies the fluctuation-dissipation theorem:

```
⟨R_i(t)⟩ = 0
⟨R_i(t) · R_j(t')⟩ = 2 γ m_i k_B T δ_ij δ(t - t')
```

where:
- ⟨·⟩: ensemble average
- δ_ij: Kronecker delta (1 if i=j, 0 otherwise)
- δ(t - t'): Dirac delta function

**Relevance to Deep Learning.** Langevin dynamics produces uncorrelated samples suitable for training data augmentation, as the stochastic term decorrelates successive configurations more rapidly than deterministic thermostats.

### 3.5 NVT-Berendsen Thermostat

**Physical Basis.** The Berendsen thermostat rescales atomic velocities to drive the instantaneous kinetic temperature toward the target value through weak coupling:

```
dT/dt = (T_target - T) / τ_T
```

where:
- T: instantaneous temperature (K)
- T_target: target temperature (K)
- τ_T: temperature coupling time constant (fs)

The instantaneous temperature is computed from the kinetic energy:

```
T = (2/3Nk_B) Σ_i (m_i v_i²/2)
```

where v_i = dr_i/dt is the velocity of atom i.

At each timestep, velocities are rescaled by factor λ:

```
λ = √(1 + (Δt/τ_T)(T_target/T - 1))
v_i → λ v_i
```

where Δt is the integration timestep (fs).

**Relevance to Deep Learning.** Berendsen produces smoother temperature trajectories than Langevin, useful when gradual thermal equilibration is desired before sampling.

### 3.6 NPT-Berendsen Barostat

**Physical Basis.** The NPT ensemble allows both temperature and pressure control, enabling simulation cell volume fluctuations that capture thermal expansion effects. The Berendsen barostat rescales the simulation cell vectors:

```
dP/dt = (P_target - P) / τ_P
```

where:
- P: instantaneous pressure (bar)
- P_target: target pressure (bar)
- τ_P: pressure coupling time constant (fs)

The pressure is computed from the virial theorem:

```
P = (Nk_B T/V) + (1/3V) Σ_i r_i · F_i
```

The cell scaling factor μ is:

```
μ = ∛(1 - κ(Δt/τ_P)(P_target - P))
```

where κ is the isothermal compressibility (bar⁻¹). The cell vectors h and atomic positions are scaled:

```
h → μ h
r_i → μ r_i
```

**Relevance to Deep Learning.** NPT sampling captures thermal expansion effects, producing lattice parameter variations that directly affect XRD peak positions—a critical augmentation for bridging simulation-experiment domain gaps.

### 3.7 XRD Pattern Calculation

**Physical Basis.** For each MD snapshot, we compute the XRD pattern using kinematic diffraction theory. The structure factor for reflection (hkl) is:

```
F_hkl = Σ_j f_j(s) · exp(2πi(hx_j + ky_j + lz_j)) · exp(-B_j s²)
```

where:
- F_hkl: structure factor (complex, electrons)
- h, k, l: Miller indices (dimensionless integers)
- f_j(s): atomic scattering factor of atom j (electrons)
- s = sin(θ)/λ: scattering vector magnitude (Å⁻¹)
- (x_j, y_j, z_j): fractional coordinates of atom j
- B_j: atomic displacement parameter (Å²)
- i: imaginary unit

The diffracted intensity follows:

```
I_hkl = |F_hkl|² · LP(θ) · m_hkl
```

where:
- I_hkl: integrated intensity (arbitrary units)
- LP(θ): Lorentz-polarization factor
- m_hkl: multiplicity of the reflection

The Lorentz-polarization factor for unpolarized X-rays is:

```
LP(θ) = (1 + cos²(2θ)) / (sin²(θ) cos(θ))
```

Peak positions are determined by Bragg's law:

```
2d_hkl sin(θ) = nλ
```

where:
- d_hkl: interplanar spacing (Å)
- θ: Bragg angle (degrees)
- n: diffraction order (typically 1)
- λ: X-ray wavelength (Å), λ = 1.5406 Å for Cu Kα

### 3.8 Ensemble Averaging

**Physical Basis.** Experimental XRD measurements average over ~10²³ unit cells and measurement timescales (~seconds), effectively sampling the thermodynamic ensemble. We approximate this by averaging patterns from N_s MD snapshots:

```
I_avg(2θ) = (1/N_s) Σ_{i=1}^{N_s} I_i(2θ)
```

where:
- I_avg(2θ): ensemble-averaged intensity at angle 2θ
- N_s: number of sampled snapshots
- I_i(2θ): intensity from snapshot i

**Sampling Protocol.** To ensure statistical independence, we:
1. Discard initial N_equil equilibration steps
2. Sample every N_interval steps to reduce autocorrelation

The effective sampling criterion is:

```
t_sample > τ_corr
```

where:
- t_sample = N_interval × Δt: time between samples (fs)
- τ_corr: velocity autocorrelation time (fs)

### 3.9 Peak Broadening Post-Processing

**Physical Basis.** MD-derived XRD patterns contain discrete Bragg peaks. Experimental patterns exhibit broadening from multiple sources:
- Finite crystallite size (Scherrer broadening)
- Instrumental resolution function
- Microstrain distribution

We apply Gaussian convolution to simulate instrumental broadening:

```
I_broad(2θ) = I(2θ) ⊗ G(2θ; σ)
```

where:
- ⊗: convolution operator
- G(2θ; σ): Gaussian kernel with standard deviation σ

The Gaussian kernel is:

```
G(2θ; σ) = (1/√(2πσ²)) exp(-(2θ)²/(2σ²))
```

where σ is the broadening parameter (degrees), typically 0.1-0.5° for laboratory diffractometers.

### 3.10 Hyperparameters

| Parameter | Symbol | Range | Default | Physical Meaning |
|-----------|--------|-------|---------|------------------|
| Temperature | T | 100-1000 K | 300 K | Thermal energy scale |
| Timestep | Δt | 0.5-2.0 fs | 1.0 fs | Integration accuracy |
| Total steps | N_steps | 200-2000 | 400-500 | Total simulation duration |
| Sampling interval | N_interval | 20-100 | 50 | Steps between snapshots |
| Equilibration steps | N_equil | 50-200 | 100 | Thermalization period |
| Friction coefficient | γ | 0.001-0.1 fs⁻¹ | 0.01 fs⁻¹ | Langevin heat bath coupling |
| Temperature coupling | τ_T | 50-500 fs | 100 fs | Berendsen relaxation time |
| Pressure coupling | τ_P | 500-2000 fs | 1000 fs | Barostat relaxation time |
| Target pressure | P_target | 0-10 GPa | 1 atm | Thermodynamic pressure |
| Broadening | σ | 0.1-0.5° | 0.3° | Instrumental peak width |
| Supercell size | (n_a, n_b, n_c) | (2,2,2)-(4,4,4) | (2,2,2) | Finite size control |

### 3.11 Integration with Deep Geometric Learning

**Motivation.** Deep geometric learning models for crystal property prediction must generalize from idealized DFT-relaxed structures (0 K) to experimental measurements that inherently contain thermal disorder (finite T). MD-based augmentation bridges this domain gap by exposing models to physically realistic thermal perturbations.

**Benefits for Representation Learning:**

1. **Thermal Robustness**: Models learn representations invariant to thermal atomic displacements characterized by mean-squared displacement ⟨u²⟩ ∝ T

2. **Lattice Parameter Variation**: NPT sampling produces thermal expansion following:
   ```
   a(T) = a₀(1 + α_L ΔT)
   ```
   where a₀ is the 0 K lattice parameter and α_L is the linear thermal expansion coefficient (K⁻¹)

3. **Peak Position Shifts**: Thermal expansion causes systematic peak shifts:
   ```
   Δ(2θ) ≈ -2θ · α_L · ΔT · cot(θ)
   ```

4. **Intensity Modulation**: Atomic thermal motion reduces peak intensities via the Debye-Waller factor:
   ```
   I(T) = I₀ · exp(-2B sin²(θ)/λ²)
   ```
   where B = 8π²⟨u²⟩ is the temperature factor (Å²)

5. **Physical Consistency**: Unlike parametric augmentation, MD-generated structures satisfy:
   - Conservation of crystal symmetry
   - Physically reasonable bond lengths and angles
   - Proper phonon mode sampling

**Comparison with Parametric Augmentation:**

| Aspect | Parametric (noise/) | MD-based (mlmd/) |
|--------|---------------------|------------------|
| Physical basis | Empirical parameters | First-principles thermodynamics |
| Computational cost | O(N) - milliseconds | O(N²) - minutes to hours |
| Thermal effects | Approximated via broadening | Exact ensemble sampling |
| Lattice dynamics | Not captured | Naturally included |
| Anharmonicity | Not captured | Fully captured |
| Phase transitions | Cannot model | Can detect/sample |
| Use case | Large-scale augmentation | High-fidelity simulation |

### 3.12 Theoretical Foundations

**Ergodic Hypothesis.** The validity of MD-based augmentation relies on the ergodic hypothesis:

```
⟨A⟩_ensemble = lim_{τ→∞} (1/τ) ∫₀^τ A(t) dt
```

where ⟨A⟩_ensemble is the ensemble average of observable A, and the right-hand side is the time average. This ensures that sufficiently long MD trajectories sample the equilibrium distribution.

**Detailed Balance.** The thermostats employed satisfy detailed balance:

```
P(s → s') · π(s) = P(s' → s) · π(s')
```

where P(s → s') is the transition probability from state s to s', and π(s) is the equilibrium probability of state s. This guarantees sampling from the correct thermodynamic ensemble.

### References

[1] Deng, B. et al. "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling," Nature Machine Intelligence, 2023.

[2] Jain, A. et al. "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation," APL Materials, 2013.

[3] Berendsen, H. J. C. et al. "Molecular dynamics with coupling to an external bath," The Journal of Chemical Physics, 1984.

[4] Allen, M. P. & Tildesley, D. J. "Computer Simulation of Liquids," Oxford University Press, 2017.

[5] Warren, B. E. "X-ray Diffraction," Dover Publications, 1990.
