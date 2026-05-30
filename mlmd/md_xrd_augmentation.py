"""
CHGNet-based MD Data Augmentation for XRD Simulation
Molecular-dynamics sampling and XRD calculation across ensembles.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms, units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
import os
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


def create_si_supercell(size=(2, 2, 2)):
    """Create a silicon supercell."""
    si = bulk('Si', 'diamond', a=5.43)
    si = si.repeat(size)
    return si


def run_nvt_langevin(atoms, calc, temperature=300, steps=500, dt=1.0, friction=0.01):
    """Run an NVT ensemble with a Langevin thermostat."""
    atoms = atoms.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = Langevin(atoms, dt * units.fs, temperature_K=temperature, friction=friction)
    snapshots = []
    for i in range(steps):
        dyn.run(1)
        if i % 50 == 0 and i > 100:
            snapshots.append(atoms.copy())
    return snapshots


def run_nvt_berendsen(atoms, calc, temperature=300, steps=500, dt=1.0, taut=100):
    """Run an NVT ensemble with a Berendsen thermostat."""
    atoms = atoms.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = NVTBerendsen(atoms, dt * units.fs, temperature_K=temperature, taut=taut * units.fs)
    snapshots = []
    for i in range(steps):
        dyn.run(1)
        if i % 50 == 0 and i > 100:
            snapshots.append(atoms.copy())
    return snapshots


def run_npt_berendsen(atoms, calc, temperature=300, steps=500, dt=1.0, taut=100, pressure=1.01325):
    """Run an NPT ensemble with Berendsen thermostatting and barostatting."""
    atoms = atoms.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = NPTBerendsen(
        atoms, dt * units.fs,
        temperature_K=temperature, taut=taut * units.fs,
        pressure_au=pressure * units.bar, taup=1000 * units.fs,
        compressibility_au=4.57e-5 / units.bar
    )
    snapshots = []
    for i in range(steps):
        dyn.run(1)
        if i % 50 == 0 and i > 100:
            snapshots.append(atoms.copy())
    return snapshots


def ase_to_pymatgen(atoms):
    """Convert ASE Atoms to a pymatgen Structure."""
    return AseAtomsAdaptor.get_structure(atoms)


def save_snapshots_as_cif(snapshots, ensemble_name, output_dir="si_structures"):
    """Save MD snapshots as CIF files.

    Args:
        snapshots: List of ASE Atoms objects.
        ensemble_name: Ensemble name used in file names.
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for i, atoms in enumerate(snapshots):
        struct = ase_to_pymatgen(atoms)
        filename = f"{output_dir}/Si_{ensemble_name}_snapshot_{i:02d}.cif"
        CifWriter(struct).write_file(filename)
        saved_files.append(filename)

    return saved_files


def calculate_xrd_pattern(structure, two_theta_range=(10, 90)):
    """Calculate an XRD pattern."""
    xrd_calc = XRDCalculator(wavelength='CuKa')
    pattern = xrd_calc.get_pattern(structure, two_theta_range=two_theta_range)
    return pattern.x, pattern.y


def average_xrd_from_snapshots(snapshots, two_theta_range=(10, 90)):
    """Calculate an averaged XRD pattern from multiple snapshots."""
    all_patterns = []
    two_theta = None

    for atoms in snapshots:
        struct = ase_to_pymatgen(atoms)
        x, y = calculate_xrd_pattern(struct, two_theta_range)
        if two_theta is None:
            two_theta = x
        all_patterns.append(y)

    avg_pattern = np.mean(all_patterns, axis=0)
    return two_theta, avg_pattern


def broaden_pattern(two_theta, intensities, sigma=0.1, theta_min=10, theta_max=90):
    """Apply Gaussian broadening to an XRD pattern."""
    theta_fine = np.linspace(theta_min, theta_max, 1000)
    broadened = np.zeros_like(theta_fine)
    for t, i in zip(two_theta, intensities):
        broadened += i * np.exp(-0.5 * ((theta_fine - t) / sigma) ** 2)
    return theta_fine, broadened


def main():
    """Run MD sampling routes and compare the resulting XRD patterns."""
    print("=" * 60)
    print("CHGNet MD augmentation and XRD simulation")
    print("=" * 60)

    print("\n[1] Creating Si supercell...")
    si_atoms = create_si_supercell(size=(2, 2, 2))
    print(f"    Number of atoms: {len(si_atoms)}")

    print("\n[2] Loading CHGNet model...")
    chgnet = CHGNet.load()
    calc = CHGNetCalculator(chgnet, use_device="cuda")

    temperature = 300  # K
    steps = 400
    dt = 1.0  # fs

    results = {}

    print("\n[3] Calculating XRD for the original structure...")
    orig_struct = ase_to_pymatgen(si_atoms)
    orig_x, orig_y = calculate_xrd_pattern(orig_struct)
    results['Original'] = (orig_x, orig_y)

    output_dir = "si_structures"
    os.makedirs(output_dir, exist_ok=True)
    CifWriter(orig_struct).write_file(f"{output_dir}/Si_original.cif")
    print(f"    Saved: {output_dir}/Si_original.cif")

    print("\n[4] Running NVT-Langevin MD...")
    snapshots_langevin = run_nvt_langevin(si_atoms, calc, temperature, steps, dt)
    print(f"    Number of sampled structures: {len(snapshots_langevin)}")
    x_lang, y_lang = average_xrd_from_snapshots(snapshots_langevin)
    results['NVT-Langevin'] = (x_lang, y_lang)
    cif_files = save_snapshots_as_cif(snapshots_langevin, "NVT-Langevin", output_dir)
    print(f"    Saved {len(cif_files)} CIF files")

    print("\n[5] Running NVT-Berendsen MD...")
    snapshots_ber = run_nvt_berendsen(si_atoms, calc, temperature, steps, dt)
    print(f"    Number of sampled structures: {len(snapshots_ber)}")
    x_ber, y_ber = average_xrd_from_snapshots(snapshots_ber)
    results['NVT-Berendsen'] = (x_ber, y_ber)
    cif_files = save_snapshots_as_cif(snapshots_ber, "NVT-Berendsen", output_dir)
    print(f"    Saved {len(cif_files)} CIF files")

    print("\n[6] Running NPT-Berendsen MD...")
    snapshots_npt = run_npt_berendsen(si_atoms, calc, temperature, steps, dt)
    print(f"    Number of sampled structures: {len(snapshots_npt)}")
    x_npt, y_npt = average_xrd_from_snapshots(snapshots_npt)
    results['NPT-Berendsen'] = (x_npt, y_npt)
    cif_files = save_snapshots_as_cif(snapshots_npt, "NPT-Berendsen", output_dir)
    print(f"    Saved {len(cif_files)} CIF files")

    print("\n[7] Generating comparison plots...")
    import pandas as pd

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ['black', 'blue', 'red', 'green']
    labels = list(results.keys())

    ax1 = axes[0, 0]
    x, y = results['Original']
    x_b, y_b = broaden_pattern(x, y, sigma=0.3)
    ax1.plot(x_b, y_b / y_b.max(), 'k-', lw=1.5)
    ax1.set_title('Original Structure', fontsize=14)
    ax1.set_xlabel('2θ (degree)', fontsize=12)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax1.set_xlim(10, 80)
    ax1.tick_params(axis='both', labelsize=11)

    ax2 = axes[0, 1]
    x, y = results['NVT-Langevin']
    x_b, y_b = broaden_pattern(x, y, sigma=0.3)
    ax2.plot(x_b, y_b / y_b.max(), 'b-', lw=1.5)
    ax2.set_title('NVT-Langevin (300K)', fontsize=14)
    ax2.set_xlabel('2θ (degree)', fontsize=12)
    ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax2.set_xlim(10, 80)
    ax2.tick_params(axis='both', labelsize=11)

    ax3 = axes[1, 0]
    x, y = results['NVT-Berendsen']
    x_b, y_b = broaden_pattern(x, y, sigma=0.3)
    ax3.plot(x_b, y_b / y_b.max(), 'r-', lw=1.5)
    ax3.set_title('NVT-Berendsen (300K)', fontsize=14)
    ax3.set_xlabel('2θ (degree)', fontsize=12)
    ax3.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax3.set_xlim(10, 80)
    ax3.tick_params(axis='both', labelsize=11)

    ax4 = axes[1, 1]
    x, y = results['NPT-Berendsen']
    x_b, y_b = broaden_pattern(x, y, sigma=0.3)
    ax4.plot(x_b, y_b / y_b.max(), 'g-', lw=1.5)
    ax4.set_title('NPT-Berendsen (300K, 1atm)', fontsize=14)
    ax4.set_xlabel('2θ (degree)', fontsize=12)
    ax4.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax4.set_xlim(10, 80)
    ax4.tick_params(axis='both', labelsize=11)

    plt.suptitle('Si XRD Patterns: MD Augmentation Comparison (CHGNet)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/si_xrd_md_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/si_xrd_md_comparison.pdf', bbox_inches='tight')
    print(f"    Saved: {output_dir}/si_xrd_md_comparison.png/pdf")

    fig2, ax = plt.subplots(figsize=(10, 6))
    for i, (label, (x, y)) in enumerate(results.items()):
        x_b, y_b = broaden_pattern(x, y, sigma=0.3)
        offset = i * 0.3
        ax.plot(x_b, y_b / y_b.max() + offset, colors[i], lw=1.5, label=label)

    ax.set_xlabel('2θ (degree)', fontsize=13)
    ax.set_ylabel('Intensity (a.u.) + offset', fontsize=13)
    ax.set_title('Si XRD: MD Methods Comparison (CHGNet, 300K)', fontsize=14)
    ax.set_xlim(10, 80)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/si_xrd_overlay.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/si_xrd_overlay.pdf', bbox_inches='tight')
    print(f"    Saved: {output_dir}/si_xrd_overlay.png/pdf")

    csv_data = {}
    for label, (x, y) in results.items():
        x_b, y_b = broaden_pattern(x, y, sigma=0.3)
        csv_data[f'{label}_2theta'] = x_b
        csv_data[f'{label}_intensity'] = y_b / y_b.max()
    df = pd.DataFrame(csv_data)
    df.to_csv(f'{output_dir}/si_xrd_md_data.csv', index=False)
    print(f"    Saved: {output_dir}/si_xrd_md_data.csv")

    print("\n[8] Drawing crystal structures...")
    from ase.io import write

    write(f'{output_dir}/Si_original_structure.png', si_atoms, rotation='10x,-10y')
    print(f"    Saved: {output_dir}/Si_original_structure.png")

    all_snapshots = {
        'NVT-Langevin': snapshots_langevin,
        'NVT-Berendsen': snapshots_ber,
        'NPT-Berendsen': snapshots_npt
    }
    for ensemble_name, snapshots in all_snapshots.items():
        for i, atoms in enumerate(snapshots):
            img_path = f'{output_dir}/Si_{ensemble_name}_snapshot_{i:02d}.png'
            write(img_path, atoms, rotation='10x,-10y')
        print(f"    Saved: {output_dir}/Si_{ensemble_name}_snapshot_*.png ({len(snapshots)} images)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
