"""
CHGNet-based MD Data Augmentation for XRD Simulation
多种系综下的分子动力学采样与XRD谱图计算
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
    """创建Si超胞"""
    si = bulk('Si', 'diamond', a=5.43)
    si = si.repeat(size)
    return si


def run_nvt_langevin(atoms, calc, temperature=300, steps=500, dt=1.0, friction=0.01):
    """NVT系综 - Langevin恒温器"""
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
    """NVT系综 - Berendsen恒温器"""
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
    """NPT系综 - Berendsen恒温恒压"""
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
    """ASE Atoms转换为Pymatgen Structure"""
    return AseAtomsAdaptor.get_structure(atoms)


def save_snapshots_as_cif(snapshots, ensemble_name, output_dir="si_structures"):
    """将MD快照保存为CIF文件

    Args:
        snapshots: ASE Atoms对象列表
        ensemble_name: 系综名称 (用于文件命名)
        output_dir: 输出目录
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
    """计算XRD谱图"""
    xrd_calc = XRDCalculator(wavelength='CuKa')
    pattern = xrd_calc.get_pattern(structure, two_theta_range=two_theta_range)
    return pattern.x, pattern.y


def average_xrd_from_snapshots(snapshots, two_theta_range=(10, 90)):
    """从多个快照计算平均XRD谱图"""
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
    """对XRD谱图进行高斯展宽"""
    theta_fine = np.linspace(theta_min, theta_max, 1000)
    broadened = np.zeros_like(theta_fine)
    for t, i in zip(two_theta, intensities):
        broadened += i * np.exp(-0.5 * ((theta_fine - t) / sigma) ** 2)
    return theta_fine, broadened


def main():
    """主函数：运行多种MD方法并比较XRD"""
    print("=" * 60)
    print("MACE MD数据增广与XRD模拟")
    print("=" * 60)

    # 创建Si超胞
    print("\n[1] 创建Si超胞...")
    si_atoms = create_si_supercell(size=(2, 2, 2))
    print(f"    原子数: {len(si_atoms)}")

    # 加载CHGNet计算器
    print("\n[2] 加载CHGNet模型...")
    chgnet = CHGNet.load()
    calc = CHGNetCalculator(chgnet, use_device="cuda")

    # 参数设置
    temperature = 300  # K
    steps = 400
    dt = 1.0  # fs

    results = {}

    # 原始结构XRD
    print("\n[3] 计算原始结构XRD...")
    orig_struct = ase_to_pymatgen(si_atoms)
    orig_x, orig_y = calculate_xrd_pattern(orig_struct)
    results['Original'] = (orig_x, orig_y)

    # 保存原始结构
    output_dir = "si_structures"
    os.makedirs(output_dir, exist_ok=True)
    CifWriter(orig_struct).write_file(f"{output_dir}/Si_original.cif")
    print(f"    已保存: {output_dir}/Si_original.cif")

    # NVT Langevin
    print("\n[4] 运行NVT-Langevin MD...")
    snapshots_langevin = run_nvt_langevin(si_atoms, calc, temperature, steps, dt)
    print(f"    采样构型数: {len(snapshots_langevin)}")
    x_lang, y_lang = average_xrd_from_snapshots(snapshots_langevin)
    results['NVT-Langevin'] = (x_lang, y_lang)
    # 保存NVT-Langevin快照
    cif_files = save_snapshots_as_cif(snapshots_langevin, "NVT-Langevin", output_dir)
    print(f"    已保存 {len(cif_files)} 个CIF文件")

    # NVT Berendsen
    print("\n[5] 运行NVT-Berendsen MD...")
    snapshots_ber = run_nvt_berendsen(si_atoms, calc, temperature, steps, dt)
    print(f"    采样构型数: {len(snapshots_ber)}")
    x_ber, y_ber = average_xrd_from_snapshots(snapshots_ber)
    results['NVT-Berendsen'] = (x_ber, y_ber)
    # 保存NVT-Berendsen快照
    cif_files = save_snapshots_as_cif(snapshots_ber, "NVT-Berendsen", output_dir)
    print(f"    已保存 {len(cif_files)} 个CIF文件")

    # NPT Berendsen
    print("\n[6] 运行NPT-Berendsen MD...")
    snapshots_npt = run_npt_berendsen(si_atoms, calc, temperature, steps, dt)
    print(f"    采样构型数: {len(snapshots_npt)}")
    x_npt, y_npt = average_xrd_from_snapshots(snapshots_npt)
    results['NPT-Berendsen'] = (x_npt, y_npt)
    # 保存NPT-Berendsen快照
    cif_files = save_snapshots_as_cif(snapshots_npt, "NPT-Berendsen", output_dir)
    print(f"    已保存 {len(cif_files)} 个CIF文件")

    # 绘图
    print("\n[7] 生成对比图...")
    import pandas as pd

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ['black', 'blue', 'red', 'green']
    labels = list(results.keys())

    # 子图1: 原始结构
    ax1 = axes[0, 0]
    x, y = results['Original']
    x_b, y_b = broaden_pattern(x, y, sigma=0.3)
    ax1.plot(x_b, y_b / y_b.max(), 'k-', lw=1.5)
    ax1.set_title('Original Structure', fontsize=14)
    ax1.set_xlabel('2θ (degree)', fontsize=12)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax1.set_xlim(10, 80)
    ax1.tick_params(axis='both', labelsize=11)

    # 子图2: NVT-Langevin
    ax2 = axes[0, 1]
    x, y = results['NVT-Langevin']
    x_b, y_b = broaden_pattern(x, y, sigma=0.3)
    ax2.plot(x_b, y_b / y_b.max(), 'b-', lw=1.5)
    ax2.set_title('NVT-Langevin (300K)', fontsize=14)
    ax2.set_xlabel('2θ (degree)', fontsize=12)
    ax2.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax2.set_xlim(10, 80)
    ax2.tick_params(axis='both', labelsize=11)

    # 子图3: NVT-Berendsen
    ax3 = axes[1, 0]
    x, y = results['NVT-Berendsen']
    x_b, y_b = broaden_pattern(x, y, sigma=0.3)
    ax3.plot(x_b, y_b / y_b.max(), 'r-', lw=1.5)
    ax3.set_title('NVT-Berendsen (300K)', fontsize=14)
    ax3.set_xlabel('2θ (degree)', fontsize=12)
    ax3.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax3.set_xlim(10, 80)
    ax3.tick_params(axis='both', labelsize=11)

    # 子图4: NPT-Berendsen
    ax4 = axes[1, 1]
    x, y = results['NPT-Berendsen']
    x_b, y_b = broaden_pattern(x, y, sigma=0.3)
    ax4.plot(x_b, y_b / y_b.max(), 'g-', lw=1.5)
    ax4.set_title('NPT-Berendsen (300K, 1atm)', fontsize=14)
    ax4.set_xlabel('2θ (degree)', fontsize=12)
    ax4.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax4.set_xlim(10, 80)
    ax4.tick_params(axis='both', labelsize=11)

    plt.suptitle('Si XRD Patterns: MD Augmentation Comparison (MACE)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/si_xrd_md_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/si_xrd_md_comparison.pdf', bbox_inches='tight')
    print(f"    图片已保存: {output_dir}/si_xrd_md_comparison.png/pdf")

    # 叠加对比图
    fig2, ax = plt.subplots(figsize=(10, 6))
    for i, (label, (x, y)) in enumerate(results.items()):
        x_b, y_b = broaden_pattern(x, y, sigma=0.3)
        offset = i * 0.3
        ax.plot(x_b, y_b / y_b.max() + offset, colors[i], lw=1.5, label=label)

    ax.set_xlabel('2θ (degree)', fontsize=13)
    ax.set_ylabel('Intensity (a.u.) + offset', fontsize=13)
    ax.set_title('Si XRD: MD Methods Comparison (MACE, 300K)', fontsize=14)
    ax.set_xlim(10, 80)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/si_xrd_overlay.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/si_xrd_overlay.pdf', bbox_inches='tight')
    print(f"    图片已保存: {output_dir}/si_xrd_overlay.png/pdf")

    # 保存CSV数据
    csv_data = {}
    for label, (x, y) in results.items():
        x_b, y_b = broaden_pattern(x, y, sigma=0.3)
        csv_data[f'{label}_2theta'] = x_b
        csv_data[f'{label}_intensity'] = y_b / y_b.max()
    df = pd.DataFrame(csv_data)
    df.to_csv(f'{output_dir}/si_xrd_md_data.csv', index=False)
    print(f"    数据已保存: {output_dir}/si_xrd_md_data.csv")

    # [8] 使用ASE绘制晶体结构
    print("\n[8] 绘制晶体结构...")
    from ase.io import write

    # 绘制原始结构
    write(f'{output_dir}/Si_original_structure.png', si_atoms, rotation='10x,-10y')
    print(f"    已保存: {output_dir}/Si_original_structure.png")

    # 绘制各系综的采样结构
    all_snapshots = {
        'NVT-Langevin': snapshots_langevin,
        'NVT-Berendsen': snapshots_ber,
        'NPT-Berendsen': snapshots_npt
    }
    for ensemble_name, snapshots in all_snapshots.items():
        for i, atoms in enumerate(snapshots):
            img_path = f'{output_dir}/Si_{ensemble_name}_snapshot_{i:02d}.png'
            write(img_path, atoms, rotation='10x,-10y')
        print(f"    已保存: {output_dir}/Si_{ensemble_name}_snapshot_*.png ({len(snapshots)}张)")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
