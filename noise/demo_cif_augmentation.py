"""
Demo: Generate augmented XRD spectra from CIF files using pymatgen
==================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

# Import our augmentation module
from ceder_augmentation import (
    UniformShiftAugmenter,
    PeakBroadeningAugmenter,
    TextureAugmenter,
    MixedAugmenter,
    apply_peak_broadening,
    add_gaussian_noise
)


def get_xrd_from_structure(structure, min_angle=10.0, max_angle=80.0):
    """
    从pymatgen Structure对象计算XRD峰位和强度
    """
    calculator = XRDCalculator(wavelength='CuKa')
    pattern = calculator.get_pattern(structure, two_theta_range=(min_angle, max_angle))

    angles = pattern.x
    intensities = pattern.y
    hkls = [v[0]['hkl'] for v in pattern.hkls]

    return angles, intensities, hkls


def generate_augmented_spectra(cif_path, n_augment=10, min_angle=10.0, max_angle=80.0):
    """
    从CIF文件生成多个增广XRD谱

    Args:
        cif_path: CIF文件路径
        n_augment: 每种增广方法生成的谱数量
        min_angle, max_angle: 2theta范围
    """
    # 读取结构
    structure = Structure.from_file(cif_path)
    print(f"Loaded: {structure.formula}")
    print(f"Space group: {structure.get_space_group_info()}")

    # 计算理论XRD
    angles, intensities, hkls = get_xrd_from_structure(
        structure, min_angle, max_angle
    )
    is_hex = structure.lattice.is_hexagonal()

    # 初始化增广器
    shift_aug = UniformShiftAugmenter(max_shift=0.5)
    broad_aug = PeakBroadeningAugmenter(min_domain_size=5, max_domain_size=50)
    texture_aug = TextureAugmenter(max_texture=0.6)
    mixed_aug = MixedAugmenter()

    results = {
        'clean': [],
        'shifted': [],
        'broadened': [],
        'textured': [],
        'mixed': []
    }

    # 生成干净谱（仅加噪声）
    for _ in range(n_augment):
        spec = apply_peak_broadening(angles, intensities, 25.0, min_angle, max_angle)
        spec = add_gaussian_noise(spec, 0.25)
        results['clean'].append(spec)

    # 生成各类增广谱
    for _ in range(n_augment):
        results['shifted'].append(shift_aug.augment(angles, intensities))
        results['broadened'].append(broad_aug.augment(angles, intensities))
        results['textured'].append(texture_aug.augment(angles, intensities, hkls, is_hex))
        results['mixed'].append(mixed_aug.augment(angles, intensities, hkls, is_hex))

    return results, np.linspace(min_angle, max_angle, 4501)


def plot_augmented_spectra(results, two_theta, output_dir, save_path=None):
    """可视化不同增广方法的效果"""
    import os
    import pandas as pd

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    titles = ['Clean (noise only)', 'Uniform Shift', 'Peak Broadening',
              'Texture', 'Mixed', 'All Overlaid']
    keys = ['clean', 'shifted', 'broadened', 'textured', 'mixed']
    method_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 前5个子图：每种增强方法的多个样本叠加
    for i, (key, title) in enumerate(zip(keys, titles[:5])):
        ax = axes[i]
        for spec in results[key][:5]:
            ax.plot(two_theta, spec, alpha=0.6, linewidth=0.8, color=method_colors[i])
        ax.set_xlabel('2θ (°)', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(10, 80)
        ax.tick_params(axis='both', labelsize=11)

    # 最后一个子图：叠加所有类型
    ax = axes[5]
    lines = []
    for i, (key, color) in enumerate(zip(keys, method_colors)):
        line, = ax.plot(two_theta, results[key][0], alpha=0.7, color=color)
        lines.append(line)
    ax.set_xlabel('2θ (°)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title('All Overlaid', fontsize=13)
    ax.set_xlim(10, 80)
    ax.tick_params(axis='both', labelsize=11)

    # 在图的顶部添加统一图例
    fig.legend(lines, titles[:5], loc='upper center', ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        # 保存PNG
        full_path = os.path.join(output_dir, save_path)
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {full_path}")

        # 保存PDF
        pdf_path = full_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved to {pdf_path}")

        # 保存CSV
        csv_data = {'two_theta': two_theta}
        for key in keys:
            for j, spec in enumerate(results[key]):
                csv_data[f'{key}_sample{j+1}'] = spec
        df = pd.DataFrame(csv_data)
        csv_path = full_path.replace('.png', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")

    plt.show()


def create_example_cif(output_dir="si_augmentation_results"):
    """创建一个示例CIF文件 (Si diamond) - 与MD模块使用相同结构"""
    import os
    from ase.build import bulk
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.cif import CifWriter

    os.makedirs(output_dir, exist_ok=True)

    # 使用ASE创建Si diamond结构，与MD模块完全一致
    si_atoms = bulk('Si', 'diamond', a=5.43)
    structure = AseAtomsAdaptor.get_structure(si_atoms)

    # 保存为CIF
    cif_path = f'{output_dir}/Si_original.cif'
    CifWriter(structure).write_file(cif_path)
    return cif_path, output_dir


if __name__ == "__main__":
    import sys
    import os

    output_dir = "si_augmentation_results"

    # 使用命令行参数或示例CIF
    if len(sys.argv) > 1:
        cif_path = sys.argv[1]
        os.makedirs(output_dir, exist_ok=True)
    else:
        print("No CIF provided, creating example Si...")
        cif_path, output_dir = create_example_cif(output_dir)

    # 生成增广谱
    results, two_theta = generate_augmented_spectra(cif_path, n_augment=5)

    # 打印统计
    print(f"\nGenerated spectra:")
    for key, specs in results.items():
        print(f"  {key}: {len(specs)} spectra, shape {specs[0].shape}")

    # 保存增广谱数据为npy文件
    for key, specs in results.items():
        npy_path = os.path.join(output_dir, f'spectra_{key}.npy')
        np.save(npy_path, np.array(specs))
        print(f"  Saved {npy_path}")

    # 保存2theta轴
    np.save(os.path.join(output_dir, 'two_theta.npy'), two_theta)

    # 可视化
    plot_augmented_spectra(results, two_theta, output_dir, save_path='augmented_xrd.png')

    print(f"\n所有结果已保存到: {output_dir}/")
