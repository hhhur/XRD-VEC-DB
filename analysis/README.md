# Si XRD 数据增强分析

本文件夹包含基于Si (金刚石结构) 的XRD数据增强实验结果。

## 结构信息

| 参数 | 值 |
|------|-----|
| 材料 | Si (硅) |
| 结构类型 | Diamond (金刚石) |
| 空间群 | Fd-3m (No. 227) |
| 晶格常数 | a = 5.43 Å |
| X射线波长 | Cu Kα (1.5406 Å) |
| 2θ范围 | 10° - 80° |

## 文件结构

```
analysis/
├── README.md                    # 本说明文件
├── plots/                       # 所有对比图 (PNG + PDF)
│   ├── augmented_xrd.png/pdf    # 参数化增强对比图
│   ├── si_xrd_md_comparison.png/pdf # MD增强4子图对比
│   └── si_xrd_overlay.png/pdf   # MD增强叠加对比图
├── structures/                  # CIF结构文件 + ASE可视化图
│   ├── Si_original.cif          # MD模拟原始结构 (2×2×2超胞, 16原子)
│   ├── Si_original_structure.png # 原始结构可视化
│   ├── Si_parametric_original.cif # 参数化增强原始结构 (单胞, 2原子)
│   ├── Si_NVT-Langevin_snapshot_*.cif    # NVT-Langevin采样结构
│   ├── Si_NVT-Langevin_snapshot_*.png    # NVT-Langevin结构可视化
│   ├── Si_NVT-Berendsen_snapshot_*.cif   # NVT-Berendsen采样结构
│   ├── Si_NVT-Berendsen_snapshot_*.png   # NVT-Berendsen结构可视化
│   ├── Si_NPT-Berendsen_snapshot_*.cif   # NPT-Berendsen采样结构
│   └── Si_NPT-Berendsen_snapshot_*.png   # NPT-Berendsen结构可视化
└── data/                        # 增强谱图数据 (npy + csv格式)
    ├── two_theta.npy            # 2θ轴数据 (4501点)
    ├── spectra_clean.npy        # 干净谱 (仅噪声)
    ├── spectra_shifted.npy      # 均匀位移增强
    ├── spectra_broadened.npy    # 峰展宽增强
    ├── spectra_textured.npy     # 织构增强
    ├── spectra_mixed.npy        # 混合增强
    ├── augmented_xrd.csv        # 参数化增强谱图CSV
    └── si_xrd_md_data.csv       # MD增强谱图CSV
```

## 增强方法对比

### 1. 参数化增强 (Parametric Augmentation)

快速、经验性的谱图变换，毫秒级处理速度。

| 增强类型 | 物理含义 | 参数范围 |
|----------|----------|----------|
| Clean | 仅添加高斯噪声 | σ = 0.25 |
| Shifted | 均匀峰位移 (样品高度误差) | ±0.5° |
| Broadened | 峰展宽 (Scherrer效应) | D = 5-50 nm |
| Textured | 织构效应 (择优取向) | τ = 0.6 |
| Mixed | 以上所有效果组合 | - |

### 2. MD增强 (Molecular Dynamics Augmentation)

基于物理的热扰动采样，使用CHGNet机器学习势函数。

| 系综 | 温度 | 压力 | 特点 |
|------|------|------|------|
| Original | 0 K | - | 理想晶体结构 |
| NVT-Langevin | 300 K | - | 随机热浴耦合 |
| NVT-Berendsen | 300 K | - | 弱耦合恒温 |
| NPT-Berendsen | 300 K | 1 atm | 恒温恒压，含热膨胀 |

**MD模拟参数:**
- 时间步长: 1.0 fs
- 总步数: 400
- 采样间隔: 50步
- 平衡期: 100步
- 每个系综采样: 5个构型

## 图片说明

### augmented_xrd.png
参数化增强效果对比，展示5种增强方法对Si XRD谱图的影响。

### si_xrd_md_comparison.png
MD增强4子图对比，分别展示原始结构和3种系综的XRD谱图。

### si_xrd_overlay.png
MD增强叠加对比图，将4种谱图叠加显示便于比较峰位和强度变化。

## 数据加载示例

```python
import numpy as np

# 加载2θ轴
two_theta = np.load('data/two_theta.npy')

# 加载增强谱图
spectra_mixed = np.load('data/spectra_mixed.npy')
print(f"Shape: {spectra_mixed.shape}")  # (5, 4501)
```

## 参考文献

1. Szymanski et al., "Probabilistic Deep Learning Approach to Automate the Interpretation of Multi-phase Diffraction Spectra", Chemistry of Materials, 2021.
2. Deng et al., "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling", Nature Machine Intelligence, 2023.
