# ElasticPanoramaDepthRefine

**单张全景图的各向异性弹性深度矫正系统**

[English](#english) | 中文

---

## 📖 项目简介

ElasticPanoramaDepthRefine 是一个用于单张全景图深度矫正的系统，通过各向异性弹性位移场在 log-depth 空间中进行优化，将稀疏但准确的参考点云（来自 COLMAP/LiDAR）与 DAP 预测的初始深度图进行对齐。

### 核心思想

> **"钉子绷橡皮筋"模型**：稀疏参考点作为"钉子"固定深度，各向异性弹性权重控制修正传播，在结构边缘和天空边界处自动中断传播。

### 主要特点

- ✅ **各向异性弹性传播**：修正只在结构内部传播，不会跨越楼宇边缘、天空边界或深度不连续处
- ✅ **Log-depth 空间优化**：所有深度计算在 log-depth 空间进行，保证数值稳定性
- ✅ **凸优化求解**：能量函数严格二次型，一次求解即可得到全局最优解
- ✅ **自动边缘检测**：基于深度梯度和边缘 mask 自动识别结构边界
- ✅ **天空区域保护**：天空区域不受深度修正影响，保持原始预测

---

## 🚀 快速开始

### 安装依赖

```bash
pip install numpy scipy opencv-python matplotlib pyyaml imageio
```

**要求**：
- Python 3.9+
- NumPy
- SciPy
- OpenCV-Python
- Matplotlib
- PyYAML
- ImageIO

### 基本使用

1. **准备配置文件** (`config.yaml`)：

```yaml
paths:
  depth_dap: /path/to/depth_dap.npy      # DAP 深度图（.npy 文件）
  rgb: /path/to/rgb.png                  # RGB 图像（可选）
  anchors: logs/anchor.npy                # Anchor 点文件
  output_dir: /path/to/output             # 输出目录

optimization:
  lambda_g: 1.0          # 梯度权重参数
  lambda_e: 5.0          # 边缘权重参数
  lambda_grad: 0.1       # 梯度保持项权重
  alpha_anchor: 10.0     # Anchor 约束权重
```

2. **运行深度矫正**：

```bash
python main.py --config config.yaml
```

3. **查看结果**：

- **最终输出**：`output_dir/point5_median.ply`（PLY 点云文件）
- **中间结果**：`logs/` 目录下
  - `depth_refined.npy`：矫正后的深度图
  - `depth_comparison.png`：深度对比可视化
  - `anchor_visualization.png`：Anchor 点分布可视化
  - `depth_change_log_space.png`：Log 空间深度变化热力图
  - `depth_change_linear_space.png`：真实空间深度变化热力图

---

## 📁 项目结构

```
ElasticPanoramaDepthRefine/
├── config.yaml                 # 配置文件
├── main.py                     # 主入口程序
├── data/
│   ├── __init__.py
│   └── io.py                   # 数据读取和坐标转换
├── geometry/
│   ├── __init__.py
│   └── sphere.py               # 球面邻接和梯度计算
├── graph/
│   ├── __init__.py
│   ├── weights.py              # 各向异性弹性权重
│   └── laplacian.py            # 稀疏 Laplacian 矩阵构建
├── solver/
│   ├── __init__.py
│   └── elastic_solver.py       # 线性系统求解器
├── utils/
│   ├── __init__.py
│   ├── masks.py                # Sky/edge mask 检测
│   ├── visualization.py        # 可视化工具
│   ├── pointcloud.py           # 点云生成（深度图 → PLY）
│   └── generate_anchors_from_ply.py  # Anchor 生成工具
├── logs/                       # 中间结果目录
├── test_step1_data_io.py       # 数据 I/O 测试脚本
├── README.md                   # 本文档
├── IMPLEMENTATION_LOG.md       # 实现记录
└── ANCHOR_SOURCE.md            # Anchor 数据来源说明
```

---

## 🔧 配置说明

### 数据路径配置

```yaml
paths:
  depth_dap: /path/to/depth_dap.npy      # DAP 深度图（必需）
  rgb: /path/to/rgb.png                  # RGB 图像（可选，用于可视化）
  anchors: logs/anchor.npy               # Anchor 点文件（必需）
  output_dir: /path/to/output             # 输出目录（必需）
  
  # Anchor 自动生成配置（当 anchor 文件不存在时使用）
  anchor_generation:
    fused_ply: /path/to/fused.ply        # COLMAP fused.ply 文件
    colmap_dir: /path/to/sparse/0        # COLMAP 重建目录
    camera_name: pano_camera12            # 相机名称
    sample_rate: 0.1                      # 采样率 [0, 1]
```

### 优化参数配置

```yaml
optimization:
  lambda_g: 1.0          # 梯度权重参数（越大，梯度差异大的边权重越小）
  lambda_e: 5.0          # 边缘权重参数（越大，边缘处权重越小）
  lambda_grad: 0.1       # 梯度保持项权重（保持 DAP 原始梯度）
  alpha_anchor: 10.0     # Anchor 约束权重（越大，anchor 越硬）
```

**参数调优建议**：
- `lambda_g`：控制梯度敏感性，默认 1.0
- `lambda_e`：控制边缘阻断强度，默认 5.0（边缘处权重接近 0）
- `lambda_grad`：控制梯度保持强度，默认 0.1（较小，允许适度修正）
- `alpha_anchor`：控制 anchor 约束强度，默认 10.0（较大，确保 anchor 处精确对齐）

### Anchor 过滤配置

```yaml
anchor_filter:
  max_depth: 100.0       # 最大深度阈值（米），距离球心大于此值的点将被剔除
```

### 边缘检测配置

```yaml
edge:
  use_rgb_edge: false              # 是否使用 RGB 边缘检测（已禁用）
  use_depth_edge: true             # 是否使用深度边缘检测
  depth_edge_threshold: 0.5        # 深度边缘阈值（log-depth 空间）
```

### 天空检测配置

```yaml
sky:
  enable: true                      # 是否启用 sky mask
  brightness_threshold: 0.9        # 天空亮度阈值（RGB，已禁用）
```

**注意**：实际使用基于深度的天空 mask（DAP 深度 >= 1.0 视为天空）。

### 求解器配置

```yaml
solver:
  method: cg            # 求解方法：'cg'（共轭梯度）或 'spsolve'（直接求解）
  max_iter: 500         # CG 最大迭代次数
  tol: 1e-4             # CG 收敛容差
```

**建议**：
- 大规模图像（如 4096×2048）使用 `cg` 方法
- 小规模图像可以使用 `spsolve` 直接求解

### 深度缩放配置

```yaml
depth_scale: 100.0      # DAP 深度图缩放因子（0-1 范围 → 米）
```

### 输出配置

```yaml
output:
  format: npy                       # 输出格式：'npy' 或 'png'
  save_visualization: true          # 是否保存可视化
  visualize_weight_terms: true      # 是否可视化权重项（梯度项和边缘项）
```

---

## 🧮 核心算法

### 1. 优化空间

所有深度相关量在 **log-depth 空间** 中定义：

$$
z_0(i) = \log D_0(i)
$$

$$
\Delta(i) \in \mathbb{R} \quad\text{（log-depth displacement）}
$$

最终深度：

$$
D(i) = \exp(z_0(i) + \Delta(i))
$$

### 2. 图结构

- **节点**：球面全景图的每个像素
- **邻接**：4-connected（经度方向周期 wrap）
- **边**：$(i, j)$ 表示相邻像素对

### 3. 各向异性弹性权重

对相邻像素 $(i, j)$，定义权重：

$$
w_{ij} = \exp\left(-\lambda_g |\nabla z_0(i) - \nabla z_0(j)|_2\right) \cdot \exp\left(-\lambda_e \cdot E(i,j)\right)
$$

其中：
- $\nabla z_0(i)$：log-depth 梯度
- $E(i,j) = 1$：跨越结构/天空/深度边缘
- $E(i,j) = 0$：同一连续表面

**含义**：
- 深度梯度变化大 → 弹性弱（权重小）
- 边缘处 → 弹性近似为 0（力不传导）

### 4. 能量函数

#### Anchor 项（log-depth）

$$
\mathcal{L}_{\text{anchor}} = \sum_k \alpha_k \left(z_0(i_k) + \Delta(i_k) - \log D^{\text{ref}}_k\right)^2
$$

#### 各向异性弹性项

$$
\mathcal{L}_{\text{elastic}} = \sum_{(i,j)} w_{ij} \left(\Delta(i) - \Delta(j)\right)^2
$$

#### 梯度保持项

$$
\mathcal{L}_{\text{grad}} = \sum_i |\nabla \Delta(i)|^2
$$

#### 总能量

$$
\mathcal{L} = \mathcal{L}_{\text{anchor}} + \lambda_{\text{elastic}} \mathcal{L}_{\text{elastic}} + \lambda_{\text{grad}} \mathcal{L}_{\text{grad}}
$$

**注意**：$\lambda_{\text{elastic}} = 1$（已包含在权重中），实际使用 $\lambda_{\text{grad}}$ 控制梯度保持项。

### 5. 线性系统求解

将能量写成线性系统：

$$
(\mathbf{L}_{\text{elastic}} + \lambda_{\text{grad}} \mathbf{L}_{\text{grad}} + \mathbf{A}_{\text{anchor}}) \boldsymbol{\Delta} = \mathbf{b}_{\text{anchor}}
$$

- **稀疏对称正定矩阵**
- 使用 **CG（共轭梯度）** 或 **Cholesky 直接求解**

---

## 📊 输入输出格式

### 输入文件

1. **DAP 深度图** (`depth_dap.npy`)
   - 格式：`float32`，形状 `(H, W)`
   - 范围：`[0, 1]`（需要乘以 `depth_scale` 转换为米）
   - 说明：DAP 模型预测的初始深度图

2. **RGB 图像** (`rgb.png`)（可选）
   - 格式：`uint8`，形状 `(H, W, 3)`
   - 用途：可视化、边缘检测（已禁用）

3. **Anchor 点** (`anchors.npy`)
   - 格式：`float32`，形状 `(K, 3)`
   - 内容：`(theta, phi, depth)`
     - `theta`：yaw 角 `[-π, π]`（方位角）
     - `phi`：pitch 角 `[-π/2, π/2]`（极角）
     - `depth`：到相机原点的欧式距离（米）
   - 说明：稀疏但准确的参考点云（来自 COLMAP/LiDAR）

### 输出文件

1. **矫正后的深度图** (`logs/depth_refined.npy`)
   - 格式：`float32`，形状 `(H, W)`
   - 单位：米

2. **PLY 点云** (`output_dir/point5_median.ply`)
   - 格式：Binary PLY（小端序）
   - 内容：3D 点云（包含 RGB 颜色信息）
   - 约定：`colmap_util`（与 fused_remap.py 一致）

3. **可视化图像**（`logs/` 目录）
   - `depth_comparison.png`：原始 vs 矫正深度对比
   - `anchor_visualization.png`：Anchor 点分布可视化
   - `depth_change_log_space.png`：Log 空间深度变化热力图
   - `depth_change_linear_space.png`：真实空间深度变化热力图
   - `weight_terms_visualization.png`：权重项可视化（可选）

---

## 🔍 Anchor 数据来源

**重要**：Anchor 点不是直接从 `fused.ply` 读取的，而是经过以下流程：

```
1. fused.ply (稠密点云，世界坐标)
   ↓
2. project_colmap_points_to_pano() 
   - 投影到等轴柱状图
   - 应用 depth_min/depth_max 过滤
   - Z-buffer 选择最近点
   ↓
3. 参考深度图 (D_ref.npy) + Anchor Mask (M_ref)
   ↓
4. generate_anchors_from_ref_depth()
   - 从参考深度图中采样有效像素
   - 转换为 (theta, phi, depth) 格式
   ↓
5. anchor.npy 文件
```

详细说明请参考 `ANCHOR_SOURCE.md`。

### 自动生成 Anchor

如果 `anchor.npy` 文件不存在，程序会自动从 `fused.ply` 生成（需要配置 `anchor_generation` 部分）：

```yaml
paths:
  anchor_generation:
    fused_ply: /path/to/fused.ply
    colmap_dir: /path/to/sparse/0
    camera_name: pano_camera12
    sample_rate: 0.1
```

---

## 🧪 测试

### 数据 I/O 测试

```bash
python test_step1_data_io.py
```

测试内容：
- 深度图加载
- RGB 图像加载（可选）
- Anchor 点加载
- 坐标转换（像素 ↔ 球面坐标）
- Anchor 转换（球面坐标 → 像素索引）

---

## ⚠️ 注意事项

### 1. Log-depth 空间

**必须在 log-depth 空间做的**：
- 深度梯度计算
- Anchor 误差计算
- 位移变量定义

**不要在 log 空间做的**：
- Sky/edge mask（基于原始深度值）
- 像素邻接拓扑

### 2. 坐标约定

- 默认使用 `colmap_util` 约定
- 确保与数据源一致（参考 `fused_remap.py` 和 `spherical_camera.py`）

### 3. 内存使用

- 大规模图像（如 4096×2048）会生成大型稀疏矩阵
- 建议使用 CG 方法而非直接求解
- 矩阵大小：$N \times N$，其中 $N = H \times W$

### 4. 参数调优

- `lambda_g`：控制梯度敏感性（越大，梯度差异大的边权重越小）
- `lambda_e`：控制边缘阻断（越大，边缘处权重越小）
- `alpha_anchor`：控制 anchor 约束强度（越大，anchor 越硬）

---

## 📚 相关文档

- `IMPLEMENTATION_LOG.md`：详细的实现记录和模块说明
- `ANCHOR_SOURCE.md`：Anchor 数据来源和生成流程说明
- `CURSOR.md`：项目设计文档和实现要求

---

## 🎯 方法解释（一句话版）

> **参考点把 log-depth 钉死，各向异性 Laplacian 让修正只在结构内部像橡皮筋一样传播，在边缘与天空处自然断裂。**

---

## 📝 更新日志

### 最新更新

- ✅ 完成项目结构创建和所有核心模块实现
- ✅ 添加点云生成功能（深度图 → PLY）
- ✅ 添加深度过滤功能（`anchor_filter.max_depth`）
- ✅ 添加 anchor 可视化功能
- ✅ 支持自动生成 anchor（从 fused.ply）
- ✅ 完整流程测试通过

详细更新记录请参考 `IMPLEMENTATION_LOG.md`。

---

## 📄 许可证

本项目遵循 MIT 许可证。

---

## 🙏 致谢

本项目参考了以下开源项目：
- DAP (Depth Any Panoramas)
- COLMAP
- 相关深度估计和点云处理工具

---

<a name="english"></a>
# ElasticPanoramaDepthRefine

**Anisotropic Elastic Depth Refinement for Single Panorama**

English | [中文](#-项目简介)

---

## 📖 Overview

ElasticPanoramaDepthRefine is a depth refinement system for single panoramas that optimizes an anisotropic elastic displacement field in log-depth space, aligning sparse but accurate reference point clouds (from COLMAP/LiDAR) with initial depth maps predicted by DAP.

### Core Idea

> **"Nails and Rubber Bands" Model**: Sparse reference points act as "nails" to fix depth, while anisotropic elastic weights control correction propagation, automatically stopping at structural edges and sky boundaries.

### Key Features

- ✅ **Anisotropic Elastic Propagation**: Corrections propagate only within structures, not across building edges, sky boundaries, or depth discontinuities
- ✅ **Log-depth Space Optimization**: All depth calculations are performed in log-depth space for numerical stability
- ✅ **Convex Optimization**: Strictly quadratic energy function, global optimum in one solve
- ✅ **Automatic Edge Detection**: Automatically identifies structural boundaries based on depth gradients and edge masks
- ✅ **Sky Region Protection**: Sky regions remain unaffected by depth corrections, preserving original predictions

---

## 🚀 Quick Start

### Installation

```bash
pip install numpy scipy opencv-python matplotlib pyyaml imageio
```

**Requirements**:
- Python 3.9+
- NumPy, SciPy, OpenCV-Python, Matplotlib, PyYAML, ImageIO

### Basic Usage

1. **Prepare configuration** (`config.yaml`)

2. **Run depth refinement**:
```bash
python main.py --config config.yaml
```

3. **Check results**:
- **Final output**: `output_dir/point5_median.ply` (PLY point cloud)
- **Intermediate results**: `logs/` directory

---

## 🔧 Configuration

See the Chinese section above for detailed configuration options.

---

## 🧮 Core Algorithm

See the Chinese section above for mathematical details.

---

## 📊 Input/Output Formats

### Input Files

1. **DAP Depth Map** (`depth_dap.npy`)
   - Format: `float32`, shape `(H, W)`
   - Range: `[0, 1]` (multiply by `depth_scale` to convert to meters)

2. **RGB Image** (`rgb.png`) (optional)
   - Format: `uint8`, shape `(H, W, 3)`

3. **Anchor Points** (`anchors.npy`)
   - Format: `float32`, shape `(K, 3)`
   - Content: `(theta, phi, depth)`

### Output Files

1. **Refined Depth Map** (`logs/depth_refined.npy`)
2. **PLY Point Cloud** (`output_dir/point5_median.ply`)
3. **Visualization Images** (`logs/` directory)

---

## ⚠️ Important Notes

1. **Log-depth Space**: All depth-related calculations must be in log-depth space
2. **Coordinate Convention**: Default `colmap_util` convention
3. **Memory Usage**: Large images generate large sparse matrices; use CG method
4. **Parameter Tuning**: See configuration section for parameter tuning guidelines

---

## 📚 Related Documentation

- `IMPLEMENTATION_LOG.md`: Detailed implementation log and module descriptions
- `ANCHOR_SOURCE.md`: Anchor data source and generation pipeline
- `CURSOR.md`: Project design document and implementation requirements

---

## 🎯 Method Explanation (One Sentence)

> **Reference points nail log-depth in place, anisotropic Laplacian propagates corrections like rubber bands only within structures, naturally breaking at edges and sky regions.**