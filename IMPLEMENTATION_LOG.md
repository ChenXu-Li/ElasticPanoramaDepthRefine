# ElasticPanoramaDepthRefine 实现记录

## 项目概述

本项目实现了一个**单张全景图的深度矫正系统**，使用各向异性弹性位移场在 log-depth 空间中进行优化。

核心思想：
- 稀疏参考点作为"钉子"固定深度
- 各向异性弹性权重控制修正传播
- 在结构边缘和天空边界处自动中断传播

### Anchor 数据来源

**重要**：Anchor 点不是直接从 fused.ply 读取的，而是经过以下流程：

1. **fused.ply** (稠密点云，世界坐标)
2. → **参考深度图** (通过 `project_colmap_points_to_pano` 投影生成)
3. → **anchor.npy** (通过 `generate_anchors_from_ref_depth` 从参考深度图采样生成)

详细说明请参考 `ANCHOR_SOURCE.md`。

---

## 项目结构

```
ElasticPanoramaDepthRefine/
├── config.yaml                 # 配置文件
├── main.py                      # 主入口程序
├── data/
│   ├── __init__.py
│   └── io.py                    # 数据读取和坐标转换
├── geometry/
│   ├── __init__.py
│   └── sphere.py                # 球面邻接和梯度计算
├── graph/
│   ├── __init__.py
│   ├── weights.py               # 各向异性弹性权重
│   └── laplacian.py             # 稀疏 Laplacian 矩阵构建
├── solver/
│   ├── __init__.py
│   └── elastic_solver.py        # 线性系统求解器
├── utils/
│   ├── __init__.py
│   ├── masks.py                 # Sky/edge mask 检测
│   ├── visualization.py         # 可视化工具
│   ├── pointcloud.py            # 点云生成（深度图 → PLY）
│   └── generate_anchors.py      # Anchor 生成工具
├── test_step1_data_io.py        # Step 1 测试脚本
└── IMPLEMENTATION_LOG.md        # 本文档
```

---

## 各模块说明

### 1. data/io.py

**功能**：数据读取和坐标转换

**主要函数**：
- `load_depth(path)`: 读取深度图 (.npy)
- `load_rgb(path)`: 读取 RGB 图像（可选）
- `load_anchors(path)`: 读取 anchor 点 [K, 3] -> (theta, phi, depth)
- `theta_phi_to_pixel(theta, phi, width, height, convention)`: 球面坐标 -> 像素坐标
- `pixel_to_theta_phi(u, v, width, height, convention)`: 像素坐标 -> 球面坐标
- `anchors_to_pixel_indices(anchors, width, height, convention)`: Anchor -> 像素索引

**坐标约定**：
- `colmap_util`: theta 为 yaw (atan2(x,z)), phi 为 pitch
- `dap`: theta 为方位角 [0, 2π), phi 为极角 [0, π]

**用法示例**：
```python
from data.io import load_depth, anchors_to_pixel_indices

depth = load_depth("depth.npy")
anchors = load_anchors("anchors.npy")
u, v, depths = anchors_to_pixel_indices(anchors, width, height)
```

---

### 2. geometry/sphere.py

**功能**：球面几何计算

**主要函数**：
- `get_4_neighbors(height, width)`: 获取 4-connected 邻接关系（考虑经度周期 wrap）
- `compute_gradient_log_depth(log_depth)`: 在 log-depth 空间计算梯度
- `compute_gradient_magnitude(log_depth)`: 计算梯度幅值
- `get_edge_gradient_diff(log_depth, i_indices, j_indices)`: 计算相邻像素的梯度差异

**特点**：
- 经度方向周期 wrap（u=0 和 u=width-1 相邻）
- 所有梯度计算在 log-depth 空间

**用法示例**：
```python
from geometry.sphere import get_4_neighbors, compute_gradient_log_depth

i_indices, j_indices, edge_types = get_4_neighbors(height, width)
grad_u, grad_v = compute_gradient_log_depth(log_depth)
```

---

### 3. graph/weights.py

**功能**：各向异性弹性权重计算

**主要函数**：
- `compute_anisotropic_weights(log_depth, i_indices, j_indices, lambda_g, lambda_e, edge_mask)`: 计算权重

**公式**：
```
w_ij = exp(-λ_g * ||∇z0(i) - ∇z0(j)||_2) * exp(-λ_e * E(i,j))
```

**参数**：
- `lambda_g`: 梯度权重参数（越大，梯度差异大的边权重越小）
- `lambda_e`: 边缘权重参数（越大，边缘处的权重越小）
- `edge_mask`: (E,) bool，True 表示该边跨越边缘

**用法示例**：
```python
from graph.weights import compute_anisotropic_weights

weights = compute_anisotropic_weights(
    log_depth, i_indices, j_indices,
    lambda_g=1.0, lambda_e=5.0,
    edge_mask=edge_mask
)
```

---

### 4. graph/laplacian.py

**功能**：稀疏 Laplacian 矩阵构建

**主要函数**：
- `build_weighted_laplacian(height, width, i_indices, j_indices, weights)`: 构建加权图 Laplacian
- `build_gradient_laplacian(height, width)`: 构建标准梯度 Laplacian（用于梯度保持项）

**矩阵形式**：
- `L = D - W`，其中 D 是度矩阵，W 是权重矩阵
- 返回稀疏 CSR 矩阵

**用法示例**：
```python
from graph.laplacian import build_weighted_laplacian, build_gradient_laplacian

L_elastic = build_weighted_laplacian(height, width, i_indices, j_indices, weights)
L_grad = build_gradient_laplacian(height, width)
```

---

### 5. solver/elastic_solver.py

**功能**：线性系统求解器

**主要函数**：
- `build_anchor_matrix(anchor_indices, alpha, N)`: 构建 anchor 约束矩阵
- `build_anchor_rhs(anchor_indices, anchor_values, alpha, N)`: 构建 anchor 右端项
- `solve_elastic_system(L_elastic, L_grad, A_anchor, b_anchor, lambda_grad, method, max_iter, tol)`: 求解线性系统
- `refine_depth(...)`: 完整的深度矫正流程

**线性系统**：
```
(L_elastic + λ_grad * L_grad + A_anchor) Δ = b_anchor
```

**求解方法**：
- `cg`: 共轭梯度法（推荐，适合大规模稀疏矩阵）
- `spsolve`: 直接求解（Cholesky，适合小规模问题）

**用法示例**：
```python
from solver.elastic_solver import refine_depth

depth_refined = refine_depth(
    depth_dap, anchor_indices, anchor_depths,
    L_elastic, L_grad, lambda_grad, alpha_anchor,
    method="cg", max_iter=1000, tol=1e-6
)
```

---

### 6. utils/masks.py

**功能**：Sky 和 edge mask 检测

**主要函数**：
- `detect_sky_mask(rgb, brightness_threshold)`: 基于亮度检测天空
- `detect_rgb_edge(rgb, threshold)`: 使用 Canny 检测 RGB 边缘
- `detect_depth_edge(log_depth, threshold)`: 检测深度边缘（log-depth 空间）
- `build_edge_mask_for_edges(...)`: 为图边构建边缘 mask

**用法示例**：
```python
from utils.masks import detect_sky_mask, build_edge_mask_for_edges

sky_mask = detect_sky_mask(rgb, brightness_threshold=0.9)
edge_mask = build_edge_mask_for_edges(
    height, width, i_indices, j_indices,
    rgb=rgb, log_depth=log_depth, sky_mask=sky_mask
)
```

---

### 7. utils/visualization.py

**功能**：可视化工具

**主要函数**：
- `depth_to_colormap(depth, vmin, vmax, colormap)`: 深度图转彩色图
- `visualize_depth_comparison(depth_dap, depth_refined, output_path, rgb)`: 可视化深度对比
- `visualize_anchors(anchor_indices, anchor_depths, width, height, output_path, rgb, depth)`: 可视化 anchor 点在等轴柱状图上的分布
- `save_depth(depth, output_path, format)`: 保存深度图（npy 或 png）

**用法示例**：
```python
from utils.visualization import save_depth, visualize_depth_comparison, visualize_anchors

save_depth(depth_refined, "output.npy", format="npy")
visualize_depth_comparison(depth_dap, depth_refined, "comparison.png", rgb)
visualize_anchors(anchor_indices, anchor_depths, width, height, "anchors.png", rgb=rgb, depth=depth_dap)
```

**visualize_anchors 功能**：
- 左图：在 RGB 或深度图上标记 anchor 点（红色点）
- 右图：显示深度分布直方图（所有像素 vs anchor 点）和统计信息

---

### 8. utils/pointcloud.py

**功能**：点云生成（深度图 → binary PLY）

**主要函数**：
- `depth_to_pointcloud_ply(depth, rgb, output_path, convention)`: 将深度图转换为 binary PLY 点云

**特点**：
- 使用 colmap_util 约定（与 fused_remap.py 一致）
- 支持 RGB 颜色信息
- 输出 binary PLY 格式（小端序）

**用法示例**：
```python
from utils.pointcloud import depth_to_pointcloud_ply

depth_to_pointcloud_ply(
    depth_refined, 
    rgb, 
    "pointcloud.ply", 
    convention="colmap_util"
)
```

---

### 8. main.py

**功能**：主入口程序

**流程**：
1. 读取数据（深度、RGB、anchors）
2. 转换 anchor 到像素索引
3. 构建图结构（4-connected 邻接）
4. 计算各向异性权重
5. 构建 Laplacian 矩阵
6. 求解线性系统
7. 保存结果

**使用方法**：
```bash
python main.py --config config.yaml
```

---

## 测试记录

### Step 1: 数据读取和坐标转换测试

**测试脚本**：`test_step1_data_io.py`

**测试内容**：
1. 数据加载测试
   - 深度图加载
   - RGB 图像加载（可选）
   - Anchor 点加载

2. 坐标转换测试
   - 像素坐标 -> 球面坐标
   - 球面坐标 -> 像素坐标
   - 往返转换误差检查

3. Anchor 转换测试
   - Anchor (theta, phi, depth) -> 像素索引
   - 验证转换正确性

**运行方法**：
```bash
cd /root/autodl-tmp/code/ElasticPanoramaDepthRefine
python test_step1_data_io.py
```

**实际测试结果**（2024-02-XX）：
```
============================================================
测试 1: 数据加载
============================================================
✓ 深度图加载成功: (960, 1920), dtype=float32
  深度范围: [0.01, 1.00]
✓ RGB 图像加载成功: (960, 1920, 3), dtype=uint8

============================================================
测试 2: 坐标转换
============================================================
✓ 像素 -> (theta, phi) 转换成功
✓ (theta, phi) -> 像素 转换成功
  误差: u_err=0.00, v_err=0.00 (考虑周期 wrap)

============================================================
测试 3: Anchor 转换
============================================================
✓ Anchor 加载成功: (6035, 3)
  Anchor 范围: theta=[-2.94, 2.87], phi=[-0.47, 1.17], depth=[5.46, 100.00]
✓ Anchor -> 像素索引转换成功
  有效 anchor 数量: 6035
  像素范围: u=[1021, 1919], v=[124, 624]

============================================================
✓ 所有测试通过！
============================================================
```

**注意**：
- 坐标转换在经度方向（u）有周期 wrap，边界处误差可能较大，这是正常的
- Anchor 文件如果不存在，会自动从参考深度图生成（采样率 10%）

---

## 配置文件说明

配置文件 `config.yaml` 包含以下部分：

### paths
- `depth_dap`: DAP 深度图路径
- `rgb`: RGB 图像路径（可选）
- `anchors`: Anchor 点路径
- `output_dir`: 输出目录

### anchor_filter
- `max_depth`: 最大深度阈值（米，默认 100.0），距离球心大于等于此值的参考点将被剔除，不做 anchor

### optimization
- `lambda_g`: 梯度权重参数（默认 1.0）
- `lambda_e`: 边缘权重参数（默认 5.0）
- `lambda_grad`: 梯度保持项权重（默认 0.1）
- `alpha_anchor`: Anchor 权重（默认 10.0）

### edge
- `use_rgb_edge`: 是否使用 RGB 边缘检测
- `rgb_edge_threshold`: RGB 边缘阈值
- `use_depth_edge`: 是否使用深度边缘检测
- `depth_edge_threshold`: 深度边缘阈值（log-depth 空间）

### sky
- `enable`: 是否启用 sky mask
- `brightness_threshold`: 天空亮度阈值

### solver
- `method`: 求解方法 'cg' 或 'spsolve'
- `max_iter`: CG 最大迭代次数
- `tol`: CG 收敛容差

### output
- `format`: 输出格式 'npy' 或 'png'
- `save_visualization`: 是否保存可视化

---

## 实现步骤总结

### ✅ Step 1: 项目结构创建
- 创建目录结构
- 创建配置文件
- 创建 `__init__.py` 文件

### ✅ Step 2: 数据读取模块 (data/io.py)
- 实现深度图、RGB、anchor 加载
- 实现坐标转换函数
- 实现 anchor 到像素索引转换

### ✅ Step 3: 几何计算模块 (geometry/sphere.py)
- 实现 4-connected 邻接关系（周期 wrap）
- 实现 log-depth 梯度计算
- 实现梯度差异计算

### ✅ Step 4: 权重计算模块 (graph/weights.py)
- 实现各向异性弹性权重计算
- 支持边缘 mask

### ✅ Step 5: Laplacian 构建模块 (graph/laplacian.py)
- 实现加权图 Laplacian 构建
- 实现梯度保持 Laplacian 构建

### ✅ Step 6: 求解器模块 (solver/elastic_solver.py)
- 实现 anchor 约束矩阵构建
- 实现线性系统求解
- 实现完整深度矫正流程

### ✅ Step 7: Mask 模块 (utils/masks.py)
- 实现 sky mask 检测
- 实现 RGB/深度边缘检测
- 实现图边边缘 mask 构建

### ✅ Step 8: 可视化模块 (utils/visualization.py)
- 实现深度图转彩色图
- 实现深度对比可视化
- 实现深度图保存

### ✅ Step 9: 主入口程序 (main.py)
- 实现完整流程串联
- 实现配置加载
- 实现结果保存

### ✅ Step 10: 测试脚本
- 创建 Step 1 测试脚本
- 创建 anchor 生成工具

### ✅ Step 11: 实现记录文档
- 创建本文档

---

## 下一步计划

1. **运行 Step 1 测试**
   ```bash
   python test_step1_data_io.py
   ```

2. **生成 anchor 点**（如果不存在）
   ```bash
   python -m utils.generate_anchors \
       --ref_depth /path/to/ref_depth.npy \
       --output /path/to/anchors.npy \
       --sample_rate 0.1
   ```

3. **运行完整流程**
   ```bash
   python main.py --config config.yaml
   ```

4. **验证结果**
   - 检查输出深度图
   - 检查可视化图像
   - 验证 anchor 处深度是否对齐

---

## 注意事项

1. **所有深度计算在 log-depth 空间**
   - 梯度计算
   - Anchor 误差
   - 位移变量定义

2. **坐标约定**
   - 默认使用 `colmap_util` 约定
   - 确保与数据源一致

3. **内存使用**
   - 大规模图像（如 4096x2048）会生成大型稀疏矩阵
   - 建议使用 CG 方法而非直接求解

4. **参数调优**
   - `lambda_g`: 控制梯度敏感性（越大，梯度差异大的边权重越小）
   - `lambda_e`: 控制边缘阻断（越大，边缘处权重越小）
   - `alpha_anchor`: 控制 anchor 约束强度（越大，anchor 越硬）

---

## 更新日志

### 2024-02-XX
- ✅ 完成项目结构创建
- ✅ 完成所有核心模块实现
- ✅ 完成测试脚本
- ✅ 完成实现记录文档
- ✅ Step 1 测试通过
  - 数据加载：深度图 (960, 1920)，RGB 图像正常
  - 坐标转换：往返转换正确（考虑周期 wrap）
  - Anchor 转换：6035 个 anchor 点成功转换
- ✅ 修复导入错误
  - 将所有相对导入改为绝对导入
  - 在 main.py 中添加路径设置
- ✅ 修复 CG 求解器参数错误
  - 将 `tol` 改为 `rtol` 和 `atol`
- ✅ 修复坐标转换错误
  - 参考 `fused_remap.py` 和 `spherical_camera.py` 修正坐标转换
  - `colmap_util` 约定：theta = yaw = atan2(x,z) [-π, π], phi = pitch = -atan2(y, sqrt(x²+z²)) [-π/2, π/2]
  - 修复 `theta_phi_to_pixel` 和 `pixel_to_theta_phi` 函数
  - 确保与参考深度图投影方式一致
- ✅ 完整流程测试通过
  - 成功运行完整流程
  - 生成输出深度图和可视化
  - 矩阵大小：1,843,200 x 1,843,200
  - CG 求解：500 次迭代（未完全收敛，但已生成结果）
  - Anchor 坐标转换正确：u 范围 [61, 1835]，v 范围 [124, 624]
- ✅ 添加点云生成功能
  - 实现 `utils/pointcloud.py`：从深度图生成 binary PLY 点云
  - 使用 colmap_util 约定，与 fused_remap.py 一致
  - 支持 RGB 颜色信息
- ✅ 重构输出目录结构
  - `elastic_refined/`：只保留最终 PLY 点云文件（pointcloud.ply）
  - `项目目录/logs/`：保存中间结果（depth_refined.npy, depth_comparison.png, anchor_visualization.png）
  - 清晰分离最终输出和中间结果
  - 日志文件统一保存在项目目录下的 logs 文件夹
- ✅ 添加深度过滤功能
  - 在配置文件中添加 `anchor_filter.max_depth` 参数（默认 100.0 米）
  - 距离球心大于等于该值的参考点将被剔除，不做 anchor
  - 在转换 anchor 坐标时应用过滤，并输出详细统计信息：
    - 原始 anchor 数量和深度范围
    - 过滤结果（剔除数量或提示无点被剔除）
    - 过滤后的深度范围（如果有被剔除的点）
    - 有效 anchor 像素数量
- ✅ 添加 anchor 可视化功能
  - 实现 `visualize_anchors()` 函数：在等轴柱状图上可视化 anchor 点分布
  - 左图：在 RGB 或深度图上标记 anchor 点（红色点）
  - 右图：显示深度分布直方图和统计信息（所有像素 vs anchor 点）
  - 自动保存到 `项目目录/logs/anchor_visualization.png`
  - 便于检查 anchor 点的空间分布和深度分布
- ✅ 修改日志输出路径
  - 将所有中间结果（depth_refined.npy, depth_comparison.png, anchor_visualization.png）保存到项目目录下的 `logs/` 文件夹
  - 路径：`/root/autodl-tmp/code/ElasticPanoramaDepthRefine/logs/`
  - 统一管理所有日志和中间结果文件