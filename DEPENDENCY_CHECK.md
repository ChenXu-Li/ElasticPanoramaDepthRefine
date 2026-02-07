# 依赖检查和精简建议

## 外部依赖分析

### 必需的外部库（标准依赖）
- ✅ `numpy` - 数值计算
- ✅ `scipy` - 稀疏矩阵和线性求解器
- ✅ `matplotlib` - 可视化
- ✅ `opencv-python` (cv2) - 边缘检测
- ✅ `imageio` - 图像读写
- ✅ `pyyaml` - 配置文件解析
- ✅ `pycolmap` - COLMAP 重建数据读取（仅用于 anchor 生成）
- ✅ `plyfile` - PLY 点云文件读写（仅用于 anchor 生成）

### 项目独立性
✅ **项目完全独立**，不依赖其他项目的代码：
- 所有功能都在本项目内实现
- 注释中提到的其他项目（`fused_remap.py`, `spherical_camera.py`, `PanoramaDepthRefine`）只是用于说明坐标约定，不是实际依赖
- 可以独立运行，只需要安装上述标准库

---

## 可精简的部分

### 1. 未使用的函数/文件

#### ❌ `data/io.py` 中的 `load_anchors` 函数
- **状态**：已导入但不再使用
- **原因**：`main.py` 现在每次都重新生成 anchor，不再加载已有文件
- **建议**：可以删除或保留作为工具函数（如果用户需要手动加载 anchor）

#### ✅ `utils/generate_anchors.py` 文件
- **状态**：已删除
- **原因**：功能与 `generate_anchors_from_ply.py` 中的函数重复
- **处理**：已删除，测试文件已更新

### 2. 注释中的外部项目引用

以下注释可以简化，去掉对其他项目的具体引用：

#### `data/io.py`
- 第 110-111 行：提到 `fused_remap.py` 和 `spherical_camera.py`
- 第 160-161 行：同上
- **建议**：改为"与 COLMAP 工具链约定一致"

#### `config.yaml`
- 第 29 行：提到 `fused_remap.py`
- **建议**：改为"与 COLMAP 工具链约定一致"

#### `README.md` 和 `IMPLEMENTATION_LOG.md`
- 多处提到 `fused_remap.py` 和 `spherical_camera.py`
- **建议**：改为通用描述，不引用具体文件

#### `ANCHOR_SOURCE.md`
- 第 39 行：提到 `PanoramaDepthRefine/src/projection/project_points.py`
- **建议**：说明这是参考实现，本项目已独立实现

### 3. 重复的代码

#### `generate_anchors_from_ref_depth` 函数
- **位置1**：`utils/generate_anchors.py` (接受文件路径)
- **位置2**：`utils/generate_anchors_from_ply.py` (接受 numpy 数组)
- **建议**：保留 `generate_anchors_from_ply.py` 中的版本（更灵活），删除 `generate_anchors.py`

### 4. 可选的精简

#### `test_step1_data_io.py`
- 如果不再需要测试数据 I/O，可以删除
- 或者更新测试，移除对 `generate_anchors.py` 的依赖

#### `geometry/sphere.py` 中的 `get_4_neighbors`
- 在 `generate_anchors_from_ply.py` 中被导入但未使用
- **检查**：确认是否真的未使用

---

## 精简建议总结

### 高优先级（已完成）
1. ✅ **删除 `utils/generate_anchors.py`** - 已完成
2. ✅ **简化注释**：去掉对 `fused_remap.py` 和 `spherical_camera.py` 的具体引用 - 已完成
3. ✅ **检查并删除未使用的导入**：如 `load_anchors` 在 main.py 中，`get_4_neighbors` 在 generate_anchors_from_ply.py 中 - 已完成

### 中优先级（可选）
4. ⚠️ **更新 `test_step1_data_io.py`**：移除对已删除文件的依赖
5. ⚠️ **检查 `geometry/sphere.py` 的导入**：确认 `get_4_neighbors` 在 `generate_anchors_from_ply.py` 中是否使用

### 低优先级（保留）
6. ℹ️ **保留 `load_anchors`**：作为工具函数，可能对用户有用
7. ℹ️ **保留文档中的项目引用**：如果有助于理解坐标约定

---

## 依赖安装清单

创建 `requirements.txt` 文件：

```txt
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
opencv-python>=4.5.0
imageio>=2.9.0
pyyaml>=5.4.0
pycolmap>=0.4.0
plyfile>=0.7.0
```

---

## 独立性验证

✅ **项目完全独立**：
- 所有核心功能都在本项目内实现
- 不依赖其他项目的源代码
- 只需要标准 Python 库和上述第三方库
- 可以独立运行和部署
