# Anchor 数据来源说明

## Anchor 数据流程

Anchor 点**不是直接从 fused.ply 读取的**，而是经过以下流程：

### 完整数据流程

```
1. fused.ply (稠密点云，世界坐标)
   ↓
2. project_colmap_points_to_pano() 
   - 读取 fused.ply 点云（世界坐标）
   - 通过 cam_from_world 变换到相机坐标
   - 投影到等轴柱状图（equirectangular）
   - 应用 depth_min/depth_max 过滤
   - Z-buffer 选择最近点
   ↓
3. 参考深度图 (D_ref.npy) + Anchor Mask (M_ref)
   - D_ref: (H, W) float32，参考深度图
   - M_ref: (H, W) bool，anchor mask（有效投影点）
   ↓
4. generate_anchors_from_ref_depth()
   - 从参考深度图中采样有效像素
   - 转换为 (theta, phi, depth) 格式
   - 可选采样率（默认 10%）
   ↓
5. anchor.npy 文件
   - 格式: [K, 3] -> (theta, phi, depth)
   - theta: yaw 角 [-π, π]
   - phi: pitch 角 [-π/2, π/2]
   - depth: 到相机原点的欧式距离（米）
```

## 关键代码位置

### 1. 参考深度图生成（PanoramaDepthRefine）

**文件**: `PanoramaDepthRefine/src/projection/project_points.py`

```python
def project_colmap_points_to_pano(
    recon: pycolmap.Reconstruction,
    frame_id: int,
    camera_name: str,
    width: int,
    height: int,
    depth_min: float = 0.1,
    depth_max: float = 1000.0,
    ply_path: str | Path | None = None,  # ← 这里指定 fused.ply 路径
    ...
):
    # 读取点云（世界坐标）
    if ply_path is not None:
        points_world = load_ply_xyz(Path(ply_path))  # ← 从 fused.ply 读取
    else:
        # 回退：使用 COLMAP 稀疏点云
        ...
    
    # 世界坐标 → 相机坐标
    points_camera = world_to_cam(points_world, cam_from_world)
    
    # 投影到等轴柱状图
    u, v, depth, valid_mask = cam_points_to_equirectangular(
        points_camera, width, height,
        depth_min=depth_min,  # ← 已应用过滤
        depth_max=depth_max,
        convention="colmap_util",
    )
    
    # Z-buffer 选择最近点
    D_ref, M_ref = z_buffer_projection(u, v, depth, height, width)
    
    return D_ref, M_ref
```

### 2. Anchor 点生成（ElasticPanoramaDepthRefine）

**文件**: `ElasticPanoramaDepthRefine/utils/generate_anchors.py`

```python
def generate_anchors_from_ref_depth(
    ref_depth_path: str | Path,  # ← 从参考深度图读取
    ...
):
    # 读取参考深度图
    ref_depth = np.load(ref_depth_path)
    
    # 获取有效像素
    mask = np.isfinite(ref_depth) & (ref_depth > 0)
    v_valid, u_valid = np.where(mask)
    depths_valid = ref_depth[v_valid, u_valid]
    
    # 转换为 (theta, phi, depth)
    theta, phi = pixel_to_theta_phi(u_valid, v_valid, width, height, convention)
    anchors = np.stack([theta, phi, depths_valid], axis=1)
    
    return anchors
```

## 深度值定义

**深度值 = 到相机原点的欧式距离（radial distance）**

```python
# 在 project_colmap_points_to_pano 中：
points_camera = world_to_cam(points_world, cam_from_world)  # (N, 3)
depths = np.linalg.norm(points_camera, axis=1)  # sqrt(x² + y² + z²)
```

这与 `fused_remap.py` 和 `spherical_camera.py` 的定义一致。

## 过滤层级

### 第一层：参考深度图生成时（PanoramaDepthRefine）
- `depth_min`: 0.1 米（默认）
- `depth_max`: 1000.0 米（默认）
- 过滤位置：`cam_points_to_equirectangular()` 函数中

### 第二层：Anchor 生成时（可选）
- 采样率：`sample_rate`（默认 1.0，即使用所有有效像素）
- 如果 `sample_rate < 1.0`，会随机采样

### 第三层：ElasticPanoramaDepthRefine 使用时
- `anchor_filter.max_depth`: 100.0 米（默认）
- 过滤位置：`main.py` 第2步（转换 anchor 坐标时）

## 总结

- ✅ Anchor **不是直接从 fused.ply 读取的**
- ✅ Anchor **是从参考深度图生成的**
- ✅ 参考深度图**是从 fused.ply 投影生成的**
- ✅ 深度值定义为**到相机原点的欧式距离**（radial distance）
- ✅ 过滤发生在多个层级，最终在 ElasticPanoramaDepthRefine 中应用用户配置的 `max_depth`
