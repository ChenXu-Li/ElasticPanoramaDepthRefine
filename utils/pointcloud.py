"""
点云生成模块
"""
import numpy as np
from pathlib import Path
from typing import Optional
from plyfile import PlyData, PlyElement


def depth_to_pointcloud_ply(
    depth: np.ndarray,
    rgb: Optional[np.ndarray],
    output_path: str | Path,
    convention: str = "dap"
) -> None:
    """
    将等轴柱状深度图转换为点云，并保存为 binary PLY
    
    使用 DAP 约定（与 depth2point.py 一致）：
        theta = (1 - u) * 2π  # [0, 2π]
        phi = v * π           # [0, π]
        方向向量：
            x = sin(phi) * cos(theta)
            y = sin(phi) * sin(theta)
            z = cos(phi)
        点云：
            p = depth * dir
    
    Args:
        depth: (H, W) float32 深度图（DAP 约定）
        rgb: (H, W, 3) uint8 RGB 图像，可选
        output_path: 输出 PLY 文件路径
        convention: 坐标约定（默认 'dap'，保持与 DAP 一致）
    """
    H, W = depth.shape
    
    # 有效像素 mask
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        print("警告: 深度图中没有有效像素，跳过点云导出")
        return
    
    # 构造像素坐标网格（与 DAP 的 image_uv 函数保持一致）
    u = np.linspace(0, 1, W, dtype=np.float32)  # [W]
    v = np.linspace(0, 1, H, dtype=np.float32)  # [H]
    u_grid, v_grid = np.meshgrid(u, v)  # [H, W]
    
    # 只保留有效像素
    depth_valid = depth[valid]
    u_valid = u_grid[valid]
    v_valid = v_grid[valid]
    
    if rgb is not None:
        rgb_valid = rgb[valid].astype(np.uint8)
    else:
        rgb_valid = np.zeros((len(depth_valid), 3), dtype=np.uint8)
    
    # DAP 约定：theta/phi
    theta = (1.0 - u_valid) * (2.0 * np.pi)  # [0, 2π]
    phi = v_valid * np.pi                     # [0, π]
    
    # 计算方向向量（DAP 约定）
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # 单位方向向量（DAP 约定）
    dirs_x = sin_phi * cos_theta
    dirs_y = sin_phi * sin_theta
    dirs_z = cos_phi
    
    # 计算 3D 点云
    points = depth_valid[:, None] * np.stack([dirs_x, dirs_y, dirs_z], axis=1)
    points = points.astype(np.float32)
    
    print(f"  有效点数量: {points.shape[0]:,}")
    
    # 准备 PLY 数据
    vertices = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = rgb_valid[:, 0]
    vertices["green"] = rgb_valid[:, 1]
    vertices["blue"] = rgb_valid[:, 2]
    
    # 保存为 binary PLY
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    el = PlyElement.describe(vertices, "vertex")
    PlyData([el], byte_order="<").write(str(output_path))
    print(f"  PLY 点云已保存: {output_path}")
