"""
数据读取和坐标转换模块
- 读取 depth、anchor
- 坐标与像素索引转换（θφ ↔ 像素）
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import imageio


def load_depth(depth_path: str | Path) -> np.ndarray:
    """
    读取深度图
    
    Args:
        depth_path: .npy 文件路径
        
    Returns:
        depth: (H, W) float32 深度图
    """
    depth_path = Path(depth_path)
    if not depth_path.exists():
        raise FileNotFoundError(f"深度图不存在: {depth_path}")
    
    depth = np.load(depth_path).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"深度图应为 2D 数组，得到 {depth.ndim}D")
    
    return depth


def load_rgb(rgb_path: str | Path) -> Optional[np.ndarray]:
    """
    读取 RGB 图像（可选）
    
    Args:
        rgb_path: .png 文件路径
        
    Returns:
        rgb: (H, W, 3) uint8 RGB 图像，如果文件不存在返回 None
    """
    rgb_path = Path(rgb_path)
    if not rgb_path.exists():
        return None
    
    rgb = imageio.imread(rgb_path)
    if rgb.ndim == 2:
        # 灰度图转 RGB
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.ndim == 3 and rgb.shape[2] == 4:
        # RGBA 转 RGB
        rgb = rgb[:, :, :3]
    
    return rgb.astype(np.uint8)


def load_anchors(anchors_path: str | Path) -> np.ndarray:
    """
    读取 anchor 点
    
    Args:
        anchors_path: .npy 文件路径，格式 [K, 3] -> (theta, phi, depth)
        
    Returns:
        anchors: (K, 3) float32，每行为 (theta, phi, depth)
    """
    anchors_path = Path(anchors_path)
    if not anchors_path.exists():
        # 提供更详细的错误信息，包括如何生成 anchor 文件
        error_msg = f"\n❌ Anchor 文件不存在: {anchors_path}\n\n"
        error_msg += "💡 如何生成 anchor 文件（独立工具，不依赖其他项目）：\n"
        error_msg += "   方法1：从 fused.ply 直接生成（推荐）\n"
        error_msg += "      python -m utils.generate_anchors_from_ply \\\n"
        error_msg += "          --ply <fused.ply路径> \\\n"
        error_msg += "          --colmap_dir <COLMAP重建目录> \\\n"
        error_msg += "          --pano_name <全景图名称> \\\n"
        error_msg += "          --output logs/<pano_name>_anchor.npy \\\n"
        error_msg += "          --sample_rate 0.1\n"
        error_msg += "   例如：\n"
        error_msg += "      python -m utils.generate_anchors_from_ply \\\n"
        error_msg += "          --ply /path/to/fused.ply \\\n"
        error_msg += "          --colmap_dir /path/to/colmap/sparse/0 \\\n"
        error_msg += "          --pano_name point3_median \\\n"
        error_msg += "          --output logs/point3_median_anchor.npy\n"
        error_msg += "\n   方法2：从参考深度图生成（如果已有参考深度图）\n"
        error_msg += "      python -m utils.generate_anchors \\\n"
        error_msg += "          --ref_depth <参考深度图路径> \\\n"
        error_msg += "          --output logs/<pano_name>_anchor.npy \\\n"
        error_msg += "          --sample_rate 0.1\n"
        raise FileNotFoundError(error_msg)
    
    anchors = np.load(anchors_path).astype(np.float32)
    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError(f"Anchor 应为 [K, 3] 格式，得到形状 {anchors.shape}")
    
    return anchors


def theta_phi_to_pixel(
    theta: np.ndarray,
    phi: np.ndarray,
    width: int,
    height: int,
    convention: str = "colmap_util"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将球面坐标 (θ, φ) 转换为像素坐标 (u, v)
    
    根据 fused_remap.py 和 spherical_camera.py 的定义：
    - colmap_util: theta = yaw = atan2(x, z) [-π, π], phi = pitch = -atan2(y, sqrt(x^2+z^2)) [-π/2, π/2]
    - dap: theta = 方位角 [0, 2π), phi = 极角 [0, π]
    
    Args:
        theta: 经度角（yaw 或方位角）
        phi: 纬度角（pitch 或极角）
        width: 图像宽度
        height: 图像高度
        convention: 坐标约定
        
    Returns:
        u: (N,) 像素列索引 [0, width)
        v: (N,) 像素行索引 [0, height)
    """
    if convention == "colmap_util":
        # colmap_util 约定：theta = yaw [-π, π], phi = pitch [-π/2, π/2]
        # 与 fused_remap.py 第186-189行一致
        yaw = theta  # 直接使用，已经是 [-π, π]
        pitch = phi  # 直接使用，已经是 [-π/2, π/2]
        u = (1.0 + yaw / np.pi) * 0.5  # [0, 1]
        v = (1.0 - pitch * 2.0 / np.pi) * 0.5  # [0, 1]
    elif convention == "dap":
        # DAP 约定：theta [0, 2π), phi [0, π]
        u = 1.0 - theta / (2.0 * np.pi)  # [0, 1]
        v = phi / np.pi  # [0, 1]
    else:
        raise ValueError(f"Unknown convention: {convention}")
    
    # 映射到像素坐标
    u_pix = u * width
    v_pix = v * height
    
    # 边界处理
    u_pix = np.clip(u_pix, 0, width - 1e-6)
    v_pix = np.clip(v_pix, 0, height - 1e-6)
    
    return u_pix, v_pix


def pixel_to_theta_phi(
    u: np.ndarray,
    v: np.ndarray,
    width: int,
    height: int,
    convention: str = "colmap_util"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将像素坐标 (u, v) 转换为球面坐标 (θ, φ)
    
    根据 fused_remap.py 和 spherical_camera.py 的定义：
    - colmap_util: theta = yaw, phi = pitch
    - dap: theta = 方位角, phi = 极角
    
    Args:
        u: (N,) 像素列索引 [0, width)
        v: (N,) 像素行索引 [0, height)
        width: 图像宽度
        height: 图像高度
        convention: 坐标约定
        
    Returns:
        theta: (N,) 经度角（yaw 或方位角）
        phi: (N,) 纬度角（pitch 或极角）
    """
    # 归一化到 [0, 1]
    u_norm = u / width
    v_norm = v / height
    
    if convention == "colmap_util":
        # 与 fused_remap.py 第188-189行反向一致
        yaw = (u_norm * 2.0 - 1.0) * np.pi  # [-π, π]
        pitch = (1.0 - v_norm * 2.0) * np.pi / 2.0  # [-π/2, π/2]
        theta = yaw  # theta = yaw
        phi = pitch  # phi = pitch
    elif convention == "dap":
        theta = (1.0 - u_norm) * 2.0 * np.pi  # [0, 2π)
        phi = v_norm * np.pi  # [0, π]
    else:
        raise ValueError(f"Unknown convention: {convention}")
    
    return theta, phi


def anchors_to_pixel_indices(
    anchors: np.ndarray,
    width: int,
    height: int,
    convention: str = "colmap_util"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 anchor 点从 (theta, phi, depth) 转换为像素索引
    
    Args:
        anchors: (K, 3) float32，每行为 (theta, phi, depth)
        width: 图像宽度
        height: 图像高度
        convention: 坐标约定
        
    Returns:
        u: (K,) 像素列索引（整数）
        v: (K,) 像素行索引（整数）
        depth: (K,) 深度值
    """
    theta = anchors[:, 0]
    phi = anchors[:, 1]
    depth = anchors[:, 2]
    
    u, v = theta_phi_to_pixel(theta, phi, width, height, convention)
    
    # 转换为整数索引
    u_int = np.clip(np.floor(u).astype(np.int32), 0, width - 1)
    v_int = np.clip(np.floor(v).astype(np.int32), 0, height - 1)
    
    return u_int, v_int, depth
