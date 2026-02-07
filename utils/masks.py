"""
Mask 模块：sky / edge 检测
"""
import numpy as np
from typing import Optional
import cv2


def detect_sky_mask(
    rgb: Optional[np.ndarray],
    brightness_threshold: float = 0.9
) -> Optional[np.ndarray]:
    """
    检测天空 mask（基于亮度）
    
    Args:
        rgb: (H, W, 3) uint8 RGB 图像，如果为 None 则返回 None
        brightness_threshold: 亮度阈值 [0, 1]
        
    Returns:
        sky_mask: (H, W) bool，True 表示天空区域，如果 rgb 为 None 则返回 None
    """
    if rgb is None:
        return None
    
    # 转换为灰度并归一化
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # 高亮度区域视为天空
    sky_mask = gray > brightness_threshold
    
    return sky_mask


def detect_rgb_edge(
    rgb: Optional[np.ndarray],
    threshold: float = 0.1
) -> Optional[np.ndarray]:
    """
    检测 RGB 边缘
    
    Args:
        rgb: (H, W, 3) uint8 RGB 图像，如果为 None 则返回 None
        threshold: 边缘阈值
        
    Returns:
        edge_mask: (H, W) bool，True 表示边缘，如果 rgb 为 None 则返回 None
    """
    if rgb is None:
        return None
    
    # 转换为灰度
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Canny 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    
    # 转换为 bool mask
    edge_mask = edges > 0
    
    return edge_mask


def detect_depth_edge(
    log_depth: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    检测深度边缘（log-depth 空间）
    
    Args:
        log_depth: (H, W) float32，log-depth 值
        threshold: 边缘阈值（log-depth 梯度幅值）
        
    Returns:
        edge_mask: (H, W) bool，True 表示深度边缘
    """
    from geometry.sphere import compute_gradient_magnitude
    
    # 计算梯度幅值
    grad_mag = compute_gradient_magnitude(log_depth)
    
    # 高梯度区域视为边缘
    edge_mask = grad_mag > threshold
    
    return edge_mask


def build_edge_mask_for_edges(
    height: int,
    width: int,
    i_indices: np.ndarray,
    j_indices: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    log_depth: Optional[np.ndarray] = None,
    sky_mask: Optional[np.ndarray] = None,
    rgb_edge_threshold: float = 0.1,
    depth_edge_threshold: float = 0.5
) -> np.ndarray:
    """
    为图边构建边缘 mask
    
    Args:
        height: 图像高度
        width: 图像宽度
        i_indices: (E,) 源节点索引
        j_indices: (E,) 目标节点索引
        rgb: (H, W, 3) 可选 RGB 图像
        log_depth: (H, W) 可选 log-depth
        sky_mask: (H, W) 可选天空 mask
        rgb_edge_threshold: RGB 边缘阈值
        depth_edge_threshold: 深度边缘阈值
        
    Returns:
        edge_mask: (E,) bool，True 表示该边跨越边缘
    """
    N = height * width
    edge_mask = np.zeros(len(i_indices), dtype=bool)
    
    # 线性索引转 2D 坐标
    def linear_to_2d(idx):
        v = idx // width
        u = idx % width
        return v, u
    
    # RGB 边缘
    if rgb is not None:
        rgb_edge = detect_rgb_edge(rgb, rgb_edge_threshold)
        if rgb_edge is not None:
            rgb_edge_flat = rgb_edge.flatten()
            # 如果任一端点在边缘上，则该边跨越边缘
            edge_mask |= rgb_edge_flat[i_indices] | rgb_edge_flat[j_indices]
    
    # 深度边缘
    if log_depth is not None:
        depth_edge = detect_depth_edge(log_depth, depth_edge_threshold)
        depth_edge_flat = depth_edge.flatten()
        edge_mask |= depth_edge_flat[i_indices] | depth_edge_flat[j_indices]
    
    # 天空 mask：如果任一端点在天空区域，则该边跨越天空边界
    if sky_mask is not None:
        sky_mask_flat = sky_mask.flatten()
        # 如果一边在天空，另一边不在，则跨越边界
        i_in_sky = sky_mask_flat[i_indices]
        j_in_sky = sky_mask_flat[j_indices]
        edge_mask |= (i_in_sky != j_in_sky)
    
    return edge_mask
