"""
球面几何模块
- 球面邻接（4-connected，经度方向周期 wrap）
- 梯度计算（log-depth 空间）
"""
import numpy as np
from typing import Tuple, List


def get_4_neighbors(
    height: int,
    width: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    获取 4-connected 邻接关系（考虑经度方向周期 wrap）
    
    Args:
        height: 图像高度
        width: 图像宽度
        
    Returns:
        i_indices: (E,) 源节点索引（线性索引）
        j_indices: (E,) 目标节点索引（线性索引）
        edge_types: (E,) 边类型：0=右，1=下，2=左，3=上
        edge_weights: (E,) 边权重（初始为1，后续会被各向异性权重替换）
    """
    total_pixels = height * width
    i_list = []
    j_list = []
    edge_type_list = []
    
    # 线性索引转 (v, u)
    def linear_to_2d(idx):
        v = idx // width
        u = idx % width
        return v, u
    
    # (v, u) 转线性索引
    def coord_to_linear(v, u):
        return v * width + u
    
    for idx in range(total_pixels):
        v, u = linear_to_2d(idx)
        
        # 右邻居 (u+1, v)
        u_right = (u + 1) % width  # 周期 wrap
        j_right = coord_to_linear(v, u_right)
        i_list.append(idx)
        j_list.append(j_right)
        edge_type_list.append(0)
        
        # 下邻居 (u, v+1)
        if v + 1 < height:
            j_down = coord_to_linear(v + 1, u)
            i_list.append(idx)
            j_list.append(j_down)
            edge_type_list.append(1)
        
        # 左邻居 (u-1, v)
        u_left = (u - 1) % width  # 周期 wrap
        j_left = coord_to_linear(v, u_left)
        i_list.append(idx)
        j_list.append(j_left)
        edge_type_list.append(2)
        
        # 上邻居 (u, v-1)
        if v - 1 >= 0:
            j_up = coord_to_linear(v - 1, u)
            i_list.append(idx)
            j_list.append(j_up)
            edge_type_list.append(3)
    
    i_indices = np.array(i_list, dtype=np.int32)
    j_indices = np.array(j_list, dtype=np.int32)
    edge_types = np.array(edge_type_list, dtype=np.int32)
    
    return i_indices, j_indices, edge_types


def compute_gradient_log_depth(
    log_depth: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 log-depth 空间计算梯度
    
    Args:
        log_depth: (H, W) float32，log-depth 值
        
    Returns:
        grad_u: (H, W) float32，u 方向（经度）梯度
        grad_v: (H, W) float32，v 方向（纬度）梯度
    """
    height, width = log_depth.shape
    
    # 使用中心差分计算梯度
    # u 方向（经度，考虑周期 wrap）
    grad_u = np.zeros_like(log_depth)
    grad_u[:, :-1] = log_depth[:, 1:] - log_depth[:, :-1]
    grad_u[:, -1] = log_depth[:, 0] - log_depth[:, -1]  # 周期 wrap
    
    # v 方向（纬度，边界使用单边差分）
    grad_v = np.zeros_like(log_depth)
    grad_v[:-1, :] = log_depth[1:, :] - log_depth[:-1, :]
    grad_v[-1, :] = 0  # 边界
    
    return grad_u, grad_v


def compute_gradient_magnitude(
    log_depth: np.ndarray
) -> np.ndarray:
    """
    计算 log-depth 梯度的 L2 范数
    
    Args:
        log_depth: (H, W) float32，log-depth 值
        
    Returns:
        grad_mag: (H, W) float32，梯度幅值
    """
    grad_u, grad_v = compute_gradient_log_depth(log_depth)
    grad_mag = np.sqrt(grad_u ** 2 + grad_v ** 2)
    return grad_mag


def get_edge_gradient_diff(
    log_depth: np.ndarray,
    i_indices: np.ndarray,
    j_indices: np.ndarray
) -> np.ndarray:
    """
    计算相邻像素之间的梯度差异（用于各向异性权重）
    
    Args:
        log_depth: (H, W) float32，log-depth 值
        i_indices: (E,) 源节点线性索引
        j_indices: (E,) 目标节点线性索引
        
    Returns:
        grad_diff: (E,) float32，相邻像素的梯度差异 L2 范数
    """
    height, width = log_depth.shape
    
    # 计算每个像素的梯度
    grad_u, grad_v = compute_gradient_log_depth(log_depth)
    grad_flat = np.stack([grad_u.flatten(), grad_v.flatten()], axis=1)  # (N, 2)
    
    # 获取相邻像素的梯度
    grad_i = grad_flat[i_indices]  # (E, 2)
    grad_j = grad_flat[j_indices]  # (E, 2)
    
    # 计算梯度差异的 L2 范数
    grad_diff = np.linalg.norm(grad_i - grad_j, axis=1)
    
    return grad_diff
