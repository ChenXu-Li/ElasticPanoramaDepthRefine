"""
稀疏 Laplacian 矩阵构建模块
"""
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional


def build_weighted_laplacian(
    height: int,
    width: int,
    i_indices: np.ndarray,
    j_indices: np.ndarray,
    weights: np.ndarray
) -> sp.csr_matrix:
    """
    构建加权图 Laplacian 矩阵
    
    L = D - W
    其中 D 是度矩阵，W 是权重矩阵
    
    Args:
        height: 图像高度
        width: 图像宽度
        i_indices: (E,) 源节点索引
        j_indices: (E,) 目标节点索引
        weights: (E,) 边权重
        
    Returns:
        laplacian: (N, N) 稀疏 CSR 矩阵，N = height * width
    """
    N = height * width
    
    # 构建对称权重矩阵（无向图）
    # 对于每条边 (i, j)，在 W[i, j] 和 W[j, i] 处设置权重
    row_indices = np.concatenate([i_indices, j_indices])
    col_indices = np.concatenate([j_indices, i_indices])
    weight_values = np.concatenate([weights, weights])
    
    # 构建稀疏权重矩阵
    W = sp.csr_matrix(
        (weight_values, (row_indices, col_indices)),
        shape=(N, N)
    )
    
    # 计算度矩阵（每行的和）
    degrees = np.array(W.sum(axis=1)).flatten()
    
    # Laplacian = D - W
    D = sp.diags(degrees, format='csr')
    L = D - W
    
    return L


def build_gradient_laplacian(
    height: int,
    width: int,
    sky_mask: Optional[np.ndarray] = None
) -> sp.csr_matrix:
    """
    构建标准梯度 Laplacian（用于梯度保持项）
    
    这对应于 L_grad = Σ_i ||∇Δ(i)||^2
    
    如果提供了 sky_mask，涉及天空区域的边的权重将被设为0，
    这样梯度保持项不会在天空像素和物体边缘像素之间产生约束。
    
    Args:
        height: 图像高度
        width: 图像宽度
        sky_mask: (H, W) bool，可选，True 表示天空区域
        
    Returns:
        L_grad: (N, N) 稀疏 CSR 矩阵
    """
    from geometry.sphere import get_4_neighbors
    
    N = height * width
    i_indices, j_indices, _ = get_4_neighbors(height, width)
    
    # 标准梯度 Laplacian：所有边权重为 1
    weights = np.ones(len(i_indices), dtype=np.float32)
    
    # 如果提供了天空 mask，将涉及天空的边的权重设为0
    if sky_mask is not None:
        sky_mask_flat = sky_mask.flatten()
        i_in_sky = sky_mask_flat[i_indices]
        j_in_sky = sky_mask_flat[j_indices]
        # 如果任一端点在天空，该边涉及天空区域，权重设为0
        sky_edge_mask = i_in_sky | j_in_sky
        weights[sky_edge_mask] = 0.0
    
    L_grad = build_weighted_laplacian(height, width, i_indices, j_indices, weights)
    
    return L_grad
