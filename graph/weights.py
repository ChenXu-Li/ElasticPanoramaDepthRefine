"""
各向异性弹性权重计算模块
- 计算 w_ij = exp(-λ_g * ||∇z0(i) - ∇z0(j)||_2) * exp(-λ_e * E(i,j))
"""
import numpy as np
from typing import Optional
from geometry.sphere import get_edge_gradient_diff


def compute_anisotropic_weights(
    log_depth: np.ndarray,
    i_indices: np.ndarray,
    j_indices: np.ndarray,
    lambda_g: float,
    lambda_e: float,
    edge_mask: Optional[np.ndarray] = None,
    sky_mask: Optional[np.ndarray] = None,
    return_terms: bool = False
):
    """
    计算各向异性弹性权重
    
    w_ij = exp(-λ_g * ||∇z0(i) - ∇z0(j)||_2) * exp(-λ_e * E(i,j))
    
    如果任一端点在天空区域，权重设为0（天空不影响周围像素）
    
    Args:
        log_depth: (H, W) float32，初始 log-depth 值
        i_indices: (E,) 源节点线性索引
        j_indices: (E,) 目标节点线性索引
        lambda_g: 梯度权重参数
        lambda_e: 边缘权重参数
        edge_mask: (E,) bool，可选，True 表示跨越边缘
        sky_mask: (H, W) bool，可选，True 表示天空区域
        return_terms: 如果为 True，返回 (weights, grad_term, edge_term, grad_diff)
        
    Returns:
        weights: (E,) float32，各向异性权重
        如果 return_terms=True，返回 (weights, grad_term, edge_term, grad_diff)
    """
    # 首先识别涉及天空区域的边
    sky_edge_mask = None
    if sky_mask is not None:
        sky_mask_flat = sky_mask.flatten()
        i_in_sky = sky_mask_flat[i_indices]
        j_in_sky = sky_mask_flat[j_indices]
        # 如果任一端点在天空，该边涉及天空区域
        sky_edge_mask = i_in_sky | j_in_sky
    
    # 计算梯度差异
    grad_diff = get_edge_gradient_diff(log_depth, i_indices, j_indices)
    
    # 梯度项：exp(-λ_g * ||∇z0(i) - ∇z0(j)||_2)
    grad_term = np.exp(-lambda_g * grad_diff)
    
    # 边缘项：exp(-λ_e * E(i,j))
    if edge_mask is not None:
        edge_term = np.exp(-lambda_e * edge_mask.astype(np.float32))
    else:
        edge_term = np.ones(len(i_indices), dtype=np.float32)
    
    # 组合权重
    weights = grad_term * edge_term
    
    # 天空 mask：如果任一端点在天空区域，将权重设为0
    # 这样天空区域的像素在优化时不会影响周围像素
    # 注意：即使边缘项很小（exp(-λ_e)），只要涉及天空，权重就设为0
    if sky_edge_mask is not None:
        weights[sky_edge_mask] = 0.0
        # 同时将梯度项和边缘项也设为0（用于可视化时的一致性）
        if return_terms:
            grad_term = grad_term.copy()
            edge_term = edge_term.copy()
            grad_term[sky_edge_mask] = 0.0
            edge_term[sky_edge_mask] = 0.0
    
    if return_terms:
        return weights, grad_term, edge_term, grad_diff
    else:
        return weights
