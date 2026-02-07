"""
弹性求解器模块
- 构建稀疏 Laplacian
- 加入 anchor 约束
- 解线性系统
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve
from typing import Tuple, Optional


def build_anchor_matrix(
    anchor_indices: np.ndarray,
    alpha: float,
    N: int
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    构建 anchor 约束矩阵和右端项
    
    Args:
        anchor_indices: (K,) 整数，anchor 像素的线性索引
        alpha: anchor 权重
        N: 总像素数
        
    Returns:
        A: (N, N) 稀疏对角矩阵，anchor 位置为 alpha
        b: (N,) 右端项向量
    """
    # 构建对角矩阵：anchor 位置为 alpha，其他为 0
    A_data = np.full(len(anchor_indices), alpha, dtype=np.float32)
    A = sp.csr_matrix(
        (A_data, (anchor_indices, anchor_indices)),
        shape=(N, N)
    )
    
    # 右端项初始化为 0（将在调用时设置）
    b = np.zeros(N, dtype=np.float32)
    
    return A, b


def build_anchor_rhs(
    anchor_indices: np.ndarray,
    anchor_values: np.ndarray,
    alpha: float,
    N: int
) -> np.ndarray:
    """
    构建 anchor 约束的右端项
    
    b[i] = alpha * (log(D_ref) - log(D0))
    
    Args:
        anchor_indices: (K,) anchor 像素索引
        anchor_values: (K,) anchor 的 log-depth 目标值
        alpha: anchor 权重
        N: 总像素数
        
    Returns:
        b: (N,) 右端项向量
    """
    b = np.zeros(N, dtype=np.float32)
    b[anchor_indices] = alpha * anchor_values
    return b


def solve_elastic_system(
    L_elastic: sp.csr_matrix,
    L_grad: sp.csr_matrix,
    A_anchor: sp.csr_matrix,
    b_anchor: np.ndarray,
    lambda_grad: float,
    method: str = "cg",
    max_iter: int = 1000,
    tol: float = 1e-6
) -> np.ndarray:
    """
    求解线性系统
    
    (L_elastic + λ_grad * L_grad + A_anchor) Δ = b_anchor
    
    Args:
        L_elastic: (N, N) 加权弹性 Laplacian
        L_grad: (N, N) 梯度保持 Laplacian
        A_anchor: (N, N) anchor 对角矩阵
        b_anchor: (N,) anchor 右端项
        lambda_grad: 梯度保持项权重
        method: 求解方法 'cg' 或 'spsolve'
        max_iter: CG 最大迭代次数
        tol: CG 收敛容差
        
    Returns:
        delta: (N,) float32，位移场 Δ
    """
    N = L_elastic.shape[0]
    
    # 构建系统矩阵
    A_system = L_elastic + lambda_grad * L_grad + A_anchor
    
    # 确保矩阵是对称的（数值误差可能导致不对称）
    A_system = (A_system + A_system.T) / 2.0
    
    if method == "cg":
        # 共轭梯度法
        print(f"  开始 CG 求解（矩阵大小: {A_system.shape[0]} x {A_system.shape[1]}）...")
        delta, info = cg(
            A_system,
            b_anchor,
            maxiter=max_iter,
            rtol=tol,
            atol=0.0
        )
        if info == 0:
            print(f"  CG 收敛成功")
        elif info > 0:
            print(f"  警告: CG 在 {info} 次迭代后未收敛")
        else:
            print(f"  警告: CG 求解失败，info={info}")
    elif method == "spsolve":
        # 直接求解（Cholesky）
        delta = spsolve(A_system, b_anchor)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return delta.astype(np.float32)


def refine_depth(
    depth_dap: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_depths: np.ndarray,
    L_elastic: sp.csr_matrix,
    L_grad: sp.csr_matrix,
    lambda_grad: float,
    alpha_anchor: float,
    method: str = "cg",
    max_iter: int = 1000,
    tol: float = 1e-6
) -> np.ndarray:
    """
    完整的深度矫正流程
    
    Args:
        depth_dap: (H, W) float32，初始 DAP 深度
        anchor_indices: (K,) 整数，anchor 像素线性索引
        anchor_depths: (K,) float32，anchor 参考深度值（线性空间）
        L_elastic: (N, N) 加权弹性 Laplacian
        L_grad: (N, N) 梯度保持 Laplacian
        lambda_grad: 梯度保持项权重
        alpha_anchor: anchor 权重
        method: 求解方法
        max_iter: CG 最大迭代次数
        tol: CG 收敛容差
        
    Returns:
        depth_refined: (H, W) float32，矫正后的深度
    """
    height, width = depth_dap.shape
    N = height * width
    
    # 转换到 log-depth 空间
    log_depth_dap = np.log(np.maximum(depth_dap, 1e-6))
    log_depth_dap_flat = log_depth_dap.flatten()
    
    # Anchor 目标值（log-depth 空间）
    anchor_log_depths = np.log(np.maximum(anchor_depths, 1e-6))
    
    # 构建 anchor 约束
    A_anchor, _ = build_anchor_matrix(anchor_indices, alpha_anchor, N)
    b_anchor = build_anchor_rhs(
        anchor_indices,
        anchor_log_depths - log_depth_dap_flat[anchor_indices],
        alpha_anchor,
        N
    )
    
    # 求解线性系统
    delta_flat = solve_elastic_system(
        L_elastic,
        L_grad,
        A_anchor,
        b_anchor,
        lambda_grad,
        method=method,
        max_iter=max_iter,
        tol=tol
    )
    
    # 计算最终 log-depth
    log_depth_refined_flat = log_depth_dap_flat + delta_flat
    
    # 转换回线性空间
    depth_refined_flat = np.exp(log_depth_refined_flat)
    depth_refined = depth_refined_flat.reshape(height, width)
    
    return depth_refined
