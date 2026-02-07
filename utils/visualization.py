"""
可视化模块
"""
import numpy as np
import imageio
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def depth_to_colormap(
    depth: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colormap: str = "turbo"
) -> np.ndarray:
    """
    将深度图转换为彩色图
    
    Args:
        depth: (H, W) float32 深度图
        vmin: 最小值，如果为 None 则使用 depth 的最小值
        vmax: 最大值，如果为 None 则使用 depth 的最大值
        colormap: colormap 名称
        
    Returns:
        colored: (H, W, 3) uint8 彩色图
    """
    valid_mask = np.isfinite(depth) & (depth > 0)
    
    if vmin is None:
        vmin = np.nanmin(depth[valid_mask]) if np.any(valid_mask) else 0.0
    if vmax is None:
        vmax = np.nanmax(depth[valid_mask]) if np.any(valid_mask) else 1.0
    
    # 归一化
    depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-6), 0, 1)
    
    # 应用 colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(depth_norm)[:, :, :3]  # 去掉 alpha 通道
    colored = (colored * 255).astype(np.uint8)
    
    # 无效区域设为黑色
    colored[~valid_mask] = 0
    
    return colored


def visualize_depth_comparison(
    depth_dap: np.ndarray,
    depth_refined: np.ndarray,
    output_path: str | Path,
    rgb: Optional[np.ndarray] = None
):
    """
    可视化深度对比（DAP vs 矫正后）
    
    Args:
        depth_dap: (H, W) 初始 DAP 深度
        depth_refined: (H, W) 矫正后深度
        output_path: 输出路径
        rgb: (H, W, 3) 可选 RGB 图像用于叠加
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # DAP 深度
    axes[0, 0].imshow(depth_to_colormap(depth_dap), aspect='auto')
    axes[0, 0].set_title("DAP Depth")
    axes[0, 0].axis('off')
    
    # 矫正后深度
    axes[0, 1].imshow(depth_to_colormap(depth_refined), aspect='auto')
    axes[0, 1].set_title("Refined Depth")
    axes[0, 1].axis('off')
    
    # 差异
    depth_diff = depth_refined - depth_dap
    valid_mask = np.isfinite(depth_diff)
    if np.any(valid_mask):
        vmin = np.nanpercentile(depth_diff[valid_mask], 1)
        vmax = np.nanpercentile(depth_diff[valid_mask], 99)
        im = axes[1, 0].imshow(
            depth_diff,
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            aspect='auto'
        )
        axes[1, 0].set_title("Difference (Refined - DAP)")
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
    
    # RGB 叠加（如果有）
    if rgb is not None:
        axes[1, 1].imshow(rgb, aspect='auto')
        axes[1, 1].set_title("RGB")
        axes[1, 1].axis('off')
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_anchors(
    anchor_indices: np.ndarray,
    anchor_depths: np.ndarray,
    width: int,
    height: int,
    output_path: str | Path,
    rgb: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    point_size: int = 2
):
    """
    在等轴柱状图上可视化 anchor 点
    
    Args:
        anchor_indices: (K,) 整数，anchor 像素的线性索引
        anchor_depths: (K,) float32，anchor 深度值
        width: 图像宽度
        height: 图像高度
        output_path: 输出路径
        rgb: (H, W, 3) 可选 RGB 图像作为背景
        depth: (H, W) 可选深度图作为背景
        point_size: anchor 点标记大小
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 转换线性索引为 (v, u) 坐标
    v_anchors = anchor_indices // width
    u_anchors = anchor_indices % width
    
    # 左图：RGB 或深度图 + anchor 点
    if rgb is not None:
        axes[0].imshow(rgb, aspect='auto')
        axes[0].set_title(f"RGB with {len(anchor_indices)} Anchors")
    elif depth is not None:
        axes[0].imshow(depth_to_colormap(depth), aspect='auto')
        axes[0].set_title(f"Depth with {len(anchor_indices)} Anchors")
    else:
        # 创建空白背景
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        axes[0].imshow(bg, aspect='auto')
        axes[0].set_title(f"{len(anchor_indices)} Anchors")
    
    # 绘制 anchor 点（红色）
    axes[0].scatter(u_anchors, v_anchors, c='red', s=point_size, alpha=0.6, marker='.')
    axes[0].axis('off')
    
    # 右图：深度分布直方图 + anchor 点深度分布
    if depth is not None:
        depth_valid = depth[np.isfinite(depth) & (depth > 0)]
        if len(depth_valid) > 0:
            axes[1].hist(depth_valid.flatten(), bins=100, alpha=0.5, label='All pixels', color='blue', density=True)
    
    # Anchor 深度分布
    axes[1].hist(anchor_depths, bins=50, alpha=0.7, label=f'Anchors ({len(anchor_depths)})', color='red', density=True)
    axes[1].set_xlabel('Depth (meters)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Depth Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f"Anchor Stats:\n"
    stats_text += f"  Count: {len(anchor_depths)}\n"
    stats_text += f"  Min: {anchor_depths.min():.2f}m\n"
    stats_text += f"  Max: {anchor_depths.max():.2f}m\n"
    stats_text += f"  Mean: {anchor_depths.mean():.2f}m\n"
    stats_text += f"  Median: {np.median(anchor_depths):.2f}m"
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_depth_change(
    depth_dap: np.ndarray,
    depth_refined: np.ndarray,
    output_path_log: str | Path,
    output_path_linear: str | Path,
    rgb: Optional[np.ndarray] = None
):
    """
    可视化矫正前后深度变化（热力图）
    
    分别显示 log 空间和真实空间下的深度变化
    
    Args:
        depth_dap: (H, W) 初始 DAP 深度（米）
        depth_refined: (H, W) 矫正后深度（米）
        output_path_log: log 空间变化热力图输出路径
        output_path_linear: 真实空间变化热力图输出路径
        rgb: (H, W, 3) 可选 RGB 图像用于叠加
    """
    # Log 空间下的变化
    log_depth_dap = np.log(np.maximum(depth_dap, 1e-6))
    log_depth_refined = np.log(np.maximum(depth_refined, 1e-6))
    log_depth_change = log_depth_refined - log_depth_dap
    
    # 真实空间下的变化
    depth_change = depth_refined - depth_dap
    
    # 计算有效 mask
    valid_mask_log = np.isfinite(log_depth_change)
    valid_mask_linear = np.isfinite(depth_change)
    
    # 创建 log 空间变化热力图
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Log 空间变化 - 热力图
    if np.any(valid_mask_log):
        vmin_log = np.nanpercentile(log_depth_change[valid_mask_log], 1)
        vmax_log = np.nanpercentile(log_depth_change[valid_mask_log], 99)
        im1 = axes[0, 0].imshow(log_depth_change, cmap='RdBu_r', vmin=vmin_log, vmax=vmax_log, aspect='auto')
        axes[0, 0].set_title(f"Log Space Change\nRange: [{vmin_log:.4f}, {vmax_log:.4f}]", fontsize=14)
        axes[0, 0].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        cbar1.set_label("Δ log(depth)", rotation=270, labelpad=20)
    else:
        axes[0, 0].axis('off')
        axes[0, 0].set_title("Log Space Change (No valid data)", fontsize=14)
    
    # Log 空间变化 - RGB 叠加
    if rgb is not None:
        axes[0, 1].imshow(rgb, aspect='auto')
        if np.any(valid_mask_log):
            change_overlay = np.zeros((log_depth_change.shape[0], log_depth_change.shape[1], 4), dtype=np.float32)
            change_norm = (log_depth_change - vmin_log) / (vmax_log - vmin_log + 1e-6)
            change_norm = np.clip(change_norm, 0, 1)
            # 使用 RdBu_r colormap：正值（变深）用蓝色，负值（变浅）用红色
            change_overlay[:, :, 0] = np.where(change_norm > 0.5, 1.0, 0.0)  # 红色（负值）
            change_overlay[:, :, 2] = np.where(change_norm <= 0.5, 1.0, 0.0)  # 蓝色（正值）
            change_overlay[:, :, 3] = np.abs(change_norm - 0.5) * 2.0 * 0.6  # 透明度
            axes[0, 1].imshow(change_overlay, aspect='auto')
        axes[0, 1].set_title("RGB + Log Space Change", fontsize=14)
        axes[0, 1].axis('off')
    else:
        axes[0, 1].axis('off')
    
    # 真实空间变化 - 热力图
    if np.any(valid_mask_linear):
        vmin_linear = np.nanpercentile(depth_change[valid_mask_linear], 1)
        vmax_linear = np.nanpercentile(depth_change[valid_mask_linear], 99)
        im2 = axes[1, 0].imshow(depth_change, cmap='RdBu_r', vmin=vmin_linear, vmax=vmax_linear, aspect='auto')
        axes[1, 0].set_title(f"Linear Space Change (meters)\nRange: [{vmin_linear:.4f}, {vmax_linear:.4f}]", fontsize=14)
        axes[1, 0].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        cbar2.set_label("Δ depth (m)", rotation=270, labelpad=20)
    else:
        axes[1, 0].axis('off')
        axes[1, 0].set_title("Linear Space Change (No valid data)", fontsize=14)
    
    # 真实空间变化 - RGB 叠加
    if rgb is not None:
        axes[1, 1].imshow(rgb, aspect='auto')
        if np.any(valid_mask_linear):
            change_overlay_linear = np.zeros((depth_change.shape[0], depth_change.shape[1], 4), dtype=np.float32)
            change_norm_linear = (depth_change - vmin_linear) / (vmax_linear - vmin_linear + 1e-6)
            change_norm_linear = np.clip(change_norm_linear, 0, 1)
            change_overlay_linear[:, :, 0] = np.where(change_norm_linear > 0.5, 1.0, 0.0)  # 红色（负值）
            change_overlay_linear[:, :, 2] = np.where(change_norm_linear <= 0.5, 1.0, 0.0)  # 蓝色（正值）
            change_overlay_linear[:, :, 3] = np.abs(change_norm_linear - 0.5) * 2.0 * 0.6  # 透明度
            axes[1, 1].imshow(change_overlay_linear, aspect='auto')
        axes[1, 1].set_title("RGB + Linear Space Change", fontsize=14)
        axes[1, 1].axis('off')
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path_log, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 创建真实空间变化热力图（单独保存）
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 真实空间变化 - 热力图
    if np.any(valid_mask_linear):
        im3 = axes[0].imshow(depth_change, cmap='RdBu_r', vmin=vmin_linear, vmax=vmax_linear, aspect='auto')
        axes[0].set_title(f"Linear Space Change (meters)\nRange: [{vmin_linear:.4f}, {vmax_linear:.4f}]", fontsize=14)
        axes[0].axis('off')
        cbar3 = plt.colorbar(im3, ax=axes[0], fraction=0.046, pad=0.04)
        cbar3.set_label("Δ depth (m)", rotation=270, labelpad=20)
    else:
        axes[0].axis('off')
        axes[0].set_title("Linear Space Change (No valid data)", fontsize=14)
    
    # 真实空间变化 - RGB 叠加
    if rgb is not None:
        axes[1].imshow(rgb, aspect='auto')
        if np.any(valid_mask_linear):
            change_overlay_linear2 = np.zeros((depth_change.shape[0], depth_change.shape[1], 4), dtype=np.float32)
            change_norm_linear2 = (depth_change - vmin_linear) / (vmax_linear - vmin_linear + 1e-6)
            change_norm_linear2 = np.clip(change_norm_linear2, 0, 1)
            change_overlay_linear2[:, :, 0] = np.where(change_norm_linear2 > 0.5, 1.0, 0.0)  # 红色（负值）
            change_overlay_linear2[:, :, 2] = np.where(change_norm_linear2 <= 0.5, 1.0, 0.0)  # 蓝色（正值）
            change_overlay_linear2[:, :, 3] = np.abs(change_norm_linear2 - 0.5) * 2.0 * 0.6  # 透明度
            axes[1].imshow(change_overlay_linear2, aspect='auto')
        axes[1].set_title("RGB + Linear Space Change", fontsize=14)
        axes[1].axis('off')
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path_linear, dpi=150, bbox_inches='tight')
    plt.close()


def save_depth(
    depth: np.ndarray,
    output_path: str | Path,
    format: str = "npy"
):
    """
    保存深度图（使用 DAP 约定）
    
    注意：保存的深度图使用 DAP 约定（u=0 在右侧），与输入的 DAP 深度图保持一致。
    深度图数组本身在优化过程中保持 DAP 约定，直接保存即可。
    
    Args:
        depth: (H, W) float32 深度图（DAP 约定）
        output_path: 输出路径
        format: 格式 'npy' 或 'png'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "npy":
        # 直接保存，保持 DAP 约定
        np.save(output_path, depth)
    elif format == "png":
        # 归一化到 0-65535 (16-bit)
        depth_valid = depth[np.isfinite(depth) & (depth > 0)]
        if len(depth_valid) > 0:
            vmin = np.nanmin(depth_valid)
            vmax = np.nanmax(depth_valid)
            depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-6), 0, 1)
            depth_16bit = (depth_norm * 65535).astype(np.uint16)
        else:
            depth_16bit = np.zeros_like(depth, dtype=np.uint16)
        imageio.imwrite(output_path, depth_16bit)
    else:
        raise ValueError(f"Unknown format: {format}")


def visualize_weight_terms(
    grad_term: np.ndarray,
    edge_term: np.ndarray,
    grad_diff: np.ndarray,
    i_indices: np.ndarray,
    j_indices: np.ndarray,
    height: int,
    width: int,
    output_path: str | Path,
    rgb: Optional[np.ndarray] = None
):
    """
    可视化梯度项和边缘项
    
    将边上的值映射到像素（使用平均值）
    
    Args:
        grad_term: (E,) 梯度项值
        edge_term: (E,) 边缘项值
        grad_diff: (E,) 梯度差异值
        i_indices: (E,) 源节点索引
        j_indices: (E,) 目标节点索引
        height: 图像高度
        width: 图像宽度
        output_path: 输出路径
        rgb: (H, W, 3) 可选 RGB 图像用于叠加
    """
    N = height * width
    
    # 将边上的值映射到像素（使用平均值）
    def edge_to_pixel(edge_values):
        pixel_values = np.zeros(N, dtype=np.float32)
        pixel_counts = np.zeros(N, dtype=np.int32)
        
        # 累加所有连接到该像素的边的值
        for idx in range(len(edge_values)):
            pixel_values[i_indices[idx]] += edge_values[idx]
            pixel_values[j_indices[idx]] += edge_values[idx]
            pixel_counts[i_indices[idx]] += 1
            pixel_counts[j_indices[idx]] += 1
        
        # 计算平均值
        pixel_counts = np.maximum(pixel_counts, 1)  # 避免除零
        pixel_values = pixel_values / pixel_counts
        
        return pixel_values.reshape(height, width)
    
    # 映射到图像空间
    grad_term_img = edge_to_pixel(grad_term)
    edge_term_img = edge_to_pixel(edge_term)
    grad_diff_img = edge_to_pixel(grad_diff)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    
    # 第一行：梯度项
    im1 = axes[0, 0].imshow(grad_term_img, cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f"Gradient Term\nRange: [{grad_term_img.min():.4f}, {grad_term_img.max():.4f}]", fontsize=12)
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # 第一行：边缘项
    im2 = axes[0, 1].imshow(edge_term_img, cmap='plasma', aspect='auto')
    axes[0, 1].set_title(f"Edge Term\nRange: [{edge_term_img.min():.4f}, {edge_term_img.max():.4f}]", fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 第一行：梯度差异
    im3 = axes[0, 2].imshow(grad_diff_img, cmap='hot', aspect='auto')
    axes[0, 2].set_title(f"Gradient Difference\nRange: [{grad_diff_img.min():.4f}, {grad_diff_img.max():.4f}]", fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 第二行：RGB 叠加（如果有）
    if rgb is not None:
        axes[1, 0].imshow(rgb, aspect='auto')
        axes[1, 0].set_title("RGB", fontsize=12)
        axes[1, 0].axis('off')
        
        # RGB + 梯度项叠加
        axes[1, 1].imshow(rgb, aspect='auto')
        grad_overlay = np.zeros((height, width, 4), dtype=np.float32)
        grad_overlay[:, :, 1] = 1.0  # 绿色
        grad_norm = (grad_term_img - grad_term_img.min()) / (grad_term_img.max() - grad_term_img.min() + 1e-6)
        grad_overlay[:, :, 3] = grad_norm * 0.5
        axes[1, 1].imshow(grad_overlay, aspect='auto')
        axes[1, 1].set_title("RGB + Gradient Term", fontsize=12)
        axes[1, 1].axis('off')
        
        # RGB + 边缘项叠加
        axes[1, 2].imshow(rgb, aspect='auto')
        edge_overlay = np.zeros((height, width, 4), dtype=np.float32)
        edge_overlay[:, :, 0] = 1.0  # 红色
        edge_norm = (edge_term_img - edge_term_img.min()) / (edge_term_img.max() - edge_term_img.min() + 1e-6)
        edge_overlay[:, :, 3] = edge_norm * 0.5
        axes[1, 2].imshow(edge_overlay, aspect='auto')
        axes[1, 2].set_title("RGB + Edge Term", fontsize=12)
        axes[1, 2].axis('off')
    else:
        # 如果没有 RGB，显示统计信息
        for i in range(3):
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
