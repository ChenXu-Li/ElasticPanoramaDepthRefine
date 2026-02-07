"""
ElasticPanoramaDepthRefine 主入口
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

from data.io import load_depth, load_rgb, anchors_to_pixel_indices
from geometry.sphere import get_4_neighbors
from graph.weights import compute_anisotropic_weights
from graph.laplacian import build_weighted_laplacian, build_gradient_laplacian
from solver.elastic_solver import refine_depth
from utils.masks import detect_sky_mask, build_edge_mask_for_edges
from utils.visualization import save_depth, visualize_depth_comparison, visualize_anchors, depth_to_colormap, visualize_weight_terms, visualize_depth_change
from utils.pointcloud import depth_to_pointcloud_ply


def load_config(config_path: str | Path) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="ElasticPanoramaDepthRefine")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径"
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    paths = config["paths"]
    anchor_filter_config = config.get("anchor_filter", {"max_depth": 100.0})
    opt_config = config["optimization"]
    edge_config = config["edge"]
    sky_config = config["sky"]
    solver_config = config["solver"]
    output_config = config["output"]
    
    print("=" * 60)
    print("ElasticPanoramaDepthRefine")
    print("=" * 60)
    
    # 1. 读取数据
    print("\n[1/10] 读取数据...")
    depth_dap_raw = load_depth(paths["depth_dap"])
    rgb = load_rgb(paths.get("rgb"))
    
    # 处理 anchor 路径：如果以 logs/ 开头，则相对于项目根目录
    anchors_path = paths["anchors"]
    if anchors_path.startswith("logs/"):
        anchors_path = str(project_root / anchors_path)
    
    # 每次都重新生成 anchor 文件
    print(f"  🔧 生成 anchor 文件...")
    
    # 检查是否有自动生成配置
    anchor_gen_config = paths.get("anchor_generation")
    if anchor_gen_config is None:
        raise FileNotFoundError(
            f"未配置 anchor_generation。\n"
            f"请在 config.yaml 的 paths 部分添加 anchor_generation 配置。"
        )
    
    # 从深度图路径推断 pano_name
    depth_dap_path = Path(paths["depth_dap"])
    pano_name = depth_dap_path.stem
    
    # 获取图像尺寸
    height, width = depth_dap_raw.shape
    image_config = config.get("image", {})
    width = image_config.get("width", width)
    height = image_config.get("height", height)
    
    # 导入生成工具
    from utils.generate_anchors_from_ply import (
        project_ply_to_pano,
        generate_anchors_from_ref_depth
    )
    
    # 生成 anchor 文件
    fused_ply = Path(anchor_gen_config["fused_ply"])
    colmap_dir = Path(anchor_gen_config["colmap_dir"])
    camera_name = anchor_gen_config.get("camera_name", "pano_camera12")
    sample_rate = anchor_gen_config.get("sample_rate", 1.0)  # 默认使用所有有效像素
    depth_min = anchor_gen_config.get("depth_min", 0.1)
    depth_max = anchor_gen_config.get("depth_max", 1000.0)
    convention = anchor_gen_config.get("convention", "colmap_util")
    
    print(f"    从 fused.ply 生成 anchor: {fused_ply}")
    print(f"    COLMAP 目录: {colmap_dir}")
    print(f"    全景图名称: {pano_name}")
    print(f"    相机名称: {camera_name}")
    print(f"    图像尺寸: {width} x {height}")
    print(f"    采样率: {sample_rate} ({'使用所有有效像素' if sample_rate >= 1.0 else f'采样 {sample_rate*100:.1f}%'})")
    
    # 投影点云到全景图
    D_ref, M_ref = project_ply_to_pano(
        fused_ply,
        colmap_dir,
        pano_name,
        camera_name,
        width,
        height,
        depth_min=depth_min,
        depth_max=depth_max,
        convention=convention
    )
    
    # 生成 anchor 点
    anchors_raw = generate_anchors_from_ref_depth(
        D_ref,
        ref_mask=M_ref,
        width=width,
        height=height,
        convention=convention,
        sample_rate=sample_rate
    )
    
    # 保存 anchor 文件
    anchors_path_obj = Path(anchors_path)
    anchors_path_obj.parent.mkdir(parents=True, exist_ok=True)
    np.save(anchors_path, anchors_raw)
    print(f"  ✅ 已生成并保存 anchor 文件: {anchors_path}")
    print(f"    生成了 {len(anchors_raw):,} 个 anchor 点")
    
    # DAP 深度图在 0-1 范围，需要乘以缩放因子转换为米
    depth_scale = config.get("depth_scale", 100.0)
    depth_dap = depth_dap_raw * depth_scale
    
    height, width = depth_dap.shape
    print(f"  深度图尺寸: {height} x {width}")
    print(f"  深度缩放因子: {depth_scale} (DAP 0-1 → 米)")
    print(f"  Anchor 数量（原始）: {len(anchors_raw)}")
    print(f"  RGB 图像: {'已加载' if rgb is not None else '未提供'}")
    print(f"  说明: Anchor文件从参考深度图生成，参考深度图已应用 depth_min/depth_max 过滤")
    
    # 2. 转换 anchor 到像素索引（应用深度过滤）
    print("\n[2/10] 转换 anchor 坐标...")
    max_depth = anchor_filter_config.get("max_depth", 100.0)
    
    # 显示原始 anchor 深度统计
    # 注意：深度值定义为"到相机原点的欧式距离"（radial distance），与参考深度图一致
    anchor_depths_raw = anchors_raw[:, 2]
    print(f"  Anchor 深度范围: [{anchor_depths_raw.min():.2f}, {anchor_depths_raw.max():.2f}] 米")
    print(f"  说明: 深度值 = 到相机原点的欧式距离（radial distance）")
    
    # 过滤深度：只保留深度小于 max_depth 的点
    # 深度值 = ||point_camera|| = sqrt(x² + y² + z²)，即到球心的距离
    # 注意：虽然参考深度图生成时已应用过滤，但这里再次过滤以确保符合用户配置
    valid_depth_mask = anchor_depths_raw < max_depth
    anchors = anchors_raw[valid_depth_mask]
    
    if len(anchors) == 0:
        raise ValueError(f"所有 anchor 点都被深度过滤（max_depth={max_depth}米）剔除，请检查配置或数据")
    
    n_filtered = len(anchors_raw) - len(anchors)
    if n_filtered > 0:
        print(f"  深度过滤（< {max_depth}米）后: {len(anchors)} (剔除 {n_filtered} 个)")
        anchor_depths_filtered = anchors[:, 2]
        print(f"  过滤后深度范围: [{anchor_depths_filtered.min():.2f}, {anchor_depths_filtered.max():.2f}] 米")
    else:
        print(f"  深度过滤（< {max_depth}米）: 无点被剔除（所有点深度 < {max_depth}米）")
    
    u_anchors, v_anchors, anchor_depths = anchors_to_pixel_indices(
        anchors, width, height, convention="colmap_util"
    )
    anchor_indices = (v_anchors * width + u_anchors).astype(np.int32)
    print(f"  有效 anchor 像素: {len(anchor_indices)}")
    
    # 可视化 anchor 点（保存到 logs 目录）
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    anchor_viz_path = logs_dir / "anchor_visualization.png"
    visualize_anchors(anchor_indices, anchor_depths, width, height, anchor_viz_path, rgb=rgb, depth=depth_dap)
    print(f"  Anchor 可视化已保存: {anchor_viz_path}")
    
    # 2.5. 创建天空 mask（预测深度 >= 1.0 视为天空）
    # 注意：使用原始深度值（0-1 范围），>= 1.0 视为天空
    print("\n[2.5/10] 创建天空 mask...")
    sky_mask_depth = depth_dap_raw >= 1.0
    sky_pixel_count = np.sum(sky_mask_depth)
    sky_percentage = (sky_pixel_count / depth_dap.size) * 100.0
    print(f"  天空像素数量: {sky_pixel_count} / {depth_dap.size} ({sky_percentage:.2f}%)")
    
    # 可视化天空 mask
    sky_mask_viz_path = logs_dir / "sky_mask_visualization.png"
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 左图：RGB 图像叠加天空 mask（红色半透明）
    if rgb is not None:
        axes[0].imshow(rgb, aspect='auto')
        # 叠加天空 mask（红色半透明）
        sky_overlay = np.zeros((height, width, 4), dtype=np.float32)
        sky_overlay[:, :, 0] = 1.0  # 红色
        sky_overlay[:, :, 3] = sky_mask_depth.astype(np.float32) * 0.5  # 50% 透明度
        axes[0].imshow(sky_overlay, aspect='auto')
        axes[0].set_title(f"RGB with Sky Mask ({sky_pixel_count} pixels, {sky_percentage:.2f}%)", fontsize=14)
    else:
        # 如果没有 RGB，显示深度图叠加 mask
        axes[0].imshow(depth_to_colormap(depth_dap), aspect='auto')
        sky_overlay = np.zeros((height, width, 4), dtype=np.float32)
        sky_overlay[:, :, 0] = 1.0
        sky_overlay[:, :, 3] = sky_mask_depth.astype(np.float32) * 0.5
        axes[0].imshow(sky_overlay, aspect='auto')
        axes[0].set_title(f"Depth with Sky Mask ({sky_pixel_count} pixels, {sky_percentage:.2f}%)", fontsize=14)
    axes[0].axis('off')
    
    # 右图：纯 mask 可视化（白色=天空，黑色=非天空）
    axes[1].imshow(sky_mask_depth.astype(np.float32), cmap='gray', aspect='auto')
    axes[1].set_title("Sky Mask (White=Sky, Black=Non-Sky)", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(sky_mask_viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  天空 mask 可视化已保存: {sky_mask_viz_path}")
    
    # 3. 构建图结构
    print("\n[3/10] 构建图结构...")
    i_indices, j_indices, edge_types = get_4_neighbors(height, width)
    print(f"  边数量: {len(i_indices)}")
    
    # 4. 计算各向异性权重
    print("\n[4/10] 计算各向异性权重...")
    # 打印原始深度图统计（已转换为米）
    depth_valid = depth_dap[np.isfinite(depth_dap) & (depth_dap > 0)]
    if len(depth_valid) > 0:
        print(f"  深度图统计（已转换为米）:")
        print(f"    有效像素: {len(depth_valid)} / {depth_dap.size}")
        print(f"    深度范围: [{depth_valid.min():.4f}, {depth_valid.max():.4f}] 米")
        print(f"    均值: {depth_valid.mean():.4f} 米")
        print(f"    中位数: {np.median(depth_valid):.4f} 米")
    
    # 转换为 log 空间（深度已为米单位）
    log_depth_dap = np.log(np.maximum(depth_dap, 1e-6))
    
    # 检测边缘（只使用深度边缘，不使用 RGB 边缘）
    sky_mask_rgb = detect_sky_mask(rgb, sky_config["brightness_threshold"]) if sky_config["enable"] else None
    edge_mask = build_edge_mask_for_edges(
        height, width, i_indices, j_indices,
        rgb=None,  # 不使用 RGB 边缘
        log_depth=log_depth_dap if edge_config["use_depth_edge"] else None,
        sky_mask=sky_mask_rgb,
        rgb_edge_threshold=edge_config["rgb_edge_threshold"],
        depth_edge_threshold=edge_config["depth_edge_threshold"]
    )
    
    # 计算权重（使用基于深度的天空 mask，将涉及天空的边权重设为0）
    weights, grad_term, edge_term, grad_diff = compute_anisotropic_weights(
        log_depth_dap,
        i_indices,
        j_indices,
        opt_config["lambda_g"],
        opt_config["lambda_e"],
        edge_mask=edge_mask,
        sky_mask=sky_mask_depth,  # 使用基于深度的天空 mask
        return_terms=True  # 返回梯度项和边缘项用于可视化
    )
    zero_weight_count = np.sum(weights == 0.0)
    zero_weight_percentage = (zero_weight_count / len(weights)) * 100.0
    print(f"  权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  权重为0的边（天空区域）: {zero_weight_count} / {len(weights)} ({zero_weight_percentage:.2f}%)")
    print(f"  梯度项范围: [{grad_term.min():.4f}, {grad_term.max():.4f}]")
    print(f"  边缘项范围: [{edge_term.min():.4f}, {edge_term.max():.4f}]")
    print(f"  梯度差异范围: [{grad_diff.min():.4f}, {grad_diff.max():.4f}]")
    
    # 可视化梯度项和边缘项（可选）
    if output_config.get("visualize_weight_terms", False):
        print("\n[4.5/10] 可视化梯度项和边缘项...")
        weight_terms_viz_path = logs_dir / "weight_terms_visualization.png"
        visualize_weight_terms(
            grad_term, edge_term, grad_diff,
            i_indices, j_indices,
            height, width,
            weight_terms_viz_path,
            rgb=rgb
        )
        print(f"  权重项可视化已保存: {weight_terms_viz_path}")
    
    # 5. 构建 Laplacian 矩阵
    print("\n[5/10] 构建 Laplacian 矩阵...")
    L_elastic = build_weighted_laplacian(height, width, i_indices, j_indices, weights)
    # 梯度保持项也需要考虑天空 mask，避免在天空像素和物体边缘之间产生约束
    L_grad = build_gradient_laplacian(height, width, sky_mask=sky_mask_depth)
    print(f"  L_elastic 非零元素: {L_elastic.nnz}")
    print(f"  L_grad 非零元素: {L_grad.nnz}")
    
    # 6. 保存优化前的 log 深度图（热力图 PNG）
    print("\n[6/10] 保存优化前的 log 深度图（热力图）...")
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_depth_output = logs_dir / "log_depth_dap.png"
    
    # 计算显示范围（log 值可能是负数，所以只检查 isfinite）
    valid_mask = np.isfinite(log_depth_dap)
    if np.any(valid_mask):
        vmin = np.nanmin(log_depth_dap[valid_mask])
        vmax = np.nanmax(log_depth_dap[valid_mask])
        # 打印调试信息
        print(f"    Log 深度统计:")
        print(f"      有效像素: {np.sum(valid_mask)} / {log_depth_dap.size}")
        print(f"      范围: [{vmin:.4f}, {vmax:.4f}]")
        print(f"      均值: {np.nanmean(log_depth_dap[valid_mask]):.4f}")
        print(f"      中位数: {np.nanmedian(log_depth_dap[valid_mask]):.4f}")
    else:
        print("    ⚠️  警告: 没有有效的 log 深度值！")
        vmin = -10.0
        vmax = 10.0
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(log_depth_dap, cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto')
    ax.axis('off')
    ax.set_title("Log Depth (DAP)", fontsize=16, pad=10)
    
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Log Depth", rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(log_depth_output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Log 深度图（热力图）已保存: {log_depth_output}")
    
    # 6.5. 保存带 anchor 点的 log 深度图
    print("\n[7/10] 保存带 anchor 点的 log 深度图（热力图）...")
    log_depth_with_anchors = log_depth_dap.copy()
    
    # 将 anchor 点转换为 log 空间并替换对应位置
    anchor_log_depths = np.log(np.maximum(anchor_depths, 1e-6))
    log_depth_with_anchors.flat[anchor_indices] = anchor_log_depths
    
    # 计算显示范围（包含 anchor 点）
    valid_mask_anchors = np.isfinite(log_depth_with_anchors)
    if np.any(valid_mask_anchors):
        vmin_anchors = np.nanmin(log_depth_with_anchors[valid_mask_anchors])
        vmax_anchors = np.nanmax(log_depth_with_anchors[valid_mask_anchors])
        print(f"    带 anchor 的 Log 深度范围: [{vmin_anchors:.4f}, {vmax_anchors:.4f}]")
        print(f"    Anchor 点数量: {len(anchor_indices)}")
    else:
        vmin_anchors = vmin
        vmax_anchors = vmax
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(log_depth_with_anchors, cmap='turbo', vmin=vmin_anchors, vmax=vmax_anchors, aspect='auto')
    ax.axis('off')
    ax.set_title(f"Log Depth (DAP) with {len(anchor_indices)} Anchors", fontsize=16, pad=10)
    
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Log Depth", rotation=270, labelpad=20)
    
    plt.tight_layout()
    log_depth_anchors_output = logs_dir / "log_depth_dap_with_anchors.png"
    plt.savefig(log_depth_anchors_output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  带 anchor 点的 Log 深度图（热力图）已保存: {log_depth_anchors_output}")
    
    # 8. 求解
    print("\n[8/10] 求解线性系统...")
    depth_refined = refine_depth(
        depth_dap,
        anchor_indices,
        anchor_depths,
        L_elastic,
        L_grad,
        opt_config["lambda_grad"],
        opt_config["alpha_anchor"],
        method=solver_config["method"],
        max_iter=solver_config["max_iter"],
        tol=solver_config["tol"]
    )
    print("  求解完成")
    
    # 9. 保存结果
    print("\n[9/10] 保存结果...")
    output_dir = Path(paths["output_dir"])
    logs_dir = project_root / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 中间结果保存到 logs 目录
    depth_output = logs_dir / "depth_refined.npy"
    save_depth(depth_refined, depth_output, format=output_config["format"])
    print(f"  深度图已保存（中间结果）: {depth_output}")
    
    # 保存可视化到 logs 目录
    if output_config["save_visualization"]:
        viz_output = logs_dir / "depth_comparison.png"
        visualize_depth_comparison(depth_dap, depth_refined, viz_output, rgb)
        print(f"  可视化已保存（中间结果）: {viz_output}")
        
        # 保存深度变化热力图（log 空间和真实空间）
        print("\n[9.5/10] 保存深度变化热力图...")
        depth_change_log_output = logs_dir / "depth_change_log_space.png"
        depth_change_linear_output = logs_dir / "depth_change_linear_space.png"
        visualize_depth_change(
            depth_dap, depth_refined,
            depth_change_log_output, depth_change_linear_output,
            rgb=rgb
        )
        print(f"  Log 空间深度变化热力图已保存: {depth_change_log_output}")
        print(f"  真实空间深度变化热力图已保存: {depth_change_linear_output}")
    
    # 10. 生成并保存 PLY 点云到输出目录
    print("\n[10/10] 生成点云...")
    # PLY 文件名与输入深度图名称一致
    depth_dap_path = Path(paths["depth_dap"])
    ply_filename = depth_dap_path.stem + ".ply"  # 使用深度图文件名（不含扩展名）+ .ply
    ply_output = output_dir / ply_filename
    # 使用 DAP 约定，与输入的 DAP 深度图保持一致
    depth_to_pointcloud_ply(depth_refined, rgb, ply_output, convention="dap")
    
    print("\n" + "=" * 60)
    print("完成！")
    print(f"  最终输出（PLY点云）: {ply_output}")
    print(f"  中间结果: {logs_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
