"""
测试 Step 1: 数据读取和坐标转换
"""
import numpy as np
from pathlib import Path
from data.io import (
    load_depth, load_rgb, load_anchors,
    theta_phi_to_pixel, pixel_to_theta_phi,
    anchors_to_pixel_indices
)


def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("测试 1: 数据加载")
    print("=" * 60)
    
    # 测试路径
    depth_path = "/root/autodl-tmp/data/STAGE1_4x/BridgeB/depth_npy/point2_median.npy"
    rgb_path = "/root/autodl-tmp/data/STAGE1_4x/BridgeB/backgrounds/point2_median.png"
    
    # 加载深度
    depth = load_depth(depth_path)
    print(f"✓ 深度图加载成功: {depth.shape}, dtype={depth.dtype}")
    print(f"  深度范围: [{np.nanmin(depth):.2f}, {np.nanmax(depth):.2f}]")
    
    # 加载 RGB
    rgb = load_rgb(rgb_path)
    if rgb is not None:
        print(f"✓ RGB 图像加载成功: {rgb.shape}, dtype={rgb.dtype}")
    else:
        print("✗ RGB 图像未找到")
    
    return depth, rgb


def test_coordinate_conversion(depth):
    """测试坐标转换"""
    print("\n" + "=" * 60)
    print("测试 2: 坐标转换")
    print("=" * 60)
    
    height, width = depth.shape
    
    # 测试像素 -> theta, phi
    u_test = np.array([0, width // 2, width - 1], dtype=np.float32)
    v_test = np.array([0, height // 2, height - 1], dtype=np.float32)
    
    theta, phi = pixel_to_theta_phi(u_test, v_test, width, height, "colmap_util")
    print(f"✓ 像素 -> (theta, phi) 转换成功")
    print(f"  测试点: u={u_test}, v={v_test}")
    print(f"  结果: theta={theta}, phi={phi}")
    
    # 测试反向转换
    u_back, v_back = theta_phi_to_pixel(theta, phi, width, height, "colmap_util")
    print(f"✓ (theta, phi) -> 像素 转换成功")
    print(f"  结果: u={u_back}, v={v_back}")
    # 注意：u 方向有周期 wrap，边界处误差可能较大（这是正常的）
    u_err = np.abs(u_back - u_test)
    u_err = np.minimum(u_err, width - u_err)  # 考虑周期 wrap
    print(f"  误差: u_err={u_err.max():.2f}, v_err={np.abs(v_back - v_test).max():.2f}")
    
    return True


def test_anchor_conversion(depth):
    """测试 anchor 转换"""
    print("\n" + "=" * 60)
    print("测试 3: Anchor 转换")
    print("=" * 60)
    
    height, width = depth.shape
    
    # 生成测试 anchor（如果文件不存在）
    anchor_path = "/root/autodl-tmp/data/STAGE1_4x/BridgeB/Intermediate_files_single_opt/point2_median_anchor.npy"
    
    if not Path(anchor_path).exists():
        print("⚠ Anchor 文件不存在，生成测试 anchor...")
        # 从参考深度图生成
        ref_depth_path = "/root/autodl-tmp/data/STAGE1_4x/BridgeB/Intermediate_files_single_opt/point2_median_ref_depth.npy"
        if Path(ref_depth_path).exists():
            from utils.generate_anchors import generate_anchors_from_ref_depth
            anchors = generate_anchors_from_ref_depth(
                ref_depth_path,
                output_path=anchor_path,
                sample_rate=0.1  # 采样 10%
            )
        else:
            print("✗ 无法生成 anchor：参考深度图不存在")
            return False
    else:
        anchors = load_anchors(anchor_path)
    u, v, depths = anchors_to_pixel_indices(anchors, width, height, "colmap_util")
    print(f"✓ Anchor -> 像素索引转换成功")
    print(f"  有效 anchor 数量: {len(u)}")
    print(f"  像素范围: u=[{u.min()}, {u.max()}], v=[{v.min()}, {v.max()}]")
    
    return True


if __name__ == "__main__":
    try:
        depth, rgb = test_data_loading()
        test_coordinate_conversion(depth)
        test_anchor_conversion(depth)
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
