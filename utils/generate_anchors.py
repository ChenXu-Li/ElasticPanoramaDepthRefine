"""
从参考深度图生成 anchor 点
"""
import numpy as np
from pathlib import Path
from data.io import pixel_to_theta_phi


def generate_anchors_from_ref_depth(
    ref_depth_path: str | Path,
    ref_mask_path: str | Path = None,
    output_path: str | Path = None,
    width: int = None,
    height: int = None,
    convention: str = "colmap_util",
    sample_rate: float = 1.0
) -> np.ndarray:
    """
    从参考深度图生成 anchor 点
    
    Args:
        ref_depth_path: 参考深度图路径 (.npy)
        ref_mask_path: 可选，anchor mask 路径 (.npy 或 .png)
        output_path: 输出 anchor 文件路径 (.npy)
        width: 图像宽度，如果为 None 则从深度图读取
        height: 图像高度，如果为 None 则从深度图读取
        convention: 坐标约定
        sample_rate: 采样率 [0, 1]，1.0 表示使用所有有效像素
        
    Returns:
        anchors: (K, 3) float32，每行为 (theta, phi, depth)
    """
    ref_depth = np.load(ref_depth_path).astype(np.float32)
    height, width = ref_depth.shape
    
    # 加载 mask（如果有）
    if ref_mask_path is not None and Path(ref_mask_path).exists():
        mask = np.load(ref_mask_path) if ref_mask_path.suffix == '.npy' else None
        if mask is None:
            import imageio
            mask = imageio.imread(ref_mask_path)
            if mask.ndim == 3:
                mask = mask[:, :, 0] > 128
            else:
                mask = mask > 128
    else:
        # 使用有效深度作为 mask
        mask = np.isfinite(ref_depth) & (ref_depth > 0)
    
    # 获取有效像素位置
    v_valid, u_valid = np.where(mask)
    depths_valid = ref_depth[v_valid, u_valid]
    
    # 采样
    if sample_rate < 1.0:
        n_samples = int(len(v_valid) * sample_rate)
        indices = np.random.choice(len(v_valid), n_samples, replace=False)
        v_valid = v_valid[indices]
        u_valid = u_valid[indices]
        depths_valid = depths_valid[indices]
    
    # 转换为 theta, phi
    theta, phi = pixel_to_theta_phi(u_valid, v_valid, width, height, convention)
    
    # 构建 anchor 数组
    anchors = np.stack([theta, phi, depths_valid], axis=1).astype(np.float32)
    
    # 保存
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, anchors)
        print(f"✅ 已保存 {len(anchors)} 个 anchor 点到: {output_path}")
    
    return anchors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从参考深度图生成 anchor 点")
    parser.add_argument("--ref_depth", type=str, required=True, help="参考深度图路径")
    parser.add_argument("--ref_mask", type=str, default=None, help="参考 mask 路径（可选）")
    parser.add_argument("--output", type=str, required=True, help="输出 anchor 文件路径")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="采样率 [0, 1]")
    parser.add_argument("--convention", type=str, default="colmap_util", help="坐标约定")
    
    args = parser.parse_args()
    
    anchors = generate_anchors_from_ref_depth(
        args.ref_depth,
        args.ref_mask,
        args.output,
        convention=args.convention,
        sample_rate=args.sample_rate
    )
    
    print(f"生成了 {len(anchors)} 个 anchor 点")
