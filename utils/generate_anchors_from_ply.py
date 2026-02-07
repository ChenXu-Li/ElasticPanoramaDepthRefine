"""
从 fused.ply 直接生成 anchor 文件（独立工具，不依赖其他项目）
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pycolmap
from plyfile import PlyData

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.io import pixel_to_theta_phi


def load_ply_xyz(ply_path: Path) -> np.ndarray:
    """读取 PLY 点云（vertex: x,y,z），返回世界坐标点 (N, 3) float32"""
    ply_path = Path(ply_path)
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise RuntimeError(f"PLY 文件中缺少 'vertex' 元素: {ply_path}")
    vertex = ply["vertex"]
    if not all(k in vertex.data.dtype.names for k in ("x", "y", "z")):
        raise RuntimeError(f"PLY 须包含 x, y, z 顶点属性: {ply_path}")
    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    return positions


def world_to_cam(points_world: np.ndarray, cam_from_world: pycolmap.Rigid3d) -> np.ndarray:
    """将世界坐标点转换到相机坐标系"""
    # cam_from_world 是 Rigid3d，表示从世界到相机的变换
    # points_world: (N, 3)
    # 返回: (N, 3) 相机坐标系点
    R = cam_from_world.rotation.matrix()  # (3, 3)
    t = cam_from_world.translation  # (3,)
    
    # 变换: points_cam = R @ points_world.T + t
    points_cam_T = R @ points_world.T + t[:, None]  # (3, N)
    return points_cam_T.T  # (N, 3)


def cam_points_to_equirectangular(
    points_cam: np.ndarray,
    width: int,
    height: int,
    depth_min: float = 0.1,
    depth_max: float = 1000.0,
    convention: str = "colmap_util"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将相机坐标系下的3D点投影到等轴柱状图（equirectangular）上
    
    Args:
        points_cam: (N, 3) 相机坐标系点云，单位为米
        width: 输出图像宽度
        height: 输出图像高度
        depth_min: 最小深度阈值（米）
        depth_max: 最大深度阈值（米）
        convention: equirect 参数化约定
        
    Returns:
        u: (M,) 像素坐标u（列索引），M为有效点数量
        v: (M,) 像素坐标v（行索引）
        depth: (M,) 深度值（相机坐标系欧式距离）
        valid_mask: (N,) 布尔数组，表示哪些点是有效的
    """
    if width != 2 * height:
        raise ValueError("仅支持360°等轴柱状全景（width应为height的2倍）")
    
    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        raise ValueError("points_cam应为(N, 3)")
    
    if len(points_cam) == 0:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=bool),
        )
    
    # 深度 = 到相机原点的欧式距离（radial distance）
    depths = np.linalg.norm(points_cam, axis=1).astype(np.float32)
    
    # 过滤：零/负深度点、深度范围、数值异常
    valid = depths > depth_min
    valid &= depths < depth_max
    valid &= np.isfinite(depths)
    valid &= np.all(np.isfinite(points_cam), axis=1)
    
    if not np.any(valid):
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.zeros(len(points_cam), dtype=bool),
        )
    
    pts = points_cam[valid]
    d = depths[valid]
    
    # 单位方向向量
    dirs = pts / d[:, None]  # (M, 3)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    if convention == "colmap_util":
        # colmap_util 约定：yaw = atan2(x, z) [-π, π], pitch = -atan2(y, sqrt(x²+z²)) [-π/2, π/2]
        yaw = np.arctan2(x, z)  # [-pi, pi]
        pitch = -np.arctan2(y, np.sqrt(x * x + z * z))  # [-pi/2, pi/2]
        u = (1.0 + yaw / np.pi) * 0.5  # [0, 1]
        v = (1.0 - pitch * 2.0 / np.pi) * 0.5  # [0, 1]
    elif convention == "dap":
        # DAP 约定：theta = 方位角 [0, 2π), phi = 极角 [0, π]
        phi = np.arccos(np.clip(z, -1.0, 1.0))  # [0, pi]
        theta = np.arctan2(y, x)  # [-pi, pi]
        theta = np.mod(theta + 2.0 * np.pi, 2.0 * np.pi)  # [0, 2*pi)
        u = 1.0 - theta / (2.0 * np.pi)  # [0, 1]
        v = phi / np.pi  # [0, 1]
    else:
        raise ValueError(f"Unknown equirect convention: {convention}")

    # 映射到像素坐标
    u_pix = u * width
    v_pix = v * height
    
    # 边界处理：确保在有效范围内
    u_pix = np.clip(u_pix, 0, width - 1e-6)
    v_pix = np.clip(v_pix, 0, height - 1e-6)
    
    # 构建完整输出（包括无效点）
    u_full = np.zeros(len(points_cam), dtype=np.float32)
    v_full = np.zeros(len(points_cam), dtype=np.float32)
    d_full = np.zeros(len(points_cam), dtype=np.float32)
    
    u_full[valid] = u_pix
    v_full[valid] = v_pix
    d_full[valid] = d
    
    return u_full[valid], v_full[valid], d_full[valid], valid


def z_buffer_projection(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-buffer 投影：选择每个像素的最近点
    
    Args:
        u: (N,) 像素u坐标
        v: (N,) 像素v坐标
        depth: (N,) 深度值
        height: 图像高度
        width: 图像宽度
        
    Returns:
        depth_map: (H, W) float32，参考深度图，未投影处为NaN
        mask: (H, W) bool，有效投影点mask
    """
    depth_map = np.full((height, width), np.nan, dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    
    if len(u) == 0:
        return depth_map, mask
    
    # 转换为整数像素坐标
    u_int = np.clip(np.floor(u).astype(np.int32), 0, width - 1)
    v_int = np.clip(np.floor(v).astype(np.int32), 0, height - 1)
    
    # Z-buffer：选择最近点
    for i in range(len(u_int)):
        ui, vi, d = u_int[i], v_int[i], depth[i]
        if np.isnan(depth_map[vi, ui]) or d < depth_map[vi, ui]:
            depth_map[vi, ui] = d
            mask[vi, ui] = True
    
    return depth_map, mask


def build_pano_to_frame_mapping(recon: pycolmap.Reconstruction) -> dict:
    """建立 pano_name 到 frame_id 的映射"""
    pano_to_frame = {}
    for img_id, img in recon.images.items():
        if img.frame_id not in recon.frames:
            continue
        
        # 提取全景图名称（从路径中提取文件名，去掉扩展名）
        img_name = img.name
        if '/' in img_name:
            img_pano_name = Path(img_name.split('/')[-1]).stem
        else:
            img_pano_name = Path(img_name).stem
        
        # 如果有多个同名图像，选择有pose的frame
        if img_pano_name not in pano_to_frame:
            pano_to_frame[img_pano_name] = img.frame_id
        else:
            current_frame = recon.frames[img.frame_id]
            existing_frame = recon.frames[pano_to_frame[img_pano_name]]
            if current_frame.has_pose() and not existing_frame.has_pose():
                pano_to_frame[img_pano_name] = img.frame_id
    
    return pano_to_frame


def get_cam_from_world_for_frame(
    recon: pycolmap.Reconstruction,
    frame_id: int,
    camera_name_substr: str,
) -> pycolmap.Rigid3d | None:
    """
    获取指定 frame 中指定相机的 cam_from_world 变换
    
    Args:
        recon: COLMAP 重建结果
        frame_id: frame ID
        camera_name_substr: 相机名称子串（如 "pano_camera12"）
        
    Returns:
        cam_from_world: pycolmap.Rigid3d，如果未找到则返回 None
    """
    frame = recon.frames[frame_id]
    if not frame.has_pose():
        return None
    
    # 查找该 frame 中包含指定相机名称的图像
    for img_id, img in recon.images.items():
        if img.frame_id == frame_id and camera_name_substr in img.name:
            # 获取 cam_from_world
            if hasattr(img, 'cam_from_world'):
                cam_from_world = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
                return cam_from_world
    
    return None


def project_ply_to_pano(
    ply_path: Path,
    colmap_dir: Path,
    pano_name: str,
    camera_name: str,
    width: int,
    height: int,
    depth_min: float = 0.1,
    depth_max: float = 1000.0,
    convention: str = "colmap_util"
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 fused.ply 点云投影到全景图，生成参考深度图
    
    Args:
        ply_path: fused.ply 文件路径
        colmap_dir: COLMAP 重建目录
        pano_name: 全景图名称（不含扩展名）
        camera_name: 相机名称子串（如 "pano_camera12"）
        width: 全景图宽度
        height: 全景图高度
        depth_min: 最小深度阈值（米）
        depth_max: 最大深度阈值（米）
        convention: 坐标约定
        
    Returns:
        D_ref: (H, W) float32，参考深度图
        M_ref: (H, W) bool，anchor mask
    """
    # 加载 COLMAP 重建
    print(f"  加载 COLMAP 重建: {colmap_dir}")
    recon = pycolmap.Reconstruction(str(colmap_dir))
    
    # 建立 pano_name 到 frame_id 的映射
    print(f"  建立全景图到 frame 的映射...")
    pano_to_frame = build_pano_to_frame_mapping(recon)
    
    if pano_name not in pano_to_frame:
        raise ValueError(f"未找到全景图 {pano_name} 对应的 frame")
    
    frame_id = pano_to_frame[pano_name]
    frame = recon.frames[frame_id]
    
    if not frame.has_pose():
        raise ValueError(f"Frame {frame_id} 没有有效 pose")
    
    # 获取 cam_from_world
    print(f"  查找相机 {camera_name}...")
    cam_from_world = get_cam_from_world_for_frame(recon, frame_id, camera_name)
    if cam_from_world is None:
        raise ValueError(f"在 frame {frame_id} 中未找到包含 '{camera_name}' 的图像")
    
    # 读取点云
    print(f"  读取点云: {ply_path}")
    points_world = load_ply_xyz(ply_path)
    print(f"    点云数量: {len(points_world):,}")
    
    if len(points_world) == 0:
        D_ref = np.full((height, width), np.nan, dtype=np.float32)
        M_ref = np.zeros((height, width), dtype=bool)
        return D_ref, M_ref
    
    # 世界坐标 → 相机坐标
    print(f"  转换到相机坐标系...")
    points_camera = world_to_cam(points_world, cam_from_world)
    
    # 投影到等轴柱状图
    print(f"  投影到等轴柱状图 ({width}x{height})...")
    u, v, depth, valid_mask = cam_points_to_equirectangular(
        points_camera,
        width=width,
        height=height,
        depth_min=depth_min,
        depth_max=depth_max,
        convention=convention
    )
    
    print(f"    有效投影点: {len(u):,} / {len(points_world):,}")
    
    # Z-buffer 选择最近点
    print(f"  Z-buffer 选择最近点...")
    D_ref, M_ref = z_buffer_projection(u, v, depth, height, width)
    
    valid_pixels = np.sum(M_ref)
    print(f"    有效像素: {valid_pixels:,} / {height * width:,} ({100 * valid_pixels / (height * width):.2f}%)")
    
    return D_ref, M_ref


def generate_anchors_from_ref_depth(
    ref_depth: np.ndarray,
    ref_mask: np.ndarray | None = None,
    width: int = None,
    height: int = None,
    convention: str = "colmap_util",
    sample_rate: float = 1.0
) -> np.ndarray:
    """
    从参考深度图生成 anchor 点
    
    Args:
        ref_depth: (H, W) 参考深度图
        ref_mask: (H, W) 可选，anchor mask
        width: 图像宽度，如果为 None 则从深度图读取
        height: 图像高度，如果为 None 则从深度图读取
        convention: 坐标约定
        sample_rate: 采样率 [0, 1]，1.0 表示使用所有有效像素
        
    Returns:
        anchors: (K, 3) float32，每行为 (theta, phi, depth)
    """
    if width is None or height is None:
        height, width = ref_depth.shape
    
    # 加载 mask（如果有）
    if ref_mask is not None:
        mask = ref_mask
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
    
    return anchors


def main():
    parser = argparse.ArgumentParser(
        description="从 fused.ply 直接生成 anchor 文件（独立工具）"
    )
    parser.add_argument("--ply", type=str, required=True, help="fused.ply 文件路径")
    parser.add_argument("--colmap_dir", type=str, required=True, help="COLMAP 重建目录")
    parser.add_argument("--pano_name", type=str, required=True, help="全景图名称（不含扩展名）")
    parser.add_argument("--camera_name", type=str, default="pano_camera12", help="相机名称子串")
    parser.add_argument("--output", type=str, required=True, help="输出 anchor 文件路径")
    parser.add_argument("--width", type=int, default=1920, help="全景图宽度")
    parser.add_argument("--height", type=int, default=960, help="全景图高度")
    parser.add_argument("--depth_min", type=float, default=0.1, help="最小深度阈值（米）")
    parser.add_argument("--depth_max", type=float, default=1000.0, help="最大深度阈值（米）")
    parser.add_argument("--sample_rate", type=float, default=0.1, help="采样率 [0, 1]")
    parser.add_argument("--convention", type=str, default="colmap_util", choices=["colmap_util", "dap"], help="坐标约定")
    parser.add_argument("--save_ref_depth", type=str, default=None, help="可选：保存参考深度图路径")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("从 fused.ply 生成 anchor 文件")
    print("=" * 60)
    print(f"PLY 文件: {args.ply}")
    print(f"COLMAP 目录: {args.colmap_dir}")
    print(f"全景图名称: {args.pano_name}")
    print(f"相机名称: {args.camera_name}")
    print(f"输出文件: {args.output}")
    print(f"图像尺寸: {args.width} x {args.height}")
    print("=" * 60)
    print()
    
    # 投影点云到全景图
    D_ref, M_ref = project_ply_to_pano(
        Path(args.ply),
        Path(args.colmap_dir),
        args.pano_name,
        args.camera_name,
        args.width,
        args.height,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        convention=args.convention
    )
    
    # 保存参考深度图（可选）
    if args.save_ref_depth:
        print(f"\n保存参考深度图: {args.save_ref_depth}")
        Path(args.save_ref_depth).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_ref_depth, D_ref)
    
    # 生成 anchor 点
    print(f"\n生成 anchor 点（采样率: {args.sample_rate}）...")
    anchors = generate_anchors_from_ref_depth(
        D_ref,
        ref_mask=M_ref,
        width=args.width,
        height=args.height,
        convention=args.convention,
        sample_rate=args.sample_rate
    )
    
    # 保存 anchor 文件
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, anchors)
    
    print(f"\n✅ 完成！")
    print(f"  生成了 {len(anchors):,} 个 anchor 点")
    print(f"  已保存到: {output_path}")


if __name__ == "__main__":
    main()
