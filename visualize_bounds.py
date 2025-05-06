"""
可视化Franka Panda机器人末端执行器的活动范围
使用open3d库显示采样点云
"""
import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

def visualize_bounds(positions_file, bounds_file=None, sample_ratio=1.0):
    """
    可视化机器人末端执行器的活动范围
    
    参数:
        positions_file: 采样点位置文件(.npy格式)
        bounds_file: 边界文件(可选，.txt格式)
        sample_ratio: 采样比例，用于减少点云数量，范围[0,1]
    """
    # 加载采样点
    ee_positions = np.load(positions_file)
    print(f"加载了 {len(ee_positions)} 个采样点")
    
    # 如果需要，减少点云数量
    if sample_ratio < 1.0:
        sample_size = max(1, int(len(ee_positions) * sample_ratio))
        indices = np.random.choice(len(ee_positions), sample_size, replace=False)
        ee_positions = ee_positions[indices]
        print(f"采样后保留 {len(ee_positions)} 个点")
    
    # 计算边界
    if bounds_file:
        bounds = np.loadtxt(bounds_file)
        bounds_min = bounds[0]
        bounds_max = bounds[1]
    else:
        bounds_min = np.min(ee_positions, axis=0)
        bounds_max = np.max(ee_positions, axis=0)
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ee_positions)
    
    # 使用彩虹色图为点着色，根据z坐标
    color_map = matplotlib.colormaps["viridis"]
    z_values = ee_positions[:, 2]
    z_normalized = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
    colors = [color_map(z) for z in z_normalized]
    colors = np.array([color[:3] for color in colors])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建边界框
    points = [
        [bounds_min[0], bounds_min[1], bounds_min[2]],
        [bounds_min[0], bounds_min[1], bounds_max[2]],
        [bounds_min[0], bounds_max[1], bounds_min[2]],
        [bounds_min[0], bounds_max[1], bounds_max[2]],
        [bounds_max[0], bounds_min[1], bounds_min[2]],
        [bounds_max[0], bounds_min[1], bounds_max[2]],
        [bounds_max[0], bounds_max[1], bounds_min[2]],
        [bounds_max[0], bounds_max[1], bounds_max[2]]
    ]
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    colors = [[1, 0, 0] for _ in range(len(lines))]  # 红色边界框
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # 打印边界信息
    print("\n==== 末端执行器活动范围 ====")
    print(f"最小坐标: {bounds_min}")
    print(f"最大坐标: {bounds_max}")
    print(f"尺寸: {bounds_max - bounds_min}")
    
    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    
    # 显示点云和边界框
    print("\n按住鼠标左键旋转视图，右键平移，滚轮缩放")
    print("按 'h' 查看更多控制选项")
    o3d.visualization.draw_geometries([pcd, line_set, coord_frame])
    
    return bounds_min, bounds_max

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化机器人末端执行器的活动范围')
    parser.add_argument('--positions', type=str, default='ee_positions_isaacgym.npy',
                        help='采样点位置文件(.npy格式)')
    parser.add_argument('--bounds', type=str, default='robot_bounds_isaacgym.txt',
                        help='边界文件(.txt格式)')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='采样比例，用于减少点云数量，范围[0,1]')
    args = parser.parse_args()
    
    visualize_bounds(args.positions, args.bounds, args.sample_ratio) 