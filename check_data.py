import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import shutil

plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
plt.rcParams['axes.unicode_minus'] = False

data_dir = './data/sensor'
output_dir = './check'

# 创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def check_segmentation():
    """检查分割图"""
    seg = np.load(os.path.join(data_dir, 'actor_mask.npy'))
    unique_seg = np.unique(seg)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title("彩色图像")
    color_img = Image.open(os.path.join(data_dir, 'color.png'))
    plt.imshow(color_img)
    
    plt.subplot(122)
    plt.title("分割图")
    seg_vis = np.zeros((*seg.shape, 3), dtype=np.uint8)
    colors = {
        0: [0, 0, 0],         # 背景 - 黑色
        1: [255, 255, 255],   # 机器人 - 白色
        101: [255, 0, 0],     # 第一个盒子 - 红色
        102: [0, 255, 0]      # 第二个盒子 - 绿色
    }
    
    for label, color in colors.items():
        if label in unique_seg:
            mask = seg == label
            seg_vis[mask] = color
    
    plt.imshow(seg_vis)
    plt.savefig(os.path.join(output_dir, 'segmentation_check.png'))

def check_seg_rgb_correspondence():
    """检查分割图与RGB图像对应关系"""
    seg = np.load(os.path.join(data_dir, 'actor_mask.npy'))
    rgb = np.array(Image.open(os.path.join(data_dir, 'color.png')))
    
    unique_seg = np.unique(seg)
    expected_colors = {
        0: None,  # 背景
        1: [255, 255, 255],  # 机器人
        101: [255, 0, 0],    # 红盒子
        102: [0, 255, 0]     # 绿盒子
    }
    
    plt.figure(figsize=(12, 10))
    
    # 原始RGB图像
    plt.subplot(221)
    plt.title("原始RGB图像")
    plt.imshow(rgb)
    
    # 分割图
    plt.subplot(222)
    plt.title("分割图")
    seg_vis = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for label, color in expected_colors.items():
        if label in unique_seg and color is not None:
            mask = seg == label
            seg_vis[mask] = color
    plt.imshow(seg_vis)
    
    # 分割边界叠加在RGB上
    plt.subplot(223)
    plt.title("分割边界叠加在RGB上")
    overlay = rgb.copy()
    
    for label in unique_seg:
        if label == 0:
            continue
        mask = (seg == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boundary_color = expected_colors.get(label, [255, 255, 0])
        cv2.drawContours(overlay, contours, -1, boundary_color, 2)
    
    plt.imshow(overlay)
    
    # 半透明分割叠加在RGB上
    plt.subplot(224)
    plt.title("半透明分割叠加在RGB上")
    blend = rgb.copy()
    
    for label in unique_seg:
        if label == 0:
            continue
        mask = seg == label
        color = np.array(expected_colors.get(label, [255, 255, 0]))
        color_mask = np.zeros_like(rgb)
        color_mask[mask] = color
        blend = cv2.addWeighted(blend, 1.0, color_mask, 0.4, 0)
    
    plt.imshow(blend)
    plt.savefig(os.path.join(output_dir, 'segmentation_rgb_correspondence.png'))

def check_pointcloud():
    """检查点云"""
    points = np.load(os.path.join(data_dir, 'points_world.npy'))
    seg = np.load(os.path.join(data_dir, 'actor_mask.npy'))
    
    nan_count = np.isnan(points).sum()
    print(f"点云包含 {nan_count} 个NaN值!")
    
    z_values = points[:, :, 2].flatten()
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(221)
    plt.title("彩色图像")
    color_img = Image.open(os.path.join(data_dir, 'color.png'))
    plt.imshow(color_img)
    
    plt.subplot(222)
    plt.title("深度图")
    depth_img = Image.open(os.path.join(data_dir, 'depth.png'))
    plt.imshow(depth_img, cmap='gray')
    
    plt.subplot(223)
    plt.title("点云Z值热图")
    z_map = points[:, :, 2]
    plt.imshow(z_map, cmap='viridis')
    plt.colorbar(label='Z值 (深度)')
    
    plt.subplot(224)
    plt.title("点云Z值分布")
    plt.hist(z_values, bins=50)
    plt.xlabel('Z值')
    plt.ylabel('频率')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pointcloud_check.png'))
    
    # 添加点云密度图
    plt.figure(figsize=(10, 8))
    plt.title("点云密度图")
    xy_density = np.ones_like(seg)
    plt.imshow(xy_density, cmap='hot', alpha=0.7)
    plt.colorbar(label='点密度')
    plt.savefig(os.path.join(output_dir, 'pointcloud_density.png'))
    
    # 点云XY平面投影
    plt.figure(figsize=(10, 8))
    plt.title("点云XY平面投影")
    points_2d = points.reshape(-1, 3)
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c=points_2d[:, 2], cmap='viridis', s=0.5, alpha=0.5)
    plt.colorbar(label='Z值')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.savefig(os.path.join(output_dir, 'pointcloud_xy_projection.png'))
    
    # 点云截面图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title("点云X截面")
    mid_x = points.shape[1] // 2
    x_slice = points[:, mid_x, :]
    plt.scatter(x_slice[:, 1], x_slice[:, 2], s=1)
    plt.xlabel('Y坐标')
    plt.ylabel('Z坐标')
    
    plt.subplot(132)
    plt.title("点云Y截面")
    mid_y = points.shape[0] // 2
    y_slice = points[mid_y, :, :]
    plt.scatter(y_slice[:, 0], y_slice[:, 2], s=1)
    plt.xlabel('X坐标')
    plt.ylabel('Z坐标')
    
    plt.subplot(133)
    plt.title("点云Z直方图")
    plt.hist(z_values, bins=100, alpha=0.7)
    plt.xlabel('Z值')
    plt.ylabel('频率')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pointcloud_slices.png'))

def visualize_3d_pointcloud():
    """3D点云可视化"""
    points = np.load(os.path.join(data_dir, 'points_world.npy'))
    seg = np.load(os.path.join(data_dir, 'actor_mask.npy'))
    
    # 将点云重塑为2D数组
    points_reshaped = points.reshape(-1, 3)
    seg_reshaped = seg.flatten()
    
    # 计算mask为101和102的点的平均坐标
    mask_101 = seg_reshaped == 101
    mask_102 = seg_reshaped == 102
    
    if np.sum(mask_101) > 0:
        avg_points_101 = np.mean(points_reshaped[mask_101], axis=0)
        print(f"Mask 101 (红盒子)的平均坐标: X={avg_points_101[0]:.4f}, Y={avg_points_101[1]:.4f}, Z={avg_points_101[2]:.4f}")
    else:
        print("没有找到Mask 101 (红盒子)的点")
    
    if np.sum(mask_102) > 0:
        avg_points_102 = np.mean(points_reshaped[mask_102], axis=0)
        print(f"Mask 102 (绿盒子)的平均坐标: X={avg_points_102[0]:.4f}, Y={avg_points_102[1]:.4f}, Z={avg_points_102[2]:.4f}")
    else:
        print("没有找到Mask 102 (绿盒子)的点")
    
    # 随机采样点
    sample_size = min(10000, points_reshaped.shape[0])
    indices = np.random.choice(points_reshaped.shape[0], sample_size, replace=False)
    
    sampled_points = points_reshaped[indices]
    sampled_seg = seg_reshaped[indices]
    
    # 创建颜色映射
    colors = np.zeros((len(sampled_points), 3))
    colors[sampled_seg == 0] = [0.7, 0.7, 0.7]  # 背景
    colors[sampled_seg == 1] = [1, 1, 1]        # 机器人
    colors[sampled_seg == 101] = [1, 0, 0]      # 红盒子
    colors[sampled_seg == 102] = [0, 1, 0]      # 绿盒子
    
    # 不同视角的3D点云可视化
    views = [
        (30, 45, '正视图'),
        (0, 0, '顶视图'),
        (0, 90, '侧视图'),
        (60, 30, '俯视图')
    ]
    
    fig = plt.figure(figsize=(16, 12))
    
    for i, (elev, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
                   c=colors, s=2, alpha=0.5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pointcloud_3d_views.png'))
    
    # 按标签分离的3D点云
    fig = plt.figure(figsize=(15, 10))
    
    unique_seg = np.unique(sampled_seg)
    labels = {0: "背景", 1: "机器人", 101: "红盒子", 102: "绿盒子"}
    
    for i, label in enumerate(unique_seg, 1):
        if i > 4:  # 最多显示4个标签
            break
            
        mask = sampled_seg == label
        if np.sum(mask) == 0:
            continue
            
        ax = fig.add_subplot(2, 2, i, projection='3d')
        ax.scatter(sampled_points[mask, 0], sampled_points[mask, 1], sampled_points[mask, 2], 
                  c=colors[mask], s=3)
        ax.set_title(f"{labels.get(label, f'标签{label}')} 点云")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pointcloud_3d_by_label.png'))

if __name__ == "__main__":
    check_segmentation()
    check_seg_rgb_correspondence()
    check_pointcloud()
    visualize_3d_pointcloud()