import os
import torch
import numpy as np
from PIL import Image
from typing import List

# 导入 plyfile 库来直接处理 .ply 文件
from plyfile import PlyData

# 导入我们需要的类和函数
from .cameras import SlicePlane
# 从 .dataset_readers 导入 SceneInfo 和 BasicPointCloud
from .dataset_readers import SceneInfo, BasicPointCloud


class SliceLoader:
    """
    这个类负责加载您的切片数据。
    它会查找一个包含切片图像的文件夹，并按顺序加载它们。
    它还会加载一个初始点云，即使点云没有颜色信息。
    """
    def __init__(self,
                 source_path: str,
                 image_dir: str,
                 point_cloud_file: str = 'sparse.ply',
                 z_scale: float = 1.0,
                 resolution: tuple = (512, 512)):
        """
        初始化加载器。
        """
        self.source_path = source_path
        # 修改：如果 image_dir 是空字符串，直接使用 source_path
        self.image_dir = os.path.join(source_path, image_dir) if image_dir else source_path
        self.point_cloud_path = os.path.join(source_path, point_cloud_file)
        self.z_scale = z_scale
        self.resolution = resolution

    def load_data(self) -> SceneInfo:
        """
        执行加载操作并返回 SceneInfo 对象。
        """
        print(f"Loading slice data from: {self.source_path}")
        print(f"Z-axis scale factor: {self.z_scale}")

        # 1. 加载稀疏点云 (健壮版本)
        if not os.path.exists(self.point_cloud_path):
            raise FileNotFoundError(f"Point cloud file not found at: {self.point_cloud_path}")
        
        try:
            # 使用 plyfile 直接读取PLY文件
            plydata = PlyData.read(self.point_cloud_path)
            vertices = plydata['vertex']
            
            # 提取 x, y, z 坐标 (必需)
            points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
            
            # 尝试加载颜色，如果不存在则创建默认颜色 (灰色)
            try:
                colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
            except ValueError:
                print("Point cloud does not contain color info. Creating dummy gray colors.")
                colors = np.full_like(points, 0.5) # 0.5 for gray
            
            # 尝试加载法线，如果不存在则创建零向量
            try:
                normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
            except ValueError:
                normals = np.zeros_like(points)
            
            # 创建 BasicPointCloud 对象，这是 SceneInfo 所期望的格式
            pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
            print(f"Loaded point cloud with {len(pcd.points)} points.")

        except Exception as e:
            raise IOError(f"Could not load or parse point cloud from {self.point_cloud_path}: {e}")

        # 2. 加载切片图像并创建 SlicePlane 对象
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found at: {self.image_dir}")

        image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        if not image_files:
            raise ValueError(f"No image files found in {self.image_dir}")

        print(f"Found {len(image_files)} slice images.")

        slice_planes = []
        for idx, image_name in enumerate(image_files):
            image_path = os.path.join(self.image_dir, image_name)
            try:
                pil_image = Image.open(image_path).convert('L') # 强制转换为灰度图
            except Exception as e:
                print(f"Warning: Could not load image {image_name}. Skipping. Error: {e}")
                continue

            z_position = (idx + 1) * self.z_scale

            slice_plane = SlicePlane(
                uid=idx,
                z_position=z_position,
                image=pil_image,
                resolution=self.resolution,
                image_name=image_name
            )
            slice_planes.append(slice_plane)

        print(f"Successfully created {len(slice_planes)} SlicePlane objects.")

        # 3. 将数据打包成 SceneInfo 对象
        train_slice_planes = slice_planes
        test_slice_planes = []

        scene_info = SceneInfo(
            point_cloud=pcd,
            train_cameras=train_slice_planes,
            test_cameras=test_slice_planes,
            nerf_normalization=None,
            ply_path=self.point_cloud_path,
            is_nerf_synthetic=False
        )
        
        return scene_info

def readSliceData(source_path, images, eval=False, z_scale=1.0):
    """
    一个方便的包装函数，使其接口与原始的 readColmap3Data 等函数类似。
    """
    point_cloud_file = 'sparse.ply'
    # TODO: Make resolution configurable from args, or read from image files
    resolution = (512, 512) 

    loader = SliceLoader(
        source_path=source_path,
        image_dir=images,
        point_cloud_file=point_cloud_file,
        z_scale=z_scale,
        resolution=resolution
    )
    
    return loader.load_data()