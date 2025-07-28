import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser
import sys

def load_ply_to_torch(path):
    """从PLY文件加载高斯数据到PyTorch张量。"""
    try:
        plydata = PlyData.read(path)
    except Exception as e:
        print(f"错误: 无法读取PLY文件 '{path}': {e}")
        sys.exit(1)

    gaussians = plydata['vertex']
    
    # 检查基本属性是否存在
    required_attrs = ['x', 'y', 'z', 'opacity']
    for attr in required_attrs:
        if attr not in gaussians.data.dtype.names:
            print(f"错误: PLY文件缺少必需属性 '{attr}'。")
            sys.exit(1)
            
    # 将numpy数据转换为字典，键为属性名，值为PyTorch张量
    properties = {}
    for name in gaussians.data.dtype.names:
        data = torch.from_numpy(np.stack(gaussians.data[name]).astype(np.float32))
        if len(data.shape) == 1:
            data = data.unsqueeze(-1)
        properties[name] = data

    print(f"成功加载 {len(properties['x'])} 个高斯点。")
    return properties

def save_torch_to_ply(properties, path):
    """将PyTorch张量字典保存为PLY文件。"""
    
    # 将所有张量移到CPU并转换为numpy
    cpu_properties = {name: tensor.cpu().numpy() for name, tensor in properties.items()}

    # 创建一个结构化的numpy数组
    dtype_list = []
    attribute_list = []
    
    # 确保xyz和opacity在前
    ordered_keys = ['x', 'y', 'z']
    # 添加其他keys，但排除已添加的
    ordered_keys.extend([k for k in cpu_properties.keys() if k not in ordered_keys])
    
    for name in ordered_keys:
        if name not in cpu_properties: continue
        array = cpu_properties[name]
        # 如果是1D数组，保持原样，否则展平
        if array.shape[1] == 1:
            dtype_list.append((name, 'f4'))
            attribute_list.append(array.squeeze())
        else:
            for i in range(array.shape[1]):
                dtype_list.append((f'{name}_{i}', 'f4'))
            attribute_list.append(array)
            
    # 重新构建结构化数组
    num_points = cpu_properties['x'].shape[0]
    elements = np.empty(num_points, dtype=dtype_list)
    
    current_idx = 0
    for name in ordered_keys:
        if name not in cpu_properties: continue
        array = cpu_properties[name]
        num_cols = array.shape[1]
        if num_cols == 1:
            elements[name] = array.squeeze()
        else:
            for i in range(num_cols):
                elements[f'{name}_{i}'] = array[:, i]

    # 创建 PlyElement 并写入文件
    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(path)
    print(f"已将 {num_points} 个裁剪后的高斯点保存到 '{path}'")

def apply_pruning_rules(properties, args):
    """根据命令行参数应用所有裁剪规则。"""
    
    num_initial_points = properties['x'].shape[0]
    
    # 初始掩码，全为True，表示保留所有点
    keep_mask = torch.ones(num_initial_points, dtype=torch.bool)

    print("\n--- 开始裁剪 ---")
    
    # 规则 1: 裁剪不透明度 (Opacity)
    if args.opacity_thresh is not None:
        opacity_mask = (properties['opacity'].squeeze() >= args.opacity_thresh)
        num_removed = (~opacity_mask).sum().item()
        keep_mask &= opacity_mask
        print(f"规则[Opacity]: 裁剪掉 {num_removed} 个点 (opacity < {args.opacity_thresh})")
        
    # 规则 2: 裁剪强度/特征 (Feature f_0)
    if args.feature_thresh is not None:
        if 'f_0' in properties:
            feature_mask = (properties['f_0'].squeeze() >= args.feature_thresh)
            num_removed = (~feature_mask).sum().item()
            keep_mask &= feature_mask
            print(f"规则[Feature]: 裁剪掉 {num_removed} 个点 (f_0 < {args.feature_thresh})")
        else:
            print("警告: 未找到 'f_0' 属性，跳过特征裁剪。")

    # 规则 3: 裁剪尺度 (Scale)
    if args.scale_thresh is not None:
        if 'scale_0' in properties and 'scale_1' in properties and 'scale_2' in properties:
            scales = torch.stack([properties['scale_0'], properties['scale_1'], properties['scale_2']], dim=1)
            # 检查最大尺度，如果连最大的尺度都很小，就裁剪掉
            max_scale, _ = torch.max(torch.exp(scales), dim=1) # scale存储的是log值，需要exp转换
            scale_mask = (max_scale.squeeze() >= args.scale_thresh)
            num_removed = (~scale_mask).sum().item()
            keep_mask &= scale_mask
            print(f"规则[Scale]: 裁剪掉 {num_removed} 个点 (max_scale < {args.scale_thresh})")
        else:
            print("警告: 未找到 'scale_0, scale_1, scale_2' 属性，跳过尺度裁剪。")

    # 规则 4: 裁剪空间位置 (XYZ)
    if args.xyz_thresh is not None:
        xyz = torch.stack([properties['x'], properties['y'], properties['z']], dim=1).squeeze()
        # 检查每个坐标轴是否超出范围
        xyz_mask = torch.all(torch.abs(xyz) <= args.xyz_thresh, dim=1)
        num_removed = (~xyz_mask).sum().item()
        keep_mask &= xyz_mask
        print(f"规则[XYZ]: 裁剪掉 {num_removed} 个点 (任一坐标轴 |val| > {args.xyz_thresh})")
        
    if args.aspect_ratio_thresh is not None:
        scales = torch.exp(torch.stack([properties['scale_0'], properties['scale_1'], properties['scale_2']], dim=1))
        max_s, _ = torch.max(scales, dim=1)
        min_s, _ = torch.min(scales, dim=1)
        aspect_ratio_mask = ((max_s / (min_s + 1e-8)).squeeze() < args.aspect_ratio_thresh)
        num_removed = (~aspect_ratio_mask).sum().item()
        keep_mask &= aspect_ratio_mask
        print(f"规则[Aspect Ratio]: 裁剪掉 {num_removed} 个点 (aspect_ratio > {args.aspect_ratio_thresh})")
    
    
    # 应用最终的掩码
    pruned_properties = {name: tensor[keep_mask] for name, tensor in properties.items()}
    
    num_final_points = pruned_properties['x'].shape[0]
    num_total_removed = num_initial_points - num_final_points
    
    print("\n--- 裁剪完成 ---")
    print(f"初始点数: {num_initial_points}")
    print(f"最终保留点数: {num_final_points}")
    print(f"总共裁剪掉: {num_total_removed} ({num_total_removed / num_initial_points:.2%})")
    
    return pruned_properties

if __name__ == "__main__":
    parser = ArgumentParser(description="裁剪训练好的3D高斯PLY文件。")
    parser.add_argument("input_ply", type=str, help="输入的 point_cloud.ply 文件路径。")
    parser.add_argument("--opacity_thresh", type=float, default=None, help="不透明度阈值，低于此值的点将被裁剪。")
    parser.add_argument("--feature_thresh", type=float, default=None, help="第一个特征(f_0)的阈值，低于此值的点将被裁剪。")
    parser.add_argument("--scale_thresh", type=float, default=None, help="最大尺度阈值，最大尺度低于此值的点将被裁剪。")
    parser.add_argument("--xyz_thresh", type=float, default=None, help="空间坐标阈值，任何坐标轴的绝对值大于此值的点将被裁剪。")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=None, help="长宽比阈值，最大尺度与最小尺度之比大于此值的点将被裁剪。")

    args = parser.parse_args()
    
    # 加载数据
    properties = load_ply_to_torch(args.input_ply)
    
    # 应用裁剪规则
    pruned_properties = apply_pruning_rules(properties, args)
    
    # 生成输出文件名
    input_dir = os.path.dirname(args.input_ply)
    input_filename = os.path.basename(args.input_ply)
    output_filename = input_filename.replace('.ply', '_pruned.ply')
    output_path = os.path.join(input_dir, output_filename)
    
    # 保存裁剪后的PLY文件
    save_torch_to_ply(pruned_properties, output_path)