import numpy as np
from plyfile import PlyData, PlyElement
import os
import argparse

def convert_to_standard_format(input_path, output_path, sh_degree=3):
    """
    将自定义的单通道强度PLY文件转换为标准的3DGS格式。

    :param input_path: 输入的自定义PLY文件路径。
    :param output_path: 输出的标准格式PLY文件路径。
    :param sh_degree: 目标球谐函数的阶数，通常为3。
    """
    print(f"Reading custom PLY file from: {input_path}")
    try:
        plydata = PlyData.read(input_path)
    except FileNotFoundError:
        print(f"!!! ERROR: Input file not found at '{input_path}'")
        return
    except Exception as e:
        print(f"!!! ERROR: Could not read PLY file. Reason: {e}")
        return

    if 'vertex' not in plydata:
        print("!!! ERROR: No 'vertex' element in the PLY file.")
        return
        
    vertex_element = plydata['vertex']
    num_points = vertex_element.count
    
    print(f"Loaded {num_points} points.")

    # --- 1. 构造新的数据类型 (dtype) ---
    
    # 基础属性
    new_dtype_list = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')
    ]
    
    # 添加DC项 (基础颜色)
    for i in range(3):
        new_dtype_list.append((f'f_dc_{i}', 'f4'))
        
    # 添加Rest项 (高阶SH)
    num_rest_coeffs = (sh_degree + 1)**2 - 1
    for i in range(num_rest_coeffs * 3):
        new_dtype_list.append((f'f_rest_{i}', 'f4'))
        
    # 添加其他属性
    new_dtype_list.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ])
    
    # --- 2. 创建并填充新的数据数组 ---
    
    new_vertex_data = np.zeros(num_points, dtype=new_dtype_list)
    
    # 复制已有数据
    old_data = vertex_element.data
    common_properties = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'opacity', 
                         'scale_0', 'scale_1', 'scale_2', 
                         'rot_0', 'rot_1', 'rot_2', 'rot_3']
    for prop in common_properties:
        if prop in old_data.dtype.names:
            new_vertex_data[prop] = old_data[prop]
        else:
            print(f"Warning: Property '{prop}' not found in source file. Filling with 0.")

    # 处理特征/颜色
    if 'f_0' in old_data.dtype.names:
        print("Converting single-channel intensity 'f_0' to RGB 'f_dc' terms...")
        intensity = old_data['f_0']
        new_vertex_data['f_dc_0'] = intensity
        new_vertex_data['f_dc_1'] = intensity
        new_vertex_data['f_dc_2'] = intensity
    else:
        print("Warning: 'f_0' not found. Filling 'f_dc' terms with 0.5 (gray).")
        new_vertex_data['f_dc_0'] = 0.5
        new_vertex_data['f_dc_1'] = 0.5
        new_vertex_data['f_dc_2'] = 0.5

    # f_rest 项已经默认为0，无需操作
    print(f"Filling {num_rest_coeffs * 3} 'f_rest' terms with 0.")
    
    # --- 3. 创建新的PLY文件并保存 ---
    
    # 创建新的 PlyElement
    new_vertex_element = PlyElement.describe(new_vertex_data, 'vertex')
    
    # 创建新的 PlyData 对象
    new_plydata = PlyData([new_vertex_element], text=plydata.text)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入文件
    new_plydata.write(output_path)
    
    print(f"\nSuccessfully converted file and saved to:\n{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert custom 3DGS PLY to standard format.")
    parser.add_argument("input_path", type=str, help="Path to the input custom PLY file.")
    parser.add_argument("-o", "--output_path", type=str, default=None, 
                        help="Path to save the output standard PLY file. "
                             "Defaults to '<input_path>_standard.ply'.")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree for the output format.")
    
    args = parser.parse_args()

    if args.output_path is None:
        # 自动生成输出文件名
        base, ext = os.path.splitext(args.input_path)
        args.output_path = f"{base}_standard{ext}"

    convert_to_standard_format(args.input_path, args.output_path, args.sh_degree)