import numpy as np
from plyfile import PlyData
import argparse
import os

def convert_ply_to_txt(ply_path, txt_path):
    """
    读取一个由3D Gaussian Splatting生成的 .ply 文件，
    并将其内容以可读的文本格式写入 .txt 文件。
    """
    if not os.path.exists(ply_path):
        print(f"错误：输入文件不存在 -> {ply_path}")
        return

    try:
        # 使用 plyfile 库读取 .ply 文件
        plydata = PlyData.read(ply_path)
        vertex_element = plydata['vertex']
        
        # ======================================================================
        #  错误修复: 正确获取属性名称的方法
        # ======================================================================
        property_names = [prop.name for prop in vertex_element.properties]
        print(f"文件包含的属性: {property_names}")
        # ======================================================================

        num_gaussians = len(vertex_element.data)
        print(f"文件中包含 {num_gaussians} 个高斯球。")

        # 准备写入文件
        with open(txt_path, 'w') as f:
            # 写入文件头，描述每一列是什么
            header = "Index\t"
            header += "X\tY\tZ\t"
            header += "Intensity\t"  
            header += "Opacity\t"
            header += "Scale_0\tScale_1\tScale_2\t"
            header += "Rot_0\tRot_1\tRot_2\tRot_3\n"
            f.write(header)
            f.write("-" * 80 + "\n")

            # 逐个高斯球写入数据
            for i in range(num_gaussians):
                data = vertex_element.data[i]
                
                # --- 提取核心属性 ---
                # 位置
                x, y, z = data['x'], data['y'], data['z']
                
                # 强度 (我们的模型中 f_dc_0 代表强度)
                # 检查属性是否存在
                intensity = data['f_0'] if 'f_0' in property_names else 0.0
                
                # 不透明度
                if 'opacity' in property_names:
                    # 不透明度通常存储为logit，需要用sigmoid转换
                    opacity_logit = data['opacity']
                    opacity = 1 / (1 + np.exp(-opacity_logit))
                else:
                    opacity = 0.0
                    
                # 尺度
                if 'scale_0' in property_names:
                    # 尺度通常存储为log，需要用exp转换
                    scale_0 = np.exp(data['scale_0'])
                    scale_1 = np.exp(data['scale_1'])
                    scale_2 = np.exp(data['scale_2'])
                else:
                    scale_0, scale_1, scale_2 = 0.0, 0.0, 0.0

                # 旋转 (四元数)
                if 'rot_0' in property_names:
                    # 旋转四元数通常需要归一化
                    rot = np.array([data['rot_0'], data['rot_1'], data['rot_2'], data['rot_3']])
                    norm = np.linalg.norm(rot)
                    if norm > 1e-6: # 避免除以零
                        rot /= norm
                    rot_0, rot_1, rot_2, rot_3 = rot[0], rot[1], rot[2], rot[3]
                else:
                    rot_0, rot_1, rot_2, rot_3 = 1.0, 0.0, 0.0, 0.0 # 单位四元数
                
                # 格式化输出字符串
                line = (
                    f"{i}\t"
                    f"{x:.6f}\t{y:.6f}\t{z:.6f}\t"
                    f"{intensity:.6f}\t"
                    f"{opacity:.6f}\t"
                    f"{scale_0:.6f}\t{scale_1:.6f}\t{scale_2:.6f}\t"
                    f"{rot_0:.6f}\t{rot_1:.6f}\t{rot_2:.6f}\t{rot_3:.6f}\n"
                )
                
                f.write(line)

        print(f"成功将 {ply_path} 转换为 {txt_path}")

    except Exception as e:
        print(f"处理文件时发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 3DGS 的 .ply 文件转换为可读的 .txt 文件。")
    parser.add_argument("input_ply", type=str, help="输入的 .ply 文件路径。")
    parser.add_argument("-o", "--output_txt", type=str, help="输出的 .txt 文件路径 (可选)。")
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，则在输入文件同目录下生成同名的 .txt 文件
    if not args.output_txt:
        output_path = os.path.splitext(args.input_ply)[0] + ".txt"
    else:
        output_path = args.output_txt
        
    convert_ply_to_txt(args.input_ply, output_path)