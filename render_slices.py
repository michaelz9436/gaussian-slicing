import os
import torch
import sys
from argparse import ArgumentParser, Namespace
from torchvision.utils import save_image
import glob
import re
from PIL import Image
import numpy as np
from tqdm import tqdm

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scene import Scene, GaussianModel
from scene.cameras import SlicePlane 
from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams

def render_trained_slices(scene, gaussians, pipe, model_path, iteration):
    """
    (旧功能) 渲染所有在训练/测试集中存在的切片，并生成对比图。
    """
    bg_color = [0] # 背景为黑色
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    

    render_output_dir = os.path.join(model_path, "renders", f"iter_{iteration}")
    gt_output_dir = os.path.join(model_path, "gt")
    comparison_output_dir = os.path.join(model_path, "comparison", f"iter_{iteration}")
    os.makedirs(render_output_dir, exist_ok=True)
    os.makedirs(gt_output_dir, exist_ok=True)
    os.makedirs(comparison_output_dir, exist_ok=True)

    print(f"Rendering all TRAINED slices... Output will be saved to {comparison_output_dir}")

    slice_planes = scene.getTrainCameras() 

    with torch.no_grad():
        for slice_plane in tqdm(slice_planes, desc="Rendering trained slices"):
            rendered_image = render(slice_plane, gaussians, pipe, background)["render"]
            gt_image = slice_plane.original_image
            
            save_image(rendered_image, os.path.join(render_output_dir, f"{slice_plane.image_name}.png"))
            
            if not os.path.exists(os.path.join(gt_output_dir, f"{slice_plane.image_name}")):
                 save_image(gt_image, os.path.join(gt_output_dir, f"{slice_plane.image_name}"))
            
            rendered_gray_3ch = rendered_image.repeat(1, 3, 1, 1).squeeze(0)
            gt_gray_3ch = gt_image.repeat(1, 3, 1, 1).squeeze(0)
            comparison_image = torch.cat([gt_gray_3ch, rendered_gray_3ch], dim=2)
            
            save_image(comparison_image, os.path.join(comparison_output_dir, f"compare_{slice_plane.image_name}"))

    print(f"\nComparison images are in: {comparison_output_dir}")

def render_novel_slices(scene, gaussians, pipe, model_path, iteration, z_values):
    """
    渲染指定Z坐标的新切片。
    """
    bg_color = [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建输出目录
    novel_output_dir = os.path.join(model_path, "novel_renders", f"iter_{iteration}")
    os.makedirs(novel_output_dir, exist_ok=True)
    
    print(f"Rendering NOVEL slices at z={z_values}... Output will be saved to {novel_output_dir}")

    # 从训练集中取第一个切片作为模板
    template_slice = scene.getTrainCameras()[0]
    width, height = template_slice.image_width, template_slice.image_height
    resolution = (width, height)

    # 创建一个空白的PIL图像作为占位符
    placeholder_image = Image.fromarray(np.zeros((height, width), dtype=np.uint8))

    with torch.no_grad():
        for z in tqdm(z_values, desc="Rendering novel slices"):
            # 为新的Z坐标创建一个临时的SlicePlane对象
            novel_slice_plane = SlicePlane(
                uid=-1, 
                z_position=z,
                image=placeholder_image,
                resolution=resolution
            )
            
            # 渲染新切片
            rendered_image = render(novel_slice_plane, gaussians, pipe, background)["render"]

            z_str = f"{z:.3f}".replace('.', '_') 
            filename = f"novel_z_{z_str}.png"
            save_path = os.path.join(novel_output_dir, filename)

            # 保存
            save_image(rendered_image, save_path)
    
    print(f"\nNovel slice images are in: {novel_output_dir}")


def main_render(model_params, pipeline_params, args):
    """
    主渲染函数，加载模型并根据参数选择渲染模式。
    """
    # 1. 初始化模型和管线
    gaussians = GaussianModel(model_params.sh_degree)
    pipe = pipeline_params

    # 2. 加载场景数据 (这会加载 SlicePlane 对象)
    scene = Scene(model_params, gaussians, shuffle=False)
    
    # 3. 自动检测并加载最新的训练好的模型
    model_path = model_params.model_path
    iteration_dirs = sorted(glob.glob(os.path.join(model_path, "point_cloud", "iteration_*")))
    
    if not iteration_dirs:
        raise RuntimeError(f"未在 '{model_path}' 中找到任何 iteration_* 文件夹。")
    
    # 默认使用最新的模型，除非用户指定了iteration
    if args.iteration == -1:
        latest_dir = iteration_dirs[-1]
        match = re.search(r'iteration_(\d+)', latest_dir)
        iteration = int(match.group(1))
        print(f"自动检测到最新模型: iteration_{iteration}")
    else:
        iteration = args.iteration
        print(f"用户指定模型: iteration_{iteration}")

    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"指定的模型文件不存在: {ply_path}")

    print(f"Loading trained model from: {ply_path}")
    gaussians.load_ply(ply_path)

    # 4. 根据命令行参数选择执行哪个渲染功能
    if args.novel_z:
        # 如果用户提供了 novel_z 参数，就渲染新切片
        render_novel_slices(scene, gaussians, pipe, model_path, iteration, args.novel_z)
    else:
        # 否则，执行默认的渲染测试集功能
        render_trained_slices(scene, gaussians, pipe, model_path, iteration)

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Script to render slices from a trained model.")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", type=int, default=-1, help="指定要加载的模型迭代次数。默认为-1，表示自动加载最新模型。")
    
    # --- 添加新功能的命令行参数 ---
    parser.add_argument("--novel_z", nargs='+', type=float, default=None, 
                        help="渲染指定Z坐标的新切片，而不是渲染测试集。可以提供多个值，例如：--novel_z 0.5 1.5 2.5")

    args = parser.parse_args()
    safe_state(args.quiet)

    # 调用主函数
    main_render(lp.extract(args), pp.extract(args), args)