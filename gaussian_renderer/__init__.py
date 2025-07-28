#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.cameras import SlicePlane # 导入我们的SlicePlane类，用于类型提示
from utils.sh_utils import eval_sh

def render(slice_plane: SlicePlane, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render a 2D slice of the scene.
    
    This function has been modified to work with SlicePlane objects for orthogonal slicing
    instead of perspective rendering with Camera objects.

    :param slice_plane: The SlicePlane object defining the Z-position and dimensions of the slice.
    :param pc: The GaussianModel to be rendered.
    :param pipe: The pipeline configuration.
    :param bg_color: The background color tensor.
    :param scaling_modifier: A modifier for the size of the Gaussians.
    :param override_color: Optional tensor to override the Gaussian colors.
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # =================================================================================
    # 核心修改：设置光栅化配置，以适应正交切片
    # =================================================================================
    # 我们不再需要 FoV, viewmatrix, projmatrix, 和 campos.
    # 我们将传递新的参数，如 z_position，来控制切片。
    # 注意：这些新参数（如 z_position）目前在 GaussianRasterizationSettings 中还不存在。
    # 我们下一步需要在 C++ 绑定代码中添加它们。现在我们先在Python层准备好。

    raster_settings = GaussianRasterizationSettings(
        image_height=int(slice_plane.image_height),
        image_width=int(slice_plane.image_width),
        tanfovx=1.0,  # 正交投影下无意义，但需提供一个值
        tanfovy=1.0,  # 正交投影下无意义，但需提供一个值
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=torch.eye(4, device="cuda"), # 无意义，但需提供一个单位矩阵
        projmatrix=torch.eye(4, device="cuda"), # 无意义，但需提供一个单位矩阵
        sh_degree=pc.active_sh_degree, # 这将是0
        campos=torch.zeros(3, device="cuda"), # 无意义，但需提供一个值
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        
        # --- 我们自定义的新增参数 ---
        # 这些参数需要稍后在 C++ 层的 GaussianRasterizationSettings 结构体中定义
        is_slice_rendering=True,                 # 一个布尔标志，告诉CUDA核函数切换到切片模式
        z_position=float(slice_plane.z_position), # 切片的Z坐标
        # 假设您的数据XY范围是0-512，如果不是，需要归一化或传递min/max
        x_min=0.0,
        x_max=float(slice_plane.image_width),
        y_min=0.0,
        y_max=float(slice_plane.image_height)
    )
    # =================================================================================

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # --- 协方差计算逻辑 (保持不变) ---
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # =================================================================================
    # 核心修改：颜色/强度处理
    # =================================================================================
    # 我们不再使用球谐函数 (SH)。
    # 我们的高斯模型 (`pc`) 的 `get_features` 方法现在返回一个 (N, 1) 的张量。
    # 我们需要将其扩展到 (N, 3) 以适应光栅化器对RGB的期望，或者修改光栅化器以处理单通道。
    # 为了简化，我们先将其复制到三个通道，形成灰度图。
    shs = None
    colors_precomp = None
    if override_color is None:
        # get_features() 返回的是 [N, 1] 的强度值
        intensities = pc.get_features
        # 将 [N, 1] 扩展到 [N, 3] 来创建灰度颜色
        colors_precomp = intensities.repeat(1, 3) 
    else:
        # 如果提供了覆盖颜色，确保它是灰度的
        if override_color.shape[1] == 1:
            colors_precomp = override_color.repeat(1, 3)
        else:
            colors_precomp = override_color
    # =================================================================================

    # --- 调用光栅化器 (Rasterizer) ---
    # 我们不再需要 separate_sh 的分支，因为我们总是不使用SH
    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs, # 始终为 None
        colors_precomp = colors_precomp, # 我们预计算的灰度颜色
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
        
    # --- 移除曝光调整 ---
    # use_trained_exp 逻辑依赖于 viewpoint_camera.image_name，我们不再有这个。
    # 曝光调整对于科学数据可能也不适用。

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    
    # 因为我们的输入颜色是灰度的 (R=G=B)，所以输出的渲染图像也是灰度的。
    # 我们可以只取第一个通道用于计算损失。
    grayscale_image = rendered_image[0:1, :, :] # 取出R通道作为我们的单通道灰度图

    out = {
        "render": grayscale_image, # 返回单通道灰度图
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image # 这里的深度图可能代表了对渲染贡献最大的高斯的Z值
        }
    
    return out