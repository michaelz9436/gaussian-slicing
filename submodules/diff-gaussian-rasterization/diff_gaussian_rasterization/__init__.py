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

from typing import NamedTuple, Optional
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        
        # =================================================================================
        # 核心修改 2: 重新组织传递给 C++ 的参数
        # =================================================================================
        # 根据 raster_settings.is_slice_rendering 的值，我们准备不同的参数包
        
        if not raster_settings.is_slice_rendering:
            # 原始的透视渲染参数包 (保持不变，以防需要)
            args = (
                raster_settings.bg, 
                means3D,
                colors_precomp,
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                raster_settings.image_height,
                raster_settings.image_width,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                raster_settings.prefiltered,
                raster_settings.antialiasing,
                raster_settings.debug,
                
                # 为保持函数签名一致，为切片参数传递默认/空值
                False, 0.0, 0.0, 0.0, 0.0, 0.0
            )
        else:
            # 我们为正交切片渲染准备的新参数包
            args = (
                raster_settings.bg, 
                means3D,
                colors_precomp,
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                
                # 传递无意义的占位符给旧的相机参数
                torch.empty(0, device="cuda"), # viewmatrix
                torch.empty(0, device="cuda"), # projmatrix
                0.0, # tanfovx
                0.0, # tanfovy
                
                raster_settings.image_height,
                raster_settings.image_width,
                sh,
                raster_settings.sh_degree, # 应该是0
                torch.empty(0, device="cuda"), # campos
                raster_settings.prefiltered,
                raster_settings.antialiasing,
                raster_settings.debug,
                
                # --- 传递我们新增的自定义参数 ---
                raster_settings.is_slice_rendering,
                raster_settings.z_position,
                raster_settings.x_min,
                raster_settings.x_max,
                raster_settings.y_min,
                raster_settings.y_max
            )

        # 调用 C++/CUDA 光栅化器
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # 为反向传播保留相关张量
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):

        # 从 context 恢复必要的值
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # =================================================================================
        # 核心修改 3: 为反向传播也准备新的参数包
        # =================================================================================
        if not raster_settings.is_slice_rendering:
             # 原始的透视渲染反向传播参数包
            args = (raster_settings.bg,
                    means3D, 
                    radii, 
                    colors_precomp, 
                    opacities,
                    scales, 
                    rotations, 
                    raster_settings.scale_modifier, 
                    cov3Ds_precomp, 
                    raster_settings.viewmatrix, 
                    raster_settings.projmatrix, 
                    raster_settings.tanfovx, 
                    raster_settings.tanfovy, 
                    grad_out_color,
                    grad_out_depth, 
                    sh, 
                    raster_settings.sh_degree, 
                    raster_settings.campos,
                    geomBuffer,
                    num_rendered,
                    binningBuffer,
                    imgBuffer,
                    raster_settings.antialiasing,
                    raster_settings.debug)
        else:
            # 我们为正交切片渲染准备的新反向传播参数包
            args = (raster_settings.bg,
                    means3D, 
                    radii, 
                    colors_precomp, 
                    opacities,
                    scales, 
                    rotations, 
                    raster_settings.scale_modifier, 
                    cov3Ds_precomp, 
                    
                    # 传递无意义的占位符给旧的相机参数
                    torch.empty(0, device="cuda"), # viewmatrix
                    torch.empty(0, device="cuda"), # projmatrix
                    0.0, # tanfovx
                    0.0, # tanfovy
                    
                    grad_out_color,
                    grad_out_depth, 
                    sh, 
                    raster_settings.sh_degree, 
                    torch.empty(0, device="cuda"), # campos
                    geomBuffer,
                    num_rendered,
                    binningBuffer,
                    imgBuffer,
                    raster_settings.antialiasing,
                    raster_settings.debug)


        # 调用 C++/CUDA 反向传播方法计算梯度
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

# =================================================================================
# 核心修改 1: 扩展 GaussianRasterizationSettings
# =================================================================================
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool
    
    # --- 我们自定义的新增参数 ---
    # 使用 Optional 和默认值来保持向后兼容性
    is_slice_rendering: bool = False
    z_position: float = 0.0
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
# =================================================================================

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # 注意: mark_visible 函数仍然依赖于旧的相机模型。
        # 在我们的切片渲染模式下，我们不需要这个视锥剔除功能，
        # 所以可以暂时忽略它，或者之后创建一个不执行任何操作的替代版本。
        with torch.no_grad():
            raster_settings = self.raster_settings
            # 只有在非切片模式下才执行视锥剔除
            if not raster_settings.is_slice_rendering:
                visible = _C.mark_visible(
                    positions,
                    raster_settings.viewmatrix,
                    raster_settings.projmatrix)
                return visible
            else:
                # 在切片模式下，我们认为所有点都是“可见”的，
                # 因为剔除将在CUDA核函数内部基于Z位置来完成。
                return torch.ones(positions.shape[0], dtype=torch.bool, device=positions.device)

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        # 确保空张量是正确的类型和设备
        device = means3D.device
        if shs is None:
            shs = torch.empty(0, device=device)
        if colors_precomp is None:
            colors_precomp = torch.empty(0, device=device)

        if scales is None:
            scales = torch.empty(0, device=device)
        if rotations is None:
            rotations = torch.empty(0, device=device)
        if cov3D_precomp is None:
            cov3D_precomp = torch.empty(0, device=device)

        # 调用 C++/CUDA 光栅化例程
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )