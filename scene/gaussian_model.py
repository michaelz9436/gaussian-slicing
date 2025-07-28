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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
# from utils.sh_utils import RGB2SH  # 我们不再需要SH，可以注释掉
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    # =================================================================================
    # 修改 1: 修改 __init__ 以简化特征表示
    # =================================================================================
    def __init__(self, sh_degree, optimizer_type="default"):
        # sh_degree 参数现在被忽略，但为了保持接口兼容性，我们暂时保留它
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = 0  # 强制SH阶数为0
        self._xyz = torch.empty(0)
        
        # 将 _features_dc 改为 _features (或 _intensity)，并使其维度为 (N, 1)
        self._features = torch.empty(0) 
        
        # 完全移除 _features_rest
        # self._features_rest = torch.empty(0) 
        
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        # 移除 _features_rest
        return (
            self.active_sh_degree,
            self._xyz,
            self._features, # 使用 _features
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        # 移除 _features_rest
        (self.active_sh_degree, 
        self._xyz, 
        self._features, # 使用 _features
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        # 直接返回我们的单通道特征
        return self._features
    
    # 移除 get_features_dc 和 get_features_rest
    # @property
    # def get_features_dc(self):
    #     return self._features_dc
    
    # @property
    # def get_features_rest(self):
    #     return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        # 这个函数现在什么也不做
        pass

    # =================================================================================
    # 修改 2: 修改 create_from_pcd 以移除相机依赖
    # =================================================================================
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        
        # 初始化特征 (强度)
        # 原始代码使用 RGB2SH 处理颜色。这里我们用一个更简单的方式。
        # 我们可以取输入颜色的平均值作为初始强度，或者直接设为一个常数。
        # 假设 pcd.colors 是 (N, 3) 的 RGB 数组
        if pcd.colors is not None and pcd.colors.shape[1] == 3:
            # 将RGB转换为灰度值 (0.299*R + 0.587*G + 0.114*B)
            rgb_colors = torch.tensor(np.asarray(pcd.colors)).float().cuda() / 255.0
            grayscale = 0.299 * rgb_colors[:,0] + 0.587 * rgb_colors[:,1] + 0.114 * rgb_colors[:,2]
            # 我们需要一个逆激活函数，这里简单地用常数初始化
            fused_features = torch.ones((fused_point_cloud.shape[0], 1)).float().cuda() * 0.5
        else:
            # 如果没有颜色信息，就用一个默认值初始化
            fused_features = torch.ones((fused_point_cloud.shape[0], 1)).float().cuda() * 0.5

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 初始化协方差 (形状)
        # 原始代码使用 distCUDA2 计算到最近邻点的距离来初始化大小。这是一个好方法，我们保留它。
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 将高斯初始化为球形
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # 初始化为单位四元数（无旋转）
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 初始化不透明度
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 创建可优化的 nn.Parameter
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features = nn.Parameter(fused_features.requires_grad_(True))
        # 移除 _features_dc 和 _features_rest
        # self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # 移除与曝光相关的所有代码，因为我们的模型没有相机曝光
        # self.exposure_mapping = ...
        # self.pretrained_exposures = ...
        # self._exposure = ...

    # =================================================================================
    # 修改 3: 修改 training_setup 以简化优化器
    # =================================================================================
    def training_setup(self, training_args):
            self.percent_dense = training_args.percent_dense
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features], 'lr': training_args.feature_lr, "name": "features"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]

            if self.optimizer_type == "default":
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            elif self.optimizer_type == "sparse_adam":
                # 这里是之前可能出错的地方，确保 try/except 块完整
                try:
                    self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
                except:
                    # A special version of the rasterizer is required to enable sparse adam
                    self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
            
            # 这一行后面的代码现在有了正确的缩进级别
            self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                        lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)
        


    def update_learning_rate(self, iteration):
        # 移除曝光学习率更新
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # 别忘了更新其他参数的学习率，或者让它们保持固定
            elif param_group["name"] == "features":
                # 可以为 features 添加学习率调度，或保持不变
                pass
        # 返回 xyz 的学习率，因为 train.py 中会用到它
        return self.xyz_scheduler_args(iteration)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # 简化特征的保存
        for i in range(self._features.shape[1]):
            l.append('f_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # 简化特征保存
        features = self._features.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # 确保拼接的数组维度正确
        attributes = np.concatenate((xyz, normals, features, opacities, scale, rotation), axis=1)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    # ... (后续的 densify, prune, load_ply 等函数也需要做类似的简化)
    # 比如 densify_and_clone, densify_and_split, load_ply 等函数中
    # 所有涉及到 _features_dc 和 _features_rest 的地方，
    # 都需要统一修改为 _features。
    
    # 为了完整性，下面我将继续修改 densification 和 load_ply
    
    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # 从PLY加载我们简化的单通道特征
        feature_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_")]
        features = np.zeros((xyz.shape[0], len(feature_names)))
        for idx, attr_name in enumerate(feature_names):
            features[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # ... (加载 scales 和 rots 的代码保持不变) ...
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features = nn.Parameter(torch.tensor(features, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = 0

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # 如果没有点需要克隆，返回 None
        if not selected_pts_mask.any():
            return None, None
            
        new_xyz = self._xyz[selected_pts_mask]
        new_features = self._features[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # 将要克隆的点打包成字典并返回
        points_dict = {
            "xyz": new_xyz,
            "features": new_features,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation
        }
        
        return points_dict, selected_pts_mask


    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        将新的点属性张量追加到优化器的参数组中。
        这是处理动态点数的关键，特别是对于 Adam 优化器。
        这个函数完整地从原始 3DGS 代码适配而来。
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 我们只处理我们关心的参数组
            if group["name"] not in tensors_dict:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            # 检查参数是否在优化器状态中
            if stored_state is not None:
                # 扩展 Adam 的动量项 (exp_avg, exp_avg_sq)
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(tensors_dict[group["name"]])), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(tensors_dict[group["name"]])), dim=0)

                # 从优化器状态中删除旧的参数张量
                del self.optimizer.state[group['params'][0]]
                # 将新旧数据合并，并重新包装成 nn.Parameter
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], tensors_dict[group["name"]]), dim=0).requires_grad_(True))
                # 将新的 nn.Parameter 注册回优化器状态
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 如果参数不在优化器状态中（虽然不太可能发生），直接合并
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], tensors_dict[group["name"]]), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, xyz, features, opacity, scaling, rotation):
        """
        在增生（克隆或分裂）之后，将新创建的点添加到模型中的主函数。
        这是适配我们模型的“满血版”。
        """
        # 1. 将新属性打包成一个字典。键名必须与优化器参数组的 "name" 完全匹配。
        d = {
            "xyz": xyz,
            "features": features,
            "opacity": opacity,
            "scaling": scaling,
            "rotation": rotation,
        }

        # 2. 调用 cat_tensors_to_optimizer 来正确地将新数据和优化器状态合并。
        #    这个函数返回更新后的 nn.Parameter 对象。
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        # 3. 更新模型内部的属性张量，指向这些新的、合并后的 nn.Parameter 对象。
        self._xyz = optimizable_tensors["xyz"]
        self._features = optimizable_tensors["features"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 4. 为所有点（包括新点）创建新的、正确尺寸的零梯度统计数据缓冲区。
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def add_densification_stats(self, viewspace_point_tensor, visibility_filter):
        """
        这个函数累积用于增生的统计数据。
        """
        # 确保梯度已经被计算
        if self._xyz.grad is not None:
            # 只在可见点上累积梯度
            self.xyz_gradient_accum[visibility_filter] += torch.norm(viewspace_point_tensor.grad[visibility_filter, :2], dim=-1, keepdim=True)
            self.denom[visibility_filter] += 1

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Clone attributes for splitting
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        if not selected_pts_mask.any():
            return None, None # 如果没有点需要分裂，直接返回

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        # 修正: self._features[selected_pts_mask] 的形状是 [k, 1]，而 repeat(N,1) 会变成 [k*N, 1]。
        # 我们需要它变成 [k*N, 1, 1] 才能匹配 densification_postfix 的期望。
        # 或者直接修改 densification_postfix，但这里先保持原样，假设它能处理
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features = self._features[selected_pts_mask].repeat(N,1) # 修正：假设特征维度为 (N, F)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        
        # 返回新创建的点和用于剪枝的掩码
        return {"xyz": new_xyz, "features": new_features, "opacity": new_opacity, "scaling" : new_scaling, "rotation" : new_rotation}, selected_pts_mask


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        """
        这个函数执行增生和剪枝操作。
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # --- 增生 ---
        # 1. 克隆 (Clone)
        cloned_points_dict, clone_mask = self.densify_and_clone(grads, max_grad, extent)
        
        # 2. 分裂 (Split)
        split_points_dict, split_mask = self.densify_and_split(grads, max_grad, extent)
        
        # 3. 将所有新点收集起来
        new_points_list = []
        if cloned_points_dict is not None:
            new_points_list.append(cloned_points_dict)
        if split_points_dict is not None:
            new_points_list.append(split_points_dict)

        # 4. 如果有新点，就添加到模型中
        if new_points_list:
            new_attributes = {}
            for k in new_points_list[0].keys():
                new_attributes[k] = torch.cat([d[k] for d in new_points_list], dim=0)
            self.densification_postfix(**new_attributes)

        # --- 剪枝 ---
        # 1. 确定需要剪枝的旧点
        # clone_mask 和 split_mask 都是针对原始点集的，它们的长度都是原始点数
        # 我们需要剪掉那些被分裂的点，但保留被克隆的点
        prune_mask_split = torch.zeros_like(grads.squeeze(), dtype=torch.bool)
        if split_mask is not None:
            prune_mask_split = split_mask

        # 2. 确定那些因为不透明度或大小而需要剪枝的点
        prune_mask_opacity = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = torch.max(self.get_scaling, dim=1).values > 0.1 * extent
            prune_mask_size = torch.logical_or(big_points_vs, big_points_ws)
            prune_mask_opacity = torch.logical_or(prune_mask_opacity, prune_mask_size)

        # 3. 合并所有剪枝掩码
        # 注意：prune_mask_split 和 prune_mask_opacity 的长度可能不同！
        # prune_mask_split 对应增生前的点数
        # prune_mask_opacity 对应增生后的点数
        
        # 正确的做法是先剪掉分裂的旧点
        current_points_mask = torch.ones(len(self._xyz), dtype=torch.bool, device="cuda")
        num_old_points = len(prune_mask_split)
        # 只在前 num_old_points 个点上应用分裂剪枝
        current_points_mask[:num_old_points] = ~prune_mask_split
        self.prune_points( ~current_points_mask )

        # 然后在新点集上应用不透明度和大小剪枝
        prune_mask_opacity_final = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = torch.max(self.get_scaling, dim=1).values > 0.1 * extent
            prune_mask_size = torch.logical_or(big_points_vs, big_points_ws)
            prune_mask_opacity_final = torch.logical_or(prune_mask_opacity_final, prune_mask_size)
        
        self.prune_points(prune_mask_opacity_final)
        
        # 重置统计数据
        torch.cuda.empty_cache()
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        self.max_radii2D.zero_()

    def prune_points(self, mask):
        valid_points_mask = ~mask
        
        # 移除旧的参数
        self._xyz = nn.Parameter(self._xyz[valid_points_mask].requires_grad_(True))
        self._features = nn.Parameter(self._features[valid_points_mask].requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity[valid_points_mask].requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling[valid_points_mask].requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation[valid_points_mask].requires_grad_(True))

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        # 如果不是 SparseAdam，则必须重新构建优化器
        if self.optimizer_type != "sparse_adam":
            # 获取当前学习率等设置
            # 注意：这里简化了，理想情况下应从旧优化器中保存所有非张量状态
            # 但对于 Adam，重新设置学习率通常就足够了。
            xyz_lr = self.optimizer.param_groups[0]['lr']
            feature_lr = self.optimizer.param_groups[1]['lr']
            opacity_lr = self.optimizer.param_groups[2]['lr']
            scaling_lr = self.optimizer.param_groups[3]['lr']
            rotation_lr = self.optimizer.param_groups[4]['lr']

            l = [
                {'params': [self._xyz], 'lr': xyz_lr, "name": "xyz"},
                {'params': [self._features], 'lr': feature_lr, "name": "features"},
                {'params': [self._opacity], 'lr': opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': rotation_lr, "name": "rotation"}
            ]
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        else:
            # SparseAdam 的 prune 方法是正确的
            self.optimizer.prune(valid_points_mask)

        
    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"] 
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
