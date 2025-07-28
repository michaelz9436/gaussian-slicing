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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

# 1. 引入新的数据加载器
from scene.slice_loader import readSliceData 
sceneLoadTypeCallbacks["SliceData"] = readSliceData



class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # --- 加载检查点逻辑 (不变) ---
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        # 2. 修改场景识别和加载逻辑

        self.slice_data_image_folder = args.images if args.images is not None else 'images'
        

        self.slice_z_scale = 1 #每个切片间隔1.0个单位。可以根据需要修改。
        
        scene_info = None
        
        # 核心修改：添加对我们切片数据的识别逻辑
        slice_data_path = os.path.join(args.source_path, self.slice_data_image_folder)
        if os.path.exists(slice_data_path):
            print(f"Found slice data folder ({self.slice_data_image_folder}), assuming SliceData format!")
            # 调用加载器
            scene_info = sceneLoadTypeCallbacks["SliceData"](args.source_path, self.slice_data_image_folder, args.eval, self.slice_z_scale)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, f"Could not recognize scene type! Neither Colmap, Blender, nor SliceData format was found in {args.source_path}"

        # --- 初始设置逻辑 (为数据模型简化) ---
        if not self.loaded_iter:
            if scene_info.ply_path:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            # json_cams 和 camera.json 的逻辑无用

        if shuffle:
            # train_cameras 列表现在包含的是 SlicePlane 对象
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        #  SlicePlane 对象已经包含了所有需要的信息，所以这里可以直接赋值。
        self.cameras_extent = scene_info.nerf_normalization["radius"] if scene_info.nerf_normalization else 1.0

        for resolution_scale in resolution_scales:
            print("Loading Training SlicePlanes")
            self.train_cameras[resolution_scale] = scene_info.train_cameras
            print("Loading Test SlicePlanes")
            self.test_cameras[resolution_scale] = scene_info.test_cameras

        # 4. 修改高斯模型创建逻辑
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:

            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        # 保存逻辑基本不变
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        # 返回的是 SlicePlane 对象的列表
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        # 返回的是 SlicePlane 对象的列表
        return self.test_cameras[scale]