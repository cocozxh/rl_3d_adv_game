import os
from Extragradient import Extragradient
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import sys
import random

from torch.utils.data import DataLoader, Dataset
from skimage.io import imread
from PIL import Image
from torch import autograd

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    OpenGLPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    BlendParams,
    SoftSilhouetteShader
)

from MeshDataset import MeshDataset
from BackgroundDataset import BackgroundDataset
from darknet import Darknet
from loss import TotalVariation, dis_loss, calc_acc, TotalVariation_3d

from torchvision.utils import save_image
import random

# from faster_rcnn.config.eval_config import EvalConfig as Config
# from resnet18.resnet import *
from resnet18.resnet import *
from resnet18.MLP import *

class Generator_without_Mesh():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Create pytorch3D renderer
        self.renderer = self.create_renderer()

        # Datasets
        self.mesh_dataset = MeshDataset(config.mesh_dir, device, max_num=config.num_meshes)
        self.bg_dataset = BackgroundDataset(config.bg_dir, config.img_size, max_num=config.num_bgs)
        self.test_bg_dataset = BackgroundDataset(config.test_bg_dir, config.img_size, max_num=config.num_test_bgs)

        # Initialize adversarial patch
        self.patches = None
        self.idx = None

        # Yolo model:
        # self.dnet = Darknet(self.config.cfgfile)
        # self.dnet.load_weights(self.config.weightfile)
        # self.dnet = self.dnet.eval()
        # self.dnet = self.dnet.to(self.device)

        if self.config.patch_dir is not None:
            print(self.config.patch_dir)
            self.patches = torch.load(self.config.patch_dir + '/patch_save.pt').to(self.device)
            self.idx = torch.load(self.config.patch_dir + '/idx_save.pt').to(self.device)
        else:
            self.patches = self.initialize_patch()

        self.test_bgs = DataLoader(
          self.test_bg_dataset, 
          batch_size=1, 
          shuffle=True, 
          num_workers=1)
  
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10

        self.train_bgs = DataLoader(
            self.bg_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=1)
        
        # self.G = ResNet34().to(self.device)
        self.init_model()
        print(self.G.parameters)
        # generator_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-3)
        # self.optimizer = Extragradient(generator_optimizer, self.G.parameters())
        
    def init_model(self):
        model_list = ['mlp', 'resnet18', 'resnet34', 'resnet50']
        assert self.config.generator_model.lower() in model_list, f"Model Name does not exist"
        
        if self.config.generator_model.lower() == 'mlp':
            self.G = MLP(self.patches.size()[0]*3,self.patches.size()[0]*3).to(self.device)
        elif self.config.generator_model.lower() == 'resnet18':
            self.G = ResNet18().to(self.device)
        elif self.config.generator_model.lower() == 'resnet34':
            self.G = ResNet34().to(self.device)
        elif self.config.generator_model.lower() == 'resnet50':
            self.G = ResNet50().to(self.device)
        else:
            print("Please type the correct model name...")

    def reset(self):
        self.init_model()
    
    def create_images(self):
        total_images = []

        self.train_bgs = DataLoader(
            self.bg_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=1)
        
        for images in self.train_bgs:
            total_images.append(images.to(self.device))

        # print("Image Size:" + str(total_images[0].size())) 
        return total_images
    
    def update_patch(self, action):
        # update parameter and patch
        self.update_params(action)
        self.create_patch(self.patches)

    def create_patch(self, patch):
        if patch is None:
            adv_patch = self.initialize_patch().flatten()
        else:
            adv_patch = self.G(patch.permute(0, 3, 1, 2).flatten())
        return adv_patch
    
    def update_params(self, delta_named_params):
        new_params = {}
        for name, params in self.G.named_parameters():
            new_params[name] = params.add(torch.tensor(delta_named_params[name]).cuda())

        # leaf variable with requires_grad = True cannot used inplace operation
        for name, params in self.G.named_parameters():
            params.data.copy_(new_params[name])
            
   
    def initialize_patch(self):
        print('Initializing patches...')
        sampled_planes = np.load(self.config.idx).tolist()
        idx = torch.Tensor(sampled_planes).long().to(self.device)
        self.idx = idx
        patches = []
        for _ in range(self.config.patch_num):
            patches.append(torch.rand(len(sampled_planes), 1, 1, 3, device=(self.device), requires_grad=True))
        self.patches = patches[0]

    def create_renderer(self):
        self.num_angles_train = self.config.num_angles_train
        self.num_angles_test = self.config.num_angles_test

        azim_train = torch.linspace(-1 * self.config.angle_range_train, self.config.angle_range_train, self.num_angles_train)
        azim_test = torch.linspace(-1 * self.config.angle_range_test, self.config.angle_range_test, self.num_angles_test)

        # Cameras for SMPL meshes:
        camera_dist = 2.2
        R, T = look_at_view_transform(camera_dist, 6, azim_train)
        train_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.train_cameras = train_cameras

        R, T = look_at_view_transform(camera_dist, 6, azim_test)
        test_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.test_cameras = test_cameras
        
        raster_settings = RasterizationSettings(
            image_size=self.config.img_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        lights = PointLights(device=self.device, location=[[0.0, 85, 100.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=train_cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device, 
                cameras=train_cameras,
                lights=lights
            )
        )

        return renderer
    
    def change_cameras(self, mode, camera_dist=2.2):
        azim_train = torch.linspace(-1 * self.config.angle_range_train, self.config.angle_range_train, self.num_angles_train)
        azim_test = torch.linspace(-1 * self.config.angle_range_test, self.config.angle_range_test, self.num_angles_test)

        R, T = look_at_view_transform(camera_dist, 6, azim_train)
        train_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.train_cameras = train_cameras

        R, T = look_at_view_transform(camera_dist, 6, azim_test)
        test_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.test_cameras = test_cameras

        if mode == 'train':
            self.renderer.rasterizer.cameras=self.train_cameras
            self.renderer.shader.cameras=self.train_cameras
        elif mode == 'test':
            self.renderer.rasterizer.cameras=self.test_cameras
            self.renderer.shader.cameras=self.test_cameras

    def render_mesh_on_bg(self, mesh, bg_img, num_angles, location=None, x_translation=0, y_translation=0):
        images = self.renderer(mesh)
        bg = bg_img.unsqueeze(0)
        bg_shape = bg.shape
        new_bg = torch.zeros(bg_shape[2], bg_shape[3], 3)
        new_bg[:,:,0] = bg[0,0,:,:]
        new_bg[:,:,1] = bg[0,1,:,:]
        new_bg[:,:,2] = bg[0,2,:,:]

        human = images[:, ..., :3]
        
        human_size = self.renderer.rasterizer.raster_settings.image_size

        if location is None:
            dH = bg_shape[2] - human_size
            dW = bg_shape[3] - human_size
            location = (
                dW // 2 + x_translation,
                dW - (dW // 2) - x_translation,
                dH // 2 + y_translation,
                dH - (dH // 2) - y_translation
            )

        contour = torch.where((human == 1).cpu(), torch.zeros(1).cpu(), torch.ones(1).cpu())
        new_contour = torch.zeros(num_angles, bg_shape[2], bg_shape[3], 3)
        
        new_contour[:,:,:,0] = F.pad(contour[:,:,:,0], location, "constant", value=0)
        new_contour[:,:,:,1] = F.pad(contour[:,:,:,1], location, "constant", value=0)
        new_contour[:,:,:,2] = F.pad(contour[:,:,:,2], location, "constant", value=0)

        new_human = torch.zeros(num_angles, bg_shape[2], bg_shape[3], 3)
        new_human[:,:,:,0] = F.pad(human[:,:,:,0], location, "constant", value=0)
        new_human[:,:,:,1] = F.pad(human[:,:,:,1], location, "constant", value=0)
        new_human[:,:,:,2] = F.pad(human[:,:,:,2], location, "constant", value=0)

        final = torch.where((new_contour == 0).cpu(), new_bg.cpu(), new_human.cpu())
        return final

    def render_mesh_on_bg_batch(self, mesh, bg_imgs, num_angles,  location=None, x_translation=0, y_translation=0):
        num_bgs = bg_imgs.shape[0]

        images = self.renderer(mesh) # (num_angles, 416, 416, 4)
        images = torch.cat(num_bgs*[images], dim=0) # (num_angles * num_bgs, 416, 416, 4)

        bg_shape = bg_imgs.shape

        # bg_imgs: (num_bgs, 3, 416, 416) -> (num_bgs, 416, 416, 3)
        bg_imgs = bg_imgs.permute(0, 2, 3, 1)

        # bg_imgs: (num_bgs, 416, 416, 3) -> (num_bgs * num_angles, 416, 416, 3)
        bg_imgs = bg_imgs.repeat_interleave(repeats=num_angles, dim=0)

        # human: RGB channels of render (num_angles * num_bgs, 416, 416, 3)
        human = images[:, ..., :3]
        human_size = self.renderer.rasterizer.raster_settings.image_size

        if location is None:
            dH = bg_shape[2] - human_size
            dW = bg_shape[3] - human_size
            location = (
                dW // 2 + x_translation,
                dW - (dW // 2) - x_translation,
                dH // 2 + y_translation,
                dH - (dH // 2) - y_translation
            )

        contour = torch.where((human == 1), torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        new_contour = torch.zeros(num_angles * num_bgs, bg_shape[2], bg_shape[3], 3, device=self.device)
        
        new_contour[:,:,:,0] = F.pad(contour[:,:,:,0], location, "constant", value=0)
        new_contour[:,:,:,1] = F.pad(contour[:,:,:,1], location, "constant", value=0)
        new_contour[:,:,:,2] = F.pad(contour[:,:,:,2], location, "constant", value=0)

        new_human = torch.zeros(num_angles * num_bgs, bg_shape[2], bg_shape[3], 3, device=self.device)
        new_human[:,:,:,0] = F.pad(human[:,:,:,0], location, "constant", value=0)
        new_human[:,:,:,1] = F.pad(human[:,:,:,1], location, "constant", value=0)
        new_human[:,:,:,2] = F.pad(human[:,:,:,2], location, "constant", value=0)

        # output: (num_angles * num_bgs, 416, 416, 3)
        final = torch.where((new_contour == 0), bg_imgs, new_human)
        return final