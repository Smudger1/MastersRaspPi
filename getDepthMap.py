import cv2 as cv
import torch

import sys
import os
sys.path.append('./Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

def getDepthMap(frame):
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vits' # or 'vits', 'vitb'
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    model.eval()

    depth = model.infer_image(frame) # HxW depth map in meters in numpy

    return depth

