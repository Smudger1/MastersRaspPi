import cv2 as cv
import numpy as np
import torch

import sys
import os
sys.path.append('./Depth-Anything-V2/metric_depth')
from depth_anything_v2.dpt import DepthAnythingV2

def getDepthMap(frame):
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    encoder = 'vits' # or 'vits', 'vitb'
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model 

    checkpoint_path = f'./Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth'
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        return None

    print(f"Loading DepthAnythingV2 model with encoder: {encoder}, dataset: {dataset}")
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    print("Model loaded successfully.")
    print("Processing image for depth estimation...")
    image = cv.imread('./objectDetectionImgs/frame1.jpg')  # Load the image

    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    depth = model.infer_image(image) # HxW depth map in meters in numpy

    print(f"Depth map shape: {depth.shape}, dtype: {depth.dtype}")


    depth_normalised = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX)
    depth_scaled = depth_normalised.astype(np.uint8)
    depth_colored = cv.applyColorMap(depth_scaled, cv.COLORMAP_JET)
    cv.imwrite("./objectDetectionImgs/depthMap.jpg", depth_colored)

    return depth

