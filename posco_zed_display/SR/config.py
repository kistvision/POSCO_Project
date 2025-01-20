# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = False
# Model architecture name
model_arch_name = "espcn_x2"
# Model arch config
in_channels = 1
out_channels = 1
channels = 64
upscale_factor = 2
# Current configuration parameter method
mode = "test"
# Experiment name, easy to save weights and log files
# exp_name = "ESPCN_x4-T91"
# exp_name = "ESPCN_x2-T91+now_random_resizes"
exp_name = "ESPCN_x2-T91+gau_0.9"

if mode == "train":
    # Dataset address
    # train_gt_images_dir = f"./data/T91/ESPCN/train"
    # train_gt_images_dir = f"./data/BSDS100/ESPCN/train"
    # train_gt_images_dir = f"./data/Set5/GTmod12"
    # train_gt_images_dir = f"./data/Furnace/ESPCN/train"
    # train_gt_images_dir = f"./data/T91/ESPCN/train"
    # train_gt_images_dir = f"./data/Furnace_SmartPhone2/ESPCN/train"
    # train_gt_images_dir = f"./data/T91+1_gau/ESPCN/train"
    train_gt_images_dir = f"./data/T91/ESPCN/train"
    # train_gt_images_dir = f"./data/T91+Fur/ESPCN/train"

    # test_gt_images_dir = f"./data/Set5/GTmod12"
    # test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"
    # test_gt_images_dir = f"./data/Furnace_SmartPhone_Test/GT"
    # test_lr_images_dir = f"./data/Furnace_SmartPhone_Test/image_%{upscale_factor}"
    # test_gt_images_dir = f"./data/Set5+5/GTmod12"
    # test_lr_images_dir = f"./data/Set5+5/LRbicx{upscale_factor}_gau"
    test_gt_images_dir = f"./data/Set5+5/GTmod12"
    test_lr_images_dir = f"./data/Set5+5/LRbicx{upscale_factor}_resize"

    # gt_image_size = 120   # 
    gt_image_size = int(17 * upscale_factor)   # original

    # gt_image_size = int(16 * upscale_factor) 
    # gt_image_size = int(16 * upscale_factor) 

    # batch_size = 16 # original
    batch_size = 16
    num_workers = 4

    # The address to load the pretrained model
    pretrained_model_weights_path = f""
    # Incremental training and migration training
    resume_model_weights_path = f""

    # Total num epochs
    epochs = 3000

    # loss function weights
    loss_weights = 1.0

    # Optimizer parameter
    model_lr = 0.1 # 10-2 / 0.01
    model_momentum = 0.9
    model_weight_decay = 1e-4 #10-4 / 0.0001
    model_nesterov = False

    # gradient clipping constant
    clip_gradient = 0.01

    # EMA parameter
    # model_ema_decay = 0.999

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.1), int(epochs * 0.8)]
    lr_scheduler_gamma = 0.1

    # How many iterations to print the training result
    train_print_frequency = 100
    test_print_frequency = 1

if mode == "test":
    sr_dir = f"./results/test/{exp_name}"

    # Gaussian Test data address
    # lr_dir = f"./data/Set5+5/LRbicx{upscale_factor}_gau"
    # gt_dir = "./data/Set5+5/GTmod12"

    # Resize Test data address
    # lr_dir = f"./data/Set5+5/LRbicx{upscale_factor}_resize"
    # gt_dir = "./data/Set5+5/GTmod12"

    # Test data address
    lr_dir = f"./data/Test/LRbicx{upscale_factor}"
    gt_dir = "./data/Test/GTmod12"
  
    # lr_dir = f"./data/Furnace_SmartPhone_Test/image_%{upscale_factor}"
    # sr_dir = f"./results/test/{exp_name}"
    # gt_dir = "./data/Furnace_SmartPhone_Test/GT"
 
    
    # lr_dir = f"./data/Set5+4/LRbicx{upscale_factor}_test"
    # sr_dir = f"./results/test/{exp_name}"
    # gt_dir = "./data/Set5+4/GTmod12_test"

    # model_weights_path = "./results/pretrain/ESPCN_x2-BSD100_gau/g_best.pth.tar"
    # model_weights_path = "./results/pretrain/ESPCN_x2-T91-da809cd7.pth.tar"
    # model_weights_path = "./results/pretrain/g_best.pth.tar"
    model_weights_path = f"./results/{exp_name}/g_best.pth.tar"
    # model_weights_path = f"./results/ESPCN_x3-T91/g_best.pth.tar"
    # model_weights_path = f"./results/ESPCN_x4-T91/g_best.pth.tar"

    # model_weights_path = f"./results/ESPCN_x2-191_gau/g_best.pth.tar"

    # model_weights_path = f"./results/ESPCN_x2-BSD100_no_gau/g_best.pth.tar"
    # model_weights_path = f"./results/ESPCN_x2-BSD100_gau/g_best.pth.tar"

    # video_path = "/home/vision/packages/SR/dataset/광양제철소/zed_stereo_video/2i/video_right.mp4"
    # video_path = "/home/vision/packages/SR/dataset/광양제철소/dataset_hr/video1.mp4"
    video_path = "/home/vision/packages/video_right_2i.mp4"