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
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "ESPCN_x2-T91+gau_0.9"

if mode == "train":

    train_gt_images_dir = f"./data/T91+1_zoom/ESPCN/train"

    test_gt_images_dir = f"./data/Set5+5/GTmod12"
    test_lr_images_dir = f"./data/Set5+5/LRbicx{upscale_factor}_resize"

    gt_image_size = int(17 * upscale_factor)   # original


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


    # Test data address
    lr_dir = f"./data/Test/LRbicx{upscale_factor}"
    gt_dir = "./data/Test/GTmod12"
  

    model_weights_path = f"./results/{exp_name}/g_best.pth.tar"
