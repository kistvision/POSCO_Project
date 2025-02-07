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
# ============================================================================
import os
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

import config
import model
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter

import matplotlib.pyplot as plt

from datetime import datetime


def plot_and_save_loss_curve(time, results_dir, loss_save):

    results_dir = os.path.join(results_dir, time)
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


    plt.figure(figsize=(10, 6))
    plt.plot(loss_save, label="Training Loss")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 그래프 저장
    plot_path = os.path.join(results_dir, 'loss_curve.png')
    plt.savefig(plot_path)

    
    plt.show()
    
    plt.close()
    print(f"Loss curve saved at {plot_path}")

def save_results_to_txt(time, results_dir, epoch, best_psnr, best_ssim, loss):
    # 결과 파일 저장 경로

    results_dir = os.path.join(results_dir, time)
    

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    result_file = os.path.join(results_dir, 'training_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Best PSNR: {best_psnr}\n")
        f.write(f"Best SSIM: {best_ssim}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Loss: {loss}\n")
        f.write(f"Learning Rate: {config.model_lr}\n")
        f.write(f"Batch_size: {config.batch_size}\n")
        f.write(f"Train file: {config.train_gt_images_dir}\n")
        f.write(f"test_gt_images_dir: {config.test_gt_images_dir}\n")
        f.write(f"test_lr_images_dir: {config.test_lr_images_dir}\n")



    print(f"Training results saved to {result_file}")


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): 검증 손실이 개선된 후 기다리는 에포크 수
            verbose (bool): 조기 종료 메시지 출력 여부
            delta (float): 개선으로 간주되는 최소 변화량
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, BEST : {self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    loss_save = []

    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    espcn_model = build_model()
    print(f"Build `{config.model_arch_name}` model successfully.")

    trainable_params = sum(p.numel() for p in espcn_model.parameters() if p.requires_grad)
    print("trainable_params", trainable_params)

    criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(espcn_model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler successfully.")

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_weights_path:
        espcn_model = load_state_dict(espcn_model, config.pretrained_model_weights_path, load_mode="pretrained")
        print(f"Loaded `{config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if config.resume_model_weights_path:
        espcn_model, _, start_epoch, best_psnr, best_ssim, optimizer, _ = load_state_dict(
            espcn_model,
            config.resume_model_weights_path,
            optimizer=optimizer,
            load_mode="resume")
        print("Loaded pretrained model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    # writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    psnr_model = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim_model = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=config.device)
    ssim_model = ssim_model.to(device=config.device)

    # Set Early stopping
    early_stopping = EarlyStopping(patience=100)

    for epoch in range(start_epoch, config.epochs):
        loss = train(espcn_model,
              train_prefetcher,
              criterion,
              optimizer,
              epoch,
              scaler,
              )
        loss_save.append(loss.item())

        psnr, ssim = validate(espcn_model,
                              test_prefetcher,
                              epoch,
                              psnr_model,
                              ssim_model,
                              "Test")
        print("\n")

        # Update lr
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": espcn_model.state_dict(),
                         "optimizer": optimizer.state_dict()},
                        f"g_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)
        
        early_stopping(psnr)

        if early_stopping.early_stop:
            print("Early Stopping!!!")
            break

    now = datetime.now()
    time = now.strftime(now.strftime('%Y-%m-%d_%H:%M:%S'))
    save_results_to_txt(time,results_dir,epoch+1,best_psnr,best_ssim,loss)
    plot_and_save_loss_curve(time,results_dir,loss_save)

def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.train_gt_images_dir,
                                            config.gt_image_size,
                                            config.upscale_factor,
                                            "Train")
    test_datasets = TestImageDataset(config.test_gt_images_dir, config.test_lr_images_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return train_prefetcher, test_prefetcher


def build_model() -> nn.Module:
    espcn_model = model.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                         out_channels=config.out_channels,
                                                         channels=config.channels)
    espcn_model = espcn_model.to(device=config.device)

    return espcn_model


def define_loss() -> nn.MSELoss:
    criterion = nn.MSELoss()
    criterion = criterion.to(device=config.device)

    return criterion


def define_optimizer(espcn_model) -> optim.SGD:
    optimizer = optim.SGD(espcn_model.parameters(),
                          lr=config.model_lr,
                          momentum=config.model_momentum,
                          weight_decay=config.model_weight_decay,
                          nesterov=config.model_nesterov)

    return optimizer


def define_scheduler(optimizer) -> lr_scheduler.MultiStepLR:
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_scheduler_milestones,
                                         gamma=config.lr_scheduler_gamma)

    return scheduler


def train(
        espcn_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.MSELoss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
) -> None:
    
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    espcn_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device=config.device, non_blocking=True)
        lr = batch_data["lr"].to(device=config.device, non_blocking=True)
        # print("gt ", gt.shape)
        # print("lr ",lr.shape)
        # Initialize generator gradients
        espcn_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = espcn_model(lr)
            loss = torch.mul(config.loss_weights, criterion(sr, gt))

        # Backpropagation
        scaler.scale(loss).backward()
        # Gradient zoom + gradient clipping
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(espcn_model.parameters(), max_norm=config.clip_gradient / optimizer.param_groups[0]["lr"], norm_type=2.0)
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:
            # Record loss during training and output to file
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1

    return loss


def validate(
        espcn_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        mode: str
) -> [float, float]:
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    espcn_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            gt = batch_data["gt"].to(device=config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)

            # Use the generator model to generate a fake sample
            with amp.autocast():
                sr = espcn_model(lr)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % config.test_print_frequency == 0:
                progress.display(batch_index + 1)


            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    return psnres.avg, ssimes.avg


if __name__ == "__main__":
    main()
