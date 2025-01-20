# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import os

import cv2
import numpy as np
import torch
from natsort import natsorted

import imgproc
import model
import config
from utils import make_directory
import time

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main() -> None:
    # Initialize the super-resolution bsrgan_model
    g_model = model.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                     out_channels=config.out_channels,
                                                     channels=config.channels)
    g_model = g_model.to(device=config.device)
    print(f"Build `{config.model_arch_name}` model successfully.")
    # Load the super-resolution bsrgan_model weights
    checkpoint = torch.load(config.model_weights_path, map_location=lambda storage, loc: storage)
    g_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{config.model_arch_name}` model weights "
          f"`{os.path.abspath(config.model_weights_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    make_directory(config.sr_dir)

    # Start the verification mode of the bsrgan_model.
    g_model.eval()


    # Set the sharpness evaluation function calculation device to the specified model
    # psnr = psnr.to(device=config.device, non_blocking=True)
    # ssim = ssim.to(device=config.device, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0

    # Get a list of test image file names.
    # file_names = natsorted(os.listdir(config.lr_dir))
    # Get the number of test image files.
    # total_files = len(file_names)

    # for index in range(total_files):
    #     lr_image_path = os.path.join(config.lr_dir, file_names[index])
    #     sr_image_path = os.path.join(config.sr_dir, file_names[index])
    #     gt_image_path = os.path.join(config.gt_dir, file_names[index])

    #     print(f"Processing `{os.path.abspath(lr_image_path)}`...")
    #     gt_y_tensor, gt_cb_image, gt_cr_image = imgproc.preprocess_one_image(gt_image_path, config.device)
    #     lr_y_tensor, lr_cb_image, lr_cr_image = imgproc.preprocess_one_image(lr_image_path, config.device)
    #     lr_cb_image = imgproc.image_resize(lr_cb_image,config.upscale_factor)
    #     lr_cr_image = imgproc.image_resize(lr_cr_image,config.upscale_factor)

    #     # Only reconstruct the Y channel image data.
    #     with torch.no_grad():
    #         sr_y_tensor = g_model(lr_y_tensor)

    #     # Save image
    #     sr_y_image = imgproc.tensor_to_image(sr_y_tensor, range_norm=False, half=True)
    #     sr_y_image = sr_y_image.astype(np.float32) / 255.0
    #     sr_ycbcr_image = cv2.merge([sr_y_image, lr_cb_image, lr_cr_image])
    #     sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
    #     cv2.imwrite(sr_image_path, sr_image * 255.0)
        
    #     if file_names[index] == "furnace.png":
    #         image_path = os.path.join(config.sr_dir, "crop_" + file_names[index])
    #         print(image_path)
    #         sr_image = imgproc.crop_image(sr_image,120,200,150)
    #         cv2.imwrite(image_path, sr_image * 255.0)


    #     print("=> PSNR",psnr(sr_y_tensor, gt_y_tensor).item())
    #     print("=> SSIM",ssim(sr_y_tensor, gt_y_tensor).item())

    #     # Cal IQA metrics
    #     psnr_metrics += psnr(sr_y_tensor, gt_y_tensor).item()
    #     ssim_metrics += ssim(sr_y_tensor, gt_y_tensor).item()

    # # Calculate the average value of the sharpness evaluation index,
    # # and all index range values are cut according to the following values
    # # PSNR range value is 0~100
    # # SSIM range value is 0~1
    # avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    # avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files

    # print(f"PSNR: {avg_psnr:4.2f} [dB]\n"
    #       f"SSIM: {avg_ssim:4.4f} [u]")




    # Test Video
    Vid = cv2.VideoCapture(config.video_path)
    if Vid.isOpened():
        fps = Vid.get(cv2.CAP_PROP_FPS)
        f_count = Vid.get(cv2.CAP_PROP_FRAME_COUNT)
        f_width = Vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        f_height = Vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print('Frames per second : ',fps,'FPS')
        print('Frame count : ',f_count)
        print('Frame width : ',f_width)
        print('Frame height: ',f_height)

        frame_count = 0
        start_time = time.time()

        output_video_path = './sidebyside_video_crop_sr_clahe3.mp4'

        # 출력 비디오 파일 설정 (코덱과 확장자 주의)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(960*2), int(480*2)))


        # while Vid.isOpened():
        #     ret, frame = Vid.read() # image 를 BGR 순으로 읽어온다.
        #     if ret:
        #         image_crop_size = 480
        #         image = imgproc.center_crop(frame,image_crop_size)
        #         # image = cv2.resize(image, (120, 120), interpolation=cv2.INTER_LINEAR)
        #         # image = cv2.resize(image, (240, 240), interpolation=cv2.INTER_LINEAR)

        #         image = image.astype(np.float32)/255.
        #         ori_image = image
        #         image = imgproc.bgr_to_ycbcr(image, only_use_y_channel=False)
        #         y , cb, cr = cv2.split(image)
        #         image = y
        #         image = imgproc.image_to_tensor(image,range_norm=False,half=False).to(config.device,non_blocking=True)
        #         image = image.unsqueeze(0)
                
                            
                
        #         predicted_image = g_model(image)
        
        #         predicted_image = np.squeeze(np.transpose(predicted_image[0].detach().cpu().numpy(),(1,2,0)))
        #         cb = cv2.resize(cb, (image_crop_size*2, image_crop_size*2), interpolation=cv2.INTER_LINEAR)
        #         cr = cv2.resize(cr, (image_crop_size*2, image_crop_size*2), interpolation=cv2.INTER_LINEAR)
        #         # predicted_image = cv2.resize(predicted_image, (240, 240), interpolation=cv2.INTER_LINEAR)

        #         image = cv2.merge([predicted_image,cb,cr])
        #         image = imgproc.ycbcr_to_rgb(image)


        #         if image.dtype != np.uint8:
        #             image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        #         if ori_image.dtype != np.uint8:
        #             ori_image = np.clip(ori_image * 255.0, 0, 255).astype(np.uint8)

        #         image = imgproc.histogram_clahe_bgr(image)

        #         image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        #         ##
        #         # scaled_ori = cv2.resize(ori_image, dsize=(960, 960), interpolation=cv2.INTER_LINEAR)
        #         # scaled_img = cv2.resize(image, dsize=(960, 960), interpolation=cv2.INTER_LINEAR)

        #         # numpy_horizontal = np.concatenate((scaled_ori,scaled_img),axis=1) # 원본 이미지, 딥러닝 적용이미지 붙여서 보기 편하게
        #         ##
                
        #         scaled_ori = cv2.resize(ori_image, dsize=(image_crop_size*2, image_crop_size*2), interpolation=cv2.INTER_LINEAR)
        #         numpy_horizontal = np.concatenate((scaled_ori,image),axis=1) # 원본 이미지, 딥러닝 적용이미지 붙여서 보기 편하게
                
        #         out.write(numpy_horizontal)

        #         cv2.imshow('Video',numpy_horizontal)
        #         frame_count += 1
        #         current_time = time.time()
        #         elapsed_time = current_time - start_time
                
        #         if elapsed_time > 1:
        #             fps = frame_count / elapsed_time
        #             print("FPS ",fps)
        #             frame_count = 0
        #             start_time = time.time()


        #         key=cv2.waitKey(10)
        #         if key == ord('q'):
        #             break
        #     else:
        #         break


        # 경량화 버전
        fps_list = []
        while Vid.isOpened():
            ret, frame = Vid.read() # image 를 BGR 순으로 읽어온다.
            if ret:
                image_crop_size = 480
                image = imgproc.center_crop(frame,image_crop_size)
                # image = cv2.resize(image, (120, 120), interpolation=cv2.INTER_LINEAR)
                # image = cv2.resize(image, (240, 240), interpolation=cv2.INTER_LINEAR)

                image = image.astype(np.float32)/255.
                # image = imgproc.bgr_to_ycbcr(image, only_use_y_channel=False)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)

                y , cr, cb = cv2.split(image)

            
                image = imgproc.image_to_tensor(y,range_norm=False,half=False).to(config.device,non_blocking=True)
                image = image.unsqueeze(0)
                
                            
                
                predicted_image = g_model(image)
        
                predicted_image = np.squeeze(np.transpose(predicted_image[0].detach().cpu().numpy(),(1,2,0)))
                cb = cv2.resize(cb, (image_crop_size*2, image_crop_size*2), interpolation=cv2.INTER_LINEAR)
                cr = cv2.resize(cr, (image_crop_size*2, image_crop_size*2), interpolation=cv2.INTER_LINEAR)
                # predicted_image = cv2.resize(predicted_image, (240, 240), interpolation=cv2.INTER_LINEAR)
                if predicted_image.dtype != np.uint8:
                    predicted_image = np.clip(predicted_image * 255.0, 0, 255).astype(np.uint8)

                    
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(11, 11))
                predicted_image = clahe.apply(predicted_image)

                predicted_image = predicted_image.astype(np.float32)/255.


                image = cv2.merge([predicted_image,cb,cr])

                # image = imgproc.ycbcr_to_rgb(image)
                image = cv2.cvtColor(image,cv2.COLOR_YCR_CB2BGR)



                # image = imgproc.histogram_clahe_bgr(image)
    
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                                
                cv2.imshow('Video',image)

                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                if elapsed_time > 1:
                    fps = frame_count / elapsed_time
                    print("FPS ",fps)
                    frame_count = 0
                    start_time = time.time()


                key=cv2.waitKey(10)
                if key == ord('q'):
                    break
            else:
                break

    Vid.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
