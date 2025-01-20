# ============================================
# File Name    : zed_display.py
# Author       : Sung Jun Kim
# Institution  : 한국과학기술연구원
# Email        : sungjun4257@kist.re.kr
# Created Date : 24.05.21
# Last Updated : 25.01.09
# Version      : 1.0
# 
# Purpose      : zed 카메라로부터 영상을 받아, 영상처리 및 시각화
#               
#
# Environment  :   Python 
#                - OpenCV 
#                - NumPy
#
# Revision Log :
#    - 
#
# ============================================


import sys
import pyzed.sl as sl
import numpy as np
import time 
import os
from collections import deque
import copy
import json
import SR.config as config
import SR.imgproc as imgproc
import SR.model as model
import torch


class ZEDCamera():
    def __init__(self, resolution='1080'):
        print("Initializing...")
        init = sl.InitParameters()

        for i in range(2):
            init.set_from_camera_id(i)
            
            # Set configuration parameters
            if resolution == '1080':
                init.camera_resolution = sl.RESOLUTION.HD1080
            elif resolution == '720':
                init.camera_resolution = sl.RESOLUTION.HD720
            else:
                print("Put a correct resolution")
                exit(0)

            # Available Mode : NONE, PERFORMANCE, QUALITY, ULTRA, NEURAL, NEURAL_PLUS
            init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            init.coordinate_units = sl.UNIT.MILLIMETER
            init.camera_fps = 60                            # Set fps at 60

            # Open the camera
            self.cam = sl.Camera()
            if not self.cam.is_opened():
                print("Opening ZED Camera...")
            status = self.cam.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                return
                exit()

            # Set runtime parameters after opening the camera
            self.runtime = sl.RuntimeParameters(enable_fill_mode=True)
        #    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

            self.camera_information = self.cam.get_camera_information()
            self.cam_model = str(self.camera_information.camera_model).rstrip()
            print("Camera Name : ", self.cam_model)

            # Get Image Size
            self.image_size = self.cam.get_camera_information().camera_configuration.resolution
            self.image_size.width = self.image_size.width 
            self.image_size.height = self.image_size.height 

            # Set Depth
            self.depth_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.F32_C1)
            # self.depth_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C4)


            break

            self.cam_close()
            print("Camera bringup was failed")
            exit(0)

        self.runtime = sl.RuntimeParameters()
        
        # for ZED SDK < 4.0
        # image_size = self.cam.get_camera_information().camera_resolution

        # for ZED SDK > 4.0
        image_size = self.camera_information.camera_configuration.resolution

        self.W, self.H = image_size.width, image_size.height
        self.roi = sl.Rect()
        self.select_in_progress = False
        self.origin_rect = (-1,-1 )

        self.mat_left = sl.Mat(self.W, self.H)
        self.mat_right = sl.Mat(self.W, self.H)

        self.BRIGHTNESS = sl.VIDEO_SETTINGS.BRIGHTNESS
        self.CONTRAST = sl.VIDEO_SETTINGS.CONTRAST
        self.EXPOSURE = sl.VIDEO_SETTINGS.EXPOSURE
        self.GAIN = sl.VIDEO_SETTINGS.GAIN
        self.GAMMA = sl.VIDEO_SETTINGS.GAMMA
        self.HUE = sl.VIDEO_SETTINGS.HUE
        self.SATURATION = sl.VIDEO_SETTINGS.SATURATION
        self.SHARPNESS = sl.VIDEO_SETTINGS.SHARPNESS
        self.WHITEBALANCE_TEMPERATURE = sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE
        self.SETTINGS = [self.BRIGHTNESS, self.CONTRAST, self.EXPOSURE, self.GAIN, self.GAMMA,
                         self.HUE, self.SATURATION, self.SHARPNESS, self.WHITEBALANCE_TEMPERATURE]

        
        self.init_camera_parameter()

    def init_camera_parameter(self):
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.HUE, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, -1)
        self.cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
        print("Camera parameters are initialized!")

    def set_camera_setting(self, setting, value):
        self.cam.set_camera_settings(setting, value)
    
    def get_camera_setting(self, setting):
        return self.cam.get_camera_settings(setting)
    

    def get_image(self):
        err = self.cam.grab(self.runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            self.cam.retrieve_image(self.mat_left, sl.VIEW.LEFT)
            left_image = self.mat_left.get_data()
            self.cam.retrieve_image(self.mat_right, sl.VIEW.RIGHT)
            right_image = self.mat_right.get_data()
            return left_image, right_image
        else:
            image = np.zeros((self.W, self.H))
            return image, image
    
    def get_depth(self):
        err = self.cam.grab(self.runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # Retrieve the normalized depth image
            self.cam.retrieve_image(self.depth_zed, sl.VIEW.DEPTH)
            # Use get_data() to get the numpy array
            image_depth_ocv = self.depth_zed.get_data()

            return image_depth_ocv
        else:
            image = np.zeros((self.W, self.H))
            return image

    def get_fps(self):
        return self.cam.get_current_fps()

    def cam_close(self):
        self.cam.close()
        print("\nFINISH")

    def check_status(self):
        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            return True
            exit()

        else:
            return False

if __name__ == "__main__":
    import cv2
    args = sys.argv

    cam = ZEDCamera(resolution='1080')
    cam.set_camera_setting(cam.EXPOSURE, -1)
    cam.set_camera_setting(cam.GAIN, -1)

    # SR
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

    # 검정 픽셀 추가를 위한 변수 초기화

    deque_buffer_left = deque()  # 전역 image_deque 설정
    deque_buffer_right = deque()  # 전역 image_deque 설정

    frame_count = 0 
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.moveWindow("Image", 2560, 0)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    save_flag = False
    direction = 0
    clahe_flag = False
    crop_flag = False
    sr_flag = False
    print_freq = 10
    save_dir = '/home/vision/test'
    if not os.path.exists('{0}'.format(save_dir)):
        os.mkdir(save_dir)
    
    video_length = 30.0

    json_path = "./ZED_info.json"
    if os.path.exists(json_path): 
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    
        if data["Exposure"] == -1:
            print("Setting Auto Exposure")
            exposure_auto = True
        else:
            print("Setting {} Exposure".format(data["Exposure"]))
            exposure_auto = False
            exposure = data["Exposure"]

        if data["Gain"] == -1:
            print("Setting Auto Gain")
            gain_auto = True
        else:
            print("Setting {} Gain".format(data["Gain"]))
            gain_auto = False
            gain = data["Gain"]

        offset = data["Offset"]
        print("Setting {} Offset".format(data["Offset"]))
    else:
        exposure_auto = True
        gain_auto = True
        exposure = -1
        gain = -1
        offset = 0  # 조정량 (양수: 왼쪽 위로, 오른쪽 아래로 이동)



    side_offset = 0
    while True:
        if exposure_auto:
            cam.set_camera_setting(cam.EXPOSURE, -1)
            exposure = cam.get_camera_setting(cam.EXPOSURE)[1]
        else:
            cam.set_camera_setting(cam.EXPOSURE, exposure)
    
        if gain_auto:
            cam.set_camera_setting(cam.GAIN, -1)
            gain = cam.get_camera_setting(cam.GAIN)[1]
        else:
            cam.set_camera_setting(cam.GAIN, gain)
        
        left, right = cam.get_image()
        # left = left[:,:,:3]
        # right = right[:,:,:3]
        left = cv2.cvtColor(left, cv2.COLOR_BGRA2BGR)
        right = cv2.cvtColor(right, cv2.COLOR_BGRA2BGR)

        if crop_flag:
            image_crop_size = 480
            left = imgproc.center_crop(left,image_crop_size)
            right = imgproc.center_crop(right,image_crop_size)
            if not sr_flag:
                left = cv2.resize(left,(960,960))
                right = cv2.resize(right,(960,960))

        if sr_flag and crop_flag:
            left_image = cv2.cvtColor(left, cv2.COLOR_BGR2YCR_CB)
            right_image = cv2.cvtColor(right, cv2.COLOR_BGR2YCR_CB)

            left_y , left_cr, left_cb = cv2.split(left_image)
            left_y = left_y.astype(np.float32)/255.
            right_y , right_cr, right_cb = cv2.split(right_image)
            right_y = right_y.astype(np.float32)/255.


            left_y = imgproc.image_to_tensor(left_y,range_norm=False,half=False).to(config.device,non_blocking=True)
            left_y = left_y.unsqueeze(0)
            right_y = imgproc.image_to_tensor(right_y,range_norm=False,half=False).to(config.device,non_blocking=True)
            right_y = right_y.unsqueeze(0)       

            batch = torch.cat([left_y,right_y],dim=0)

            with torch.no_grad():
                predicted_batch = g_model(batch)

            predicted_left_y = np.squeeze(np.transpose(predicted_batch[0].detach().cpu().numpy(),(1,2,0)))
            predicted_right_y = np.squeeze(np.transpose(predicted_batch[1].detach().cpu().numpy(),(1,2,0)))

            left_cb = cv2.resize(left_cb, (image_crop_size*2, image_crop_size*2), interpolation=cv2.INTER_LINEAR)
            left_cr = cv2.resize(left_cr, (image_crop_size*2, image_crop_size*2), interpolation=cv2.INTER_LINEAR)

            right_cb = cv2.resize(right_cb, (image_crop_size*2, image_crop_size*2), interpolation=cv2.INTER_LINEAR)
            right_cr = cv2.resize(right_cr, (image_crop_size*2, image_crop_size*2), interpolation=cv2.INTER_LINEAR)

            if predicted_left_y.dtype != np.uint8:
                predicted_left_y = np.clip(predicted_left_y * 255.0, 0, 255).astype(np.uint8)
                predicted_right_y = np.clip(predicted_right_y * 255.0, 0, 255).astype(np.uint8)

            left = cv2.merge([predicted_left_y,left_cb,left_cr])
            right = cv2.merge([predicted_right_y,right_cb,right_cr])

            left = cv2.cvtColor(left,cv2.COLOR_YCR_CB2RGB)
            right = cv2.cvtColor(right,cv2.COLOR_YCR_CB2RGB)

        if clahe_flag:
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(11, 11))
            # Color image
            ycrcb_array = cv2.cvtColor(left, cv2.COLOR_BGR2YCR_CB)
            y, cr, cb = cv2.split(ycrcb_array)
            merge_array = cv2.merge([clahe.apply(y), cr, cb])
            left = cv2.cvtColor(merge_array, cv2.COLOR_YCrCb2BGR)
          
            ycrcb_array = cv2.cvtColor(right, cv2.COLOR_BGR2YCR_CB)
            y, cr, cb = cv2.split(ycrcb_array)
            merge_array = cv2.merge([clahe.apply(y), cr, cb])
            right = cv2.cvtColor(merge_array, cv2.COLOR_YCrCb2BGR)

        if crop_flag:
            left_padding = 480 + side_offset  
            right_padding = 480 - side_offset 
            top_padding = 0
            left = cv2.copyMakeBorder(left,top_padding,top_padding,left_padding,right_padding,cv2.BORDER_CONSTANT,value=(0,0,0))
            right = cv2.copyMakeBorder(right,top_padding,top_padding,right_padding,left_padding,cv2.BORDER_CONSTANT,value=(0,0,0))
            left = cv2.resize(left,(1920,1080))
            right = cv2.resize(right,(1920,1080))

        # 검정 픽셀 추가 및 크기 유지 처리
        if offset != 0:
            black_strip = np.zeros((abs(offset), left.shape[1], left.shape[2]), dtype=left.dtype)
            
            if offset > 0:  # 왼쪽 이미지 아래로, 오른쪽 이미지 위로 이동
                left = np.vstack((left[offset:], black_strip))
                right = np.vstack((black_strip, right[:-offset]))

                left[:offset, :, :] = 0 
                right[-offset:, :, :] = 0  


            else:  # 왼쪽 이미지 위로, 오른쪽 이미지 아래로 이동
                left = np.vstack((black_strip, left[:offset]))
                right = np.vstack((right[-offset:], black_strip))

                left[offset:, :, :] = 0  
                right[:-offset, :, :] = 0  

        if direction == -1:
            image = left 
        elif direction == 0:
            left = cv2.resize(left, (cam.image_size.width//2, cam.image_size.height))
            right = cv2.resize(right, (cam.image_size.width//2, cam.image_size.height))

            image = cv2.hconcat([left, right])
        elif direction == 1:
            image = right

        cv2.imshow('Image' , image)
        

        key = cv2.waitKey(1)

        # offset 조정
        if key == ord('t'):  # 왼쪽 위로, 오른쪽 아래로 이동
            offset += 1
        elif key == ord('b'):  # 왼쪽 아래로, 오른쪽 위로 이동
            offset -= 1

        # exposure 조정
        elif key == ord('q'):
            exposure_auto = False
            exposure += 5
            if exposure > 100:
                exposure = 100
        elif key == ord('a'):
            exposure_auto = True
        elif key == ord('z'):
            exposure_auto = False
            exposure -= 5
            if exposure < 0:
                exposure = 0

        # gain 조정
        elif key == ord('e'):
            gain_auto = False
            gain += 5
            if gain > 100:
                gain = 100
        elif key == ord('d'):
            gain_auto = True
        elif key == ord('c'):
            gain_auto = False
            gain -= 5
            if gain < 0: 
                gain = 0
        elif key == ord('1'):
            direction = -1
        elif key == ord('2'):
            direction = 0 
        elif key == ord('3'):
            direction = 1
        # 엔터 누르면 관련 정보를 담은 json 파일 저장
        elif key == 11 or key == 13:  
            json_data = {}
            if exposure_auto:
                json_data['Exposure'] = -1
            else:
                json_data['Exposure'] = exposure
            
            if gain_auto:
                json_data['Gain'] = -1
            else:
                json_data['Gain'] = gain
    
            json_data['Offset'] = offset
            with open('./ZED_info.json', 'w') as info:
                    json.dump(json_data, info, indent=4)
            print("Save Compled zed data : ", json_data)
        elif key == 27:
                break
        elif key == ord('l'):
            clahe_flag = not clahe_flag
        elif key == ord('k'):
            sr_flag = not sr_flag
        elif key == ord('j'):
            crop_flag = not crop_flag
        elif key == ord('f'):
            side_offset -= 1
        elif key ==ord('h'):
            side_offset += 1
        frame_count = frame_count + 1

        if frame_count > print_freq:
            frame_count = 0
            FPS = cam.get_fps()

            print("==== InFo ====")
            print("Current FPS : ",FPS)
            print("Offset      : ",offset)
            print("Side Offset : ",side_offset)
            print("Exposure    : ",exposure)
            print("Gain        : ",gain)
    cam.cam_close()
