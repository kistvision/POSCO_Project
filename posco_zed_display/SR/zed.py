import sys
import pyzed.sl as sl
import numpy as np
import time 

# Update Date : 24.09.11
# Sung Jun Kim
import config
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
import torch
import imgproc

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
                exit()

            # Set runtime parameters after opening the camera
            self.runtime = sl.RuntimeParameters(enable_fill_mode=True)
        #    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

            self.camera_information = self.cam.get_camera_information()
            self.cam_model = str(self.camera_information.camera_model).rstrip()
            print("Camera Name : ", self.cam_model)

            # Get Image Size
            self.image_size = self.cam.get_camera_information().camera_configuration.resolution
            self.image_size.width = self.image_size.width /2
            self.image_size.height = self.image_size.height /2

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

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))

if __name__ == "__main__":
    import cv2
    args = sys.argv

    cam = ZEDCamera()
    cam.set_camera_setting(cam.EXPOSURE, -1)
    cam.set_camera_setting(cam.GAIN, -1)
    
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

    # Start the verification mode of the bsrgan_model.
    g_model.eval()
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    # cv2.moveWindow("Image", 2560, 0)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    i = 0 
    while True:
        left, right = cam.get_image()
        
        
        # depth = cam.get_depth()
        # cv2.imshow('Depth ', left)
        left = cv2.cvtColor(left, cv2.COLOR_BGRA2BGR)
        right = cv2.cvtColor(right, cv2.COLOR_BGRA2BGR)
        FPS = cam.get_fps()
        print("Current FPS : ",FPS)


        image_crop_size = 480
        left_image = imgproc.center_crop(left,image_crop_size)
        right_image = imgproc.center_crop(right,image_crop_size)

        left_image = left_image.astype(np.float32)/255.
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2YCR_CB)
        right_image = right_image.astype(np.float32)/255.
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2YCR_CB)

        left_y , left_cr, left_cb = cv2.split(left_image)
        right_y , right_cr, right_cb = cv2.split(right_image)

        
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

                    
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(11, 11))
        predicted_left_y = clahe.apply(predicted_left_y)
        predicted_right_y = clahe.apply(predicted_right_y)

        predicted_left_y = predicted_left_y.astype(np.float32)/255.
        predicted_right_y = predicted_right_y.astype(np.float32)/255.


        left_image = cv2.merge([predicted_left_y,left_cb,left_cr])
        right_image = cv2.merge([predicted_right_y,right_cb,right_cr])

                # image = imgproc.ycbcr_to_rgb(image)
        left_image = cv2.cvtColor(left_image,cv2.COLOR_YCR_CB2RGB)
        right_image = cv2.cvtColor(right_image,cv2.COLOR_YCR_CB2RGB)



                # image = imgproc.histogram_clahe_bgr(image)
    

        image_concat = cv2.hconcat([left_image, right_image])
        
        cv2.imshow('Image' , image_concat)       
        # cv2.imwrite(f'./image/left_{i}.png', left)
        # cv2.imwrite(f'./image/right_{i}.png', right)
        # cv2.imwrite(f'./image/depth_{i}.png', depth)

        i = i + 1

        cv2.waitKey(1)

    cam.cam_close()