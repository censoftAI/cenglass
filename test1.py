#%%
import cv2 as cv
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from IPython.display import display
import os
import yaml
import time


import numpy as np

from ultralytics import YOLO,checks
print(f'opencv version {cv.__version__}')
checks()

import pygame

#%% read config file yaml
# Read config file yaml
config_file_path = 'config.yaml'
sample_config_file_path = 'sample.config.yaml'

# Check if config.yaml exists, if not create one from sample.config.yaml
if not os.path.exists(config_file_path):
    if os.path.exists(sample_config_file_path):
        with open(sample_config_file_path, 'r') as sample_f:
            sample_config = sample_f.read()
        with open(config_file_path, 'w') as new_f:
            new_f.write(sample_config)
        print(f"{config_file_path} created from {sample_config_file_path}")
    else:
        print(f"{sample_config_file_path} does not exist. Cannot create {config_file_path}")
else:
    print(f"{config_file_path} already exists.")

# Read the actual config file
with open(config_file_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config)



# %%
try :
    model = YOLO(config['WEIGHT'])  # load a pretrained YOLOv8n detection model
    
    print(f'model loaded success : {config["WEIGHT"]}')
    print('model names : ',model.names)
    # Load class names
    class_names = model.names
    
except Exception as e:
    print(e)
    print('model load failed')
    quit()