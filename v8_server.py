#%%
from flask import Flask, request, jsonify
from ultralytics import YOLO,checks
import cv2 as cv

import os
import yaml
import time
import pygame
import numpy as np

print(f'opencv version {cv.__version__}')
checks()


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

# %% model load
try :
    model = YOLO(config['WEIGHT'])  # load a pretrained YOLOv8n detection model
    
    print(f'model loaded success : {config["WEIGHT"]}')
    print('model names : ',model.names)
    # Load class names
    class_names = model.names
    
    # Warm-up inference
    print('try Model warmed up.')
    _start_time = time.time()
    warm_up_img = np.zeros((640, 640, 3), dtype=np.uint8)  # Creating a blank image with dimensions 416x416
    print('Warming up the model...')
    model(warm_up_img)  # Performing a warm-up inference
    print(f'Model warmed up. ok : {time.time() - _start_time:.3f}s')
    
except Exception as e:
    print(e)
    print('model load failed')
    quit()

#%% 

app = Flask(__name__)

@app.route('/about', methods=['GET'])
def about():
    about_text = """
    This is a web server for object detection using YOLOv8.
    Send a POST request with an image to /detect to get the detection results.
    """
    return jsonify({'r':'ok','msg':about_text})

@app.route('/detect', methods=['POST'])
def detect_objects():
    image_file = request.files['image']
    image_np = cv.imdecode(np.fromstring(image_file.read(), np.uint8), cv.IMREAD_UNCHANGED)
    
    frame = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)  # Convert to RGB as YOLO uses RGB
    results = model(source=frame, conf=0.25, verbose=False)
    
    img_height, img_width = frame.shape[:2]  # Get the image dimensions
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Convert pixel coordinates to ratios
            x1_ratio, y1_ratio = x1 / img_width, y1 / img_height
            x2_ratio, y2_ratio = x2 / img_width, y2 / img_height
            
            
            class_id = int(box.cls.cpu().item())
            class_name = class_names[class_id]
            conf = int(box.conf.cpu().item() * 100)
            detections.append({'class': class_name, 'confidence': conf, 'bbox': [x1_ratio, y1_ratio, x2_ratio, y2_ratio]})
    
    return jsonify({'r':'ok','d':detections} )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config['API_PORT'], debug=False)



# %%
