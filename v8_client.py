import pygame
import cv2
import sys
import numpy as np

import os
import yaml
import time
import threading  # Import the threading module

import requests  # Import the requests library

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

# Pygame 초기화
pygame.init()

# 화면 해상도 설정
screen_width = config['SCREEN_WIDTH']
screen_height = config['SCREEN_HEIGHT']

if config['WINDOWED'] :
    screen = pygame.display.set_mode((screen_width, screen_height))
else :
    # 전체 화면 모드로 창 생성
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)

# 창 제목 설정
pygame.display.set_caption('Full Screen with OpenCV Frame')

# 색상 설정
black = (0, 0, 0)
white = (255, 255, 255)

# OpenCV 카메라 초기화
cap = cv2.VideoCapture(0)

detections = []

def send_frame_to_server(frame, callback):
    def task():
        response_data = send_frame_to_server_sync(frame)
        callback(response_data)
    threading.Thread(target=task).start()

def send_frame_to_server_sync(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    response = requests.post(f'http://{config["API_IP"]}:{config["API_PORT"]}/detect', files={'image': ('image.jpg', image_bytes)})
    return response.json()

def handle_server_response(response_data):
    global detections  # Assume detections is a global variable
    # print(response_data)
    detections = response_data.get('d', [])



#%% 메인 루프
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                cap.release()
                pygame.quit()
                sys.exit()
                

    # OpenCV로 프레임 캡처
    ret, frame = cap.read()
    
    # 좌우 반전
    frame = cv2.flip(frame, 1)
    
    if ret:
        # BGR에서 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 서버에 프레임 전송 및 결과 받기
        send_frame_to_server(frame, handle_server_response)
        
        # Detection 결과 그리기
        for detection in detections:
            class_name, conf, bbox = detection['class'], detection['confidence'], detection['bbox']
            
            img_height, img_width = frame_rgb.shape[:2]
            x1, y1, x2, y2 = [int(coord * dim) for coord, dim in zip(bbox, [img_width, img_height, img_width, img_height])]
            
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw bounding box
            label_text = f"{class_name} {conf}%"
            cv2.putText(frame_rgb, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 화면에 프레임 표시
        frame = np.rot90(frame_rgb)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(pygame.transform.scale(frame, (screen_width, screen_height)), (0, 0))

        # 화면 업데이트
    pygame.display.update()
# %%

# %%
