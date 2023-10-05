#%%
from ultralytics import YOLO,checks
import cv2 as cv

import os
import yaml
import time
import pygame
import numpy as np

import threading
import queue

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


# 공유 데이터 큐 생성
data_queue = queue.Queue()
cap_data_queue = queue.Queue(maxsize=1)

msg_queue = queue.Queue()

 #종료 이벤트 객체 생성
shutdown_event = threading.Event()

def inference_task():
    try :
        model = YOLO(config['WEIGHT'])  # load a pretrained YOLOv8n detection model
        
        print(f'model loaded success : {config["WEIGHT"]}')
        print('model names : ',model.names)
        # Load class names
        class_names = model.names
        
        msg_queue.put('model loaded success')
        
        #warmup
        warm_up_img = np.zeros((640, 640, 3), dtype=np.uint8)  # Creating a blank image with dimensions 416x416
        print('Warming up the model...')
        msg_queue.put('model warmed up')
        time.sleep(0.5)
        model(warm_up_img)  # Performing a warm-up inference
        print(f'Model warmed up. ok')
        msg_queue.put('model warmed up ok')
        
        #1초 대기
        time.sleep(5)
        
        msg_queue.put('AI Model Ready')
        
        while shutdown_event.is_set() == False :
            try :
                success, frame = cap_data_queue.get(block=False)
            except queue.Empty:
                # 큐가 비어 있는 경우 처리
                continue
            # print(f'cap_data_queue success : {success}')
            if success:
                results = model(source=frame, conf=0.25, verbose=False)
                data_queue.put((results,class_names))  # 추론 결과를 큐에 넣기
        
    except Exception as e:
        print(e)
        print('model load failed')
        # quit()
    
    print('inference_task end')
        
        
    
def rendering_task():
    
    # camera init
    cap = cv.VideoCapture(0)
    print(f'width: {cap.get(cv.CAP_PROP_FRAME_WIDTH)} , height: {cap.get(cv.CAP_PROP_FRAME_HEIGHT)}')
    
    # 원하는 해상도를 설정합니다.
    #3840 x 2160  2mp
    desired_width = config['CAM_WIDTH']
    desired_height = config['CAM_HEIGHT']
    
    # 웹캠이 설정한 해상도를 지원하는지 확인합니다.
    actual_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    if actual_width == desired_width and actual_height == desired_height:
        print(f"The webcam supports the resolution {desired_width}x{desired_height}")
    else:
        print(f"The webcam does not support the resolution {desired_width}x{desired_height}")
        print(f"Available resolution is {actual_width}x{actual_height}")
    
    # pygame 초기화
    pygame.init()
    # 색상 설정
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (0, 255, 0)

    # 화면 설정
    screen_width, screen_height = int(config['SCREEN_WIDTH']), int(config['SCREEN_HEIGHT'])

    if config['WINDOWED'] == True:
        screen = pygame.display.set_mode((screen_width, screen_height))
    else:
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    
    font = pygame.font.Font(None, 36)
    msg_gont = pygame.font.Font(None, 24)
    _prev_msg_text = None    
    _prev_result = None
    break_flag = False
    while break_flag == False:
        
        # 공유 데이터 큐에서 데이터 가져오기
        ret, frame = cap.read()
        
        if ret == False:
            continue
        
        frame = cv.flip(frame, 1)
        
        ## 카메라전달 
        try:
            cap_data_queue.put((ret,frame), block=False)
        except queue.Full:
            # 큐가 가득 찬 경우 처리
            pass
        
        # 결과 받기 
        results = []
        try:
            results, class_names = data_queue.get(block=False)
            _prev_result = results
        except queue.Empty:
            # 큐가 비어 있는 경우 처리
            if _prev_result != None:
                results = _prev_result
            pass
        
        # print(f'results : {results}')
        
        frame_height, frame_width, _ = frame.shape
        
        # 카메라 이미지 출력
        # Convert the frame from BGR to RGB (as OpenCV uses BGR and Pygame uses RGB)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)

        # Convert the NumPy array to a Pygame surface
        frame_surface = pygame.surfarray.make_surface(frame_rgb)
        #flip
        frame_surface = pygame.transform.flip(frame_surface, True, False)

        # 카메라 이미지 출력
        if config['BKG_CAM'] == True:
            screen.blit(pygame.transform.scale(frame_surface, (screen_width, screen_height)), (0, 0))
        else:
            screen.fill(black)

        # 가운데 십자표시 그리기
        pygame.draw.line(screen, white, (screen_width // 2, 0), (screen_width // 2, screen_height), 1)
        pygame.draw.line(screen, white, (0, screen_height // 2), (screen_width, screen_height // 2), 1)
        
        # Draw the bounding boxes on the screen
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 카메라 해상도와 스크린해상도가 다를 경우 위치 보정
                scalex = screen_width / frame_width
                scaley = screen_height / frame_height
                
                x1, y1, x2, y2 = int(x1 * scalex), int(y1 * scaley), int(x2 * scalex), int(y2 * scaley)
                
                # 사각형 그리기
                # pygame.draw.rect(screen, green, (x1, y1, x2 - x1, y2 - y1), 2)
                
                # 클래스 이름과 신뢰도 출력
                class_id = int(box.cls.cpu().item())
                class_name = class_names[class_id]
                conf = int(box.conf.cpu().item() * 100)
                
                label_text = f"{class_name} {conf}%"
                
                text_surface = font.render(label_text, True, green)
                
                # 중심에 출력
                text_width, text_height = text_surface.get_size()
                center_x = (x1 + x2) / 2 - text_width / 2
                center_y = (y1 + y2) / 2 - text_height / 2
                screen.blit(text_surface, (center_x, center_y))

                
        try :
            _msg_text = msg_queue.get(block=False)
            
        except queue.Empty:
            if _prev_msg_text != None:
                _msg_text = _prev_msg_text
            pass
        
        msg_text_surface = msg_gont.render(_msg_text, True, green)
        screen.blit(msg_text_surface, (10, 10))
        
        pygame.display.update()

        # Pygame 이벤트 처리 (종료 등)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break_flag = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    break_flag = True
    cap.release()
    print('rendering_task end')

#%% 스레드 생성 및 시작
inference_thread = threading.Thread(target=inference_task)
inference_thread.start()
rendering_task()

time.sleep(1)
# 종료 이벤트 설정 (이 호출은 쓰레드에게 종료할 것임을 알립니다.)
shutdown_event.set()

# 스레드가 종료될 때까지 대기
inference_thread.join()

print('Done.')