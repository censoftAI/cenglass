#%%
from ultralytics import YOLO,checks
import cv2 as cv

import os
import yaml
import time
import pygame
import numpy as np
# from dotenv import load_dotenv
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


#%% 쓰레드 관련 객체 선언
# 공유 데이터 큐 생성
data_queue = queue.Queue()
cap_data_queue = queue.Queue(maxsize=1)
msg_queue = queue.Queue()

#종료 이벤트 객체 생성
shutdown_event = threading.Event()

# AI 추론 태스크 
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
        msg_queue.put('Warming up the model... please wait')
        time.sleep(0.5)
        model(warm_up_img)  # Performing a warm-up inference
        print(f'Model warmed up. ok')
        msg_queue.put('model warmed up ok')
        
        #1초 대기
        time.sleep(5)
        
        msg_queue.put('AI Model Ready')
        
        conf_threshold = config['CONF_THRESHOLD']
        
        while shutdown_event.is_set() == False :
            try :
                success, frame = cap_data_queue.get(block=False)
            except queue.Empty:
                # 큐가 비어 있는 경우 처리
                continue
            # print(f'cap_data_queue success : {success}')
            if success:
                results = model(source=frame, conf=conf_threshold, verbose=False)
                data_queue.put((results,class_names))  # 추론 결과를 큐에 넣기
        
    except Exception as e:
        print(e)
        print('model load failed')
        # quit()
    
    print('inference_task end')
    
    
class InformationWindow:
    def __init__(self,img,screenx,screeny):
        self.img = img
        self.screenx = screenx
        self.screeny = screeny
        self.closeBtn = pygame.image.load("./res/close_redX.png")
        # 닫기버튼 그리기 좌상단에 64x64 크기로 줄여서 그려준다.
        self.closeBtn = pygame.transform.scale(self.closeBtn, (64, 64))
    
    def draw(self, screen):
        if self.img != None:
            # 화면 중앙에 그리기 
            img_width, img_height = self.img.get_size()
            screen.blit(self.img, (self.screenx/2 - img_width/2, 0))
            
            screen.blit(self.closeBtn, (self.screenx - 64, 0))
            
            
            # screen.blit(self.img, (0, 0))
class FingerCursor:
    def __init__(self,screenx,screeny,color):
        # 초기값은 화면 중앙
        self.x = screenx/2
        self.y = screeny/2
        self.color = color
        pass
    def draw(self,screen):
        #빨간색 원 그리기,속이 빈원그리기 
        pygame.draw.circle(screen, self.color, (self.x, self.y), 32, 0)
        pass
        
def draw_crosshair(screen, color, size,x,y):
    pygame.draw.line(screen, color, (x - size, y), (x + size, y), 1)
    pygame.draw.line(screen, color, (x, y - size), (x, y + size), 1)
    
# 화면 랜더링 태스크
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
    
    # hide mouse
    pygame.mouse.set_visible(False)
    
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
    
    # font 설정
    # font = pygame.font.Font("path/to/your/font.ttf", 30) 
    title_font = pygame.font.Font('./font/NanumGothic.ttf', 36)
    msg_font = pygame.font.Font(None, 24)
    
    # 설명서 이미지 로드
    canon_img = pygame.image.load("./res/canon.jpg")
    canon_img = pygame.transform.scale(canon_img, (screen_width/2, screen_height))
    heli_img = pygame.image.load("./res/heli.jpg")
    heli_img = pygame.transform.scale(heli_img, (screen_width/2, screen_height))
    wemosd1_img = pygame.image.load("./res/wemosd1.png")
    wemosd1_img = pygame.transform.scale(wemosd1_img, (screen_width/2, screen_height))
    
    _prev_msg_text = None    
    _prev_result = None
    break_flag = False
    
    #설명창
    infomation_Window = InformationWindow(None,screen_width,screen_height)
    
    #손가락 커서
    finger_cursor = FingerCursor(screen_width,screen_height,(255,0,0))
    
    while break_flag == False:
        
        last_time = time.time()  # Initialize last_time before entering the loop
        
        # 공유 데이터 큐에서 데이터 가져오기
        ret, frame = cap.read()
        
        if ret == False:
            continue
        
        # frame = cv.flip(frame, 1)
        
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
        # pygame.draw.line(screen, white, (screen_width // 2, 0), (screen_width // 2, screen_height), 1)
        # pygame.draw.line(screen, white, (0, screen_height // 2), (screen_width, screen_height // 2), 1)
        draw_crosshair(screen, white, 64, screen_width // 2, screen_height // 2)
        
        drawn_boxes = []  # 이 리스트에는 그려진 박스의 좌표를 저장합니다.
        # 카메라 해상도와 스크린해상도가 다를 경우 위치 보정
        scalex = screen_width / frame_width
        scaley = screen_height / frame_height
        
        # 화면의 중점 좌표
        center_x, center_y = screen_width // 2, screen_height // 2
        # drawn_boxes 초기화
        drawn_boxes = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class and confidence
                class_id = int(box.cls.cpu().item())
                class_name = class_names[class_id]
                
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1, y1, x2, y2 = int(x1 * scalex), int(y1 * scaley), int(x2 * scalex), int(y2 * scaley)
                
                if class_name == 'finger':
                    # position at center box
                    finger_cursor.x = (x1 + x2) // 2
                    finger_cursor.y = (y1 + y2) // 2
                    continue
                
                # 겹치는 박스가 있는지 확인
                overlap = False
                for drawn_box in drawn_boxes:
                    dx1, dy1, dx2, dy2 = drawn_box
                    if x1 < dx2 and x2 > dx1 and y1 < dy2 and y2 > dy1:
                        overlap = True
                        break
                
                # 겹치는 박스가 없으면 그립니다.
                if not overlap:
                    pygame.draw.rect(screen, green, (x1, y1, x2 - x1, y2 - y1), 2)
                    drawn_boxes.append((x1, y1, x2, y2))
                    print(class_name)
                    
                    #손가락 커서와 겹치는지 확인
                    if x1 < finger_cursor.x < x2 and y1 < finger_cursor.y < y2:
                        # 이미지 출력
                        if class_name == 'canon':
                            infomation_Window.img = canon_img
                            # screen.blit(canon_img, (center_x, 0))
                        elif class_name == 'heli':
                            # screen.blit(heli_img, (center_x, 0))
                            infomation_Window.img = heli_img
                        elif class_name == 'wemosd1':
                            # screen.blit(wemosd1_img, (center_x, 0))
                            infomation_Window.img = wemosd1_img
                        break
            
                
                    
                        

                    # 화면의 중점이 박스 안에 있는지 확인
                    # if x1 < center_x < x2 and y1 < center_y < y2:
                    #     # 이미지 출력
                    #     # img_width, img_height = canon_img.get_size()
                    #     if class_name == 'canon':
                    #         infomation_Window.img = canon_img
                    #         # screen.blit(canon_img, (center_x, 0))
                    #     elif class_name == 'heli':
                    #         # screen.blit(heli_img, (center_x, 0))
                    #         infomation_Window.img = heli_img
                    #     elif class_name == 'wemosd1':
                    #         # screen.blit(wemosd1_img, (center_x, 0))
                    #         infomation_Window.img = wemosd1_img
                    #     elif class_name == 'finger':
                    #         infomation_Window.img = None
                    #     break
                    
        # 이미지창닫기 closeBtn 과 충돌검사
        if infomation_Window.closeBtn.get_rect(topleft=(screen_width - 64, 0)).collidepoint(finger_cursor.x, finger_cursor.y):
            infomation_Window.img = None
            
        # 정보창그리기             
        infomation_Window.draw(screen)
        
        # 손가락 커서 그리기
        finger_cursor.draw(screen)
                
        try :
            _msg_text = msg_queue.get(block=False)
        except queue.Empty:
            if _prev_msg_text != None:
                _msg_text = _prev_msg_text
            pass
        
        msg_text_surface = msg_font.render(_msg_text, True, green)
        screen.blit(msg_text_surface, (10, 0))
        
        # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - last_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        last_time = current_time
        
        # Render FPS
        fps_text = f"FPS: {fps:.2f}"
        fps_surface = msg_font.render(fps_text, True, green)
        screen.blit(fps_surface, (10, 25))  # Adjust the position (10, 10) as needed
        
        # 타이틀이 존재하면 상단 중앙에 타이틀 출력
        if config['TITLE'] != None:
            title_text = config['TITLE']
            title_surface = title_font.render(title_text, True, green)
            title_width, title_height = title_surface.get_size()
            screen.blit(title_surface, (screen_width / 2 - title_width / 2, 0))
        
        pygame.display.update()

        # Pygame 이벤트 처리 (종료 등)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break_flag = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    break_flag = True
                elif event.key == pygame.K_1:
                    infomation_Window.img = wemosd1_img
                    
            
    cap.release()
    print('rendering_task end')

#%% 스레드 생성 및 시작
inference_thread = threading.Thread(target=inference_task)
inference_thread.start()

#%% 메인 루프
print('start rendering_task')
rendering_task()

time.sleep(1)
# 종료 이벤트 설정 (이 호출은 쓰레드에게 종료할 것임을 알립니다.)
shutdown_event.set()
# 스레드가 종료될 때까지 대기
inference_thread.join()
print('Done.')
