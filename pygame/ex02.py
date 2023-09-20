import pygame
import cv2
import sys
import numpy as np

# Pygame 초기화
pygame.init()

# 화면 해상도 설정
screen_width = 1920
screen_height = 1080

# 전체 화면 모드로 창 생성
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)

# 창 제목 설정
pygame.display.set_caption('Full Screen with OpenCV Frame')

# 색상 설정
black = (0, 0, 0)
white = (255, 255, 255)

# OpenCV 카메라 초기화
cap = cv2.VideoCapture(0)

# 메인 루프
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

    # OpenCV로 프레임 캡처
    ret, frame = cap.read()
    if ret:
        # BGR에서 RGB로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # NumPy 배열을 Pygame 이미지로 변환
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        
        # 화면에 프레임 표시
        screen.blit(pygame.transform.scale(frame, (screen_width, screen_height)), (0, 0))

    # 화면 업데이트
    pygame.display.update()
