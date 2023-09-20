import pygame
import sys

# Pygame 초기화
pygame.init()

# 화면 해상도 설정
screen_width = 1920
screen_height = 1080

# 전체 화면 모드로 창 생성
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)

# 창 제목 설정
pygame.display.set_caption('Full Screen with Crosshair')

# 색상 설정
black = (0, 0, 0)
white = (255, 255, 255)

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

    # 배경색으로 화면을 채움
    screen.fill(black)

    # 가운데 십자표시 그리기
    pygame.draw.line(screen, white, (screen_width // 2, 0), (screen_width // 2, screen_height), 1)
    pygame.draw.line(screen, white, (0, screen_height // 2), (screen_width, screen_height // 2), 1)

    # 화면 업데이트
    pygame.display.update()
