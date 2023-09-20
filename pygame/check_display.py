#%%
import pygame
pygame.init()

# 지원하는 모든 해상도를 가져옴
available_resolutions = pygame.display.list_modes()
# print(f"Available resx/olutions: {available_resolutions}")

for i in available_resolutions:
    print(i[0], i[1])
# %%
