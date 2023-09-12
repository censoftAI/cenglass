#%%
import cv2 as cv
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from IPython.display import display

print(f'opencv version {cv.__version__}')
#%%
cap = cv.VideoCapture(0)
print(f'width: {cap.get(cv.CAP_PROP_FRAME_WIDTH)} , height: {cap.get(cv.CAP_PROP_FRAME_HEIGHT)}')


#%%
# 원하는 해상도를 설정합니다.
#3840 x 2160  2mp
desired_width = 3840
desired_height = 2160
cap.set(cv.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, desired_height)

# 웹캠이 설정한 해상도를 지원하는지 확인합니다.
actual_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
if actual_width == desired_width and actual_height == desired_height:
    print(f"The webcam supports the resolution {desired_width}x{desired_height}")
else:
    print(f"The webcam does not support the resolution {desired_width}x{desired_height}")
    
    
# %%
ret,frame = cap.read()

display(Image.fromarray( cv.cvtColor(frame,cv.COLOR_BGR2RGB) ))
# %%
