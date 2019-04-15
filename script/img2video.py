import cv2
import numpy as np
import os
 
img_array = []

dir = '/media/lq/C13E-1ED0/dataset/cleanAndJerk_example'
img_list = sorted(os.listdir(dir))

for filename in img_list:
    img = cv2.imread(os.path.join(dir,filename))
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

 
out = cv2.VideoWriter('cleanAndJerk.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 20, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()