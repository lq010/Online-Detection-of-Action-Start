import cv2
import os
import videoPaths as path

video_path = path.VALIDATION_VIDEOS_PATH
dest_path = path.VALIDATION_IMAGES_PATH

video_list = os.listdir(video_path)

import time 
tic = time.time()

for counter,video in enumerate(video_list):
    
    prefix = video.split('.')[0]
    if not os.path.exists(dest_path+'/'+prefix):
        os.makedirs(dest_path+'/'+prefix)
    save_name = dest_path + '/' + prefix + '/'
    video_name = video_path+'/'+video
    cap = cv2.VideoCapture(video_name)
    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_count = 0 
    for i in range(fps):
        ret, frame = cap.read()
        if ret:
            img = "{}{:07n}{}".format(save_name,i,'.jpg')
            cv2.imwrite(img,frame)
            fps_count += 1
    print("{:03n}: video <{}> done!".format(counter, video))
toc = time.time()
print(str(toc-tic)+'s')