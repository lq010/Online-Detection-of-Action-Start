import cv2
import os


video_path = '/media/lq/C13E-1ED0/dataset/UCF_Crimes/Videos/RoadAccidents'
dest_path = '/media/lq/C13E-1ED0/dataset/UCF_Crimes/imgs/RoadAccidents'

video_list = os.listdir(video_path)

import time 
tic = time.time()

def get_num_frames_and_duration(video_path):
    ''' Return the number of frames and duration of the video track of the video given '''
    import cv2
    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        CAP_PROP_FPS = cv2.CAP_PROP_FPS
    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS

    cap = cv2.VideoCapture(video_path)
    cap.set(CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise Exception('Could not open the video ' + video_path )
    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(CAP_PROP_FPS))
    duration = num_frames / fps
    return num_frames, duration  

for counter,video in enumerate(video_list):   
    prefix = video.split('.')[0]
    if not os.path.exists(dest_path+'/'+prefix):
        os.makedirs(dest_path+'/'+prefix)
    save_name = dest_path + '/' + prefix + '/'
    video_name = video_path+'/'+video

    a = get_num_frames_and_duration(video_name)
    print( "{}-{}".format(a, a[0]/a[1]) )


    # cap = cv2.VideoCapture(video_name)
    # fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps_count = 0 
    # for i in range(fps):
    #     ret, frame = cap.read()
    #     if ret:
    #         img = "{}{:07n}{}".format(save_name,i,'.jpg')
    #         cv2.imwrite(img,frame)
    #         fps_count += 1
    # print("{:03n}: video <{}> done!".format(counter, video))
toc = time.time()
print(str(toc-tic)+'s')