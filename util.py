import keras.backend as K
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# def load_dataset():

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
    if not cap.isOpened():
        raise Exception('Could not open the video ' + video_path )
    num_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(CAP_PROP_FPS))
    duration = num_frames / fps
    return num_frames, duration  

'''
if __name__ == '__main__':
    path = "/media/lq/C13E-1ED0/dataset/THUMOS/validation/"
    #print(get_num_frames(path))
    videos = os.listdir(path)
    videosInfo = dict()
    for video in videos:
        video_path  = path +video
        
        videosInfo[video] = get_num_frames_and_duration(video_path)
    print(sys.getsizeof(videosInfo))
'''
path = "/media/lq/C13E-1ED0/dataset/THUMOS/validation/video_validation_0000856.mp4"
print(get_num_frames_and_duration(path))