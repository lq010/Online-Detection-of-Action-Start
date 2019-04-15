import os
import random
import json
import sys
import cv2

#path of the original val videos
test_videos_path = '/media/lq/C13E-1ED0/dataset/UCF_Crimes/Videos/RoadAccidents'
#path of the val video annotations
train_annotations_path = '/home/lq/Documents/Thesis/Thesis/UCF/data/ucf_annotations_train'

FPS = 30
accidentLabel = 1

files = os.listdir(train_annotations_path)


train_data = dict()




"""###################################
get the number of frames and duration 
an example of videoInfo dict:
-  video name                    #frames  duration
-{'video_validation_0000990': (3674, 122.46666666666667),...}
"""

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

videoInfo = dict()
videos = os.listdir(test_videos_path)
for video in videos:
    videoPath = os.path.join(test_videos_path, video)
    video = video.split('.')[0]
    videoInfo[video] = get_num_frames_and_duration(videoPath)
# print(str(videoInfo))

""""""
for fileName in files:

    filepath = os.path.join(train_annotations_path, fileName)

    file = open(filepath, 'r')
    
    lines = file.readlines()
    #get the start and end time 
    for line in lines:
        items = line.strip().split()
        assert len(items) == 6

        vid = items[0].replace('.mp4','')  # video name
        #first action instance
        try:
            action = items[1]
            startFrame1 = int(items[2])
            endFrame1 = int(items[3])
            startFrame2 = int(items[4])
            endFrame2 = int(items[5])
            totFrames, duration = videoInfo[vid]
        except ValueError:
            print ("error in file :" +fileName +", on line: ",line)

        if vid not in train_data.keys():
            train_data[vid] = {}
            train_data[vid]['duration'] = duration
            train_data[vid]['totFrames'] = totFrames
            train_data[vid]['frameStamp'] = []
        train_data[vid]['frameStamp'].append([startFrame1, endFrame1, accidentLabel])

        if startFrame2!=-1 and endFrame2!=-1:

            train_data[vid]['frameStamp'].append([startFrame2, endFrame2, accidentLabel])
    file.close()


print('Writing train json data ...')
with open('train_accidents.json', 'w') as f:
    json.dump(train_data, f)
print(len(train_data))