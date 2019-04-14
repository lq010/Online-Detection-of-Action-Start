import os
import random
import json
import sys
import cv2

#path of the original val videos
test_videos_path = '/media/lq/C13E-1ED0/dataset/UCF_Crimes/Videos/RoadAccidents_test'
#path of the val video annotations
test_annotations_path = '/home/lq/Documents/Thesis/Thesis/UCF/data/ucf_annotations_test'

FPS = 30

files = os.listdir(test_annotations_path)


test_data = dict()




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

    filepath = os.path.join(test_annotations_path, fileName)

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
            startFrame1 = float(items[2])
            endFrame1 = float(items[3])
            startFrame2 = float(items[4])
            endFrame2 = float(items[5])
            totFrames, duration = videoInfo[vid]
        except ValueError:
            print ("error in file :" +fileName +", on line: ",line)

        if vid not in test_data.keys():
            test_data[vid] = {}
            test_data[vid]['duration'] = duration
            test_data[vid]['totFrames'] = totFrames
            test_data[vid]['annotations'] = []
        startTime1, endTime1 = startFrame1/FPS, endFrame1/FPS
        this_gt = [startTime1, endTime1]
        this_label = action        
        this_anno = {'timeStamp':this_gt, 'label': this_label}
        test_data[vid]['annotations'].append(this_anno)

        if startFrame2!=-1 and endFrame2!=-1:
            startTime2, endTime2 = startFrame2/FPS, endFrame2/FPS
            this_gt2 = [startTime2, endTime2]
            this_label2 = action
            thisanno2 = {'timeStamp':this_gt2, 'label': this_label2}
            test_data[vid]['annotations'].append(thisanno2)
    file.close()


print('Writing train json data ...')
with open('ucf_accident_test_ground_truth.json', 'w') as f:
    json.dump(test_data, f)