"""
perform an 80%-20% split over the training examples, 
ensureing that the distribution of activity classes is approximately the same
"""
import os
import random
import json
import sys
import videoPaths as path

#path of the original val videos
val_videos_path = path.VALIDATION_VIDEOS_PATH
#path of the val video annotations
val_annotations_path = 'TH14_Temporal_annotations_validation/annotation'

random.seed(101)

files = os.listdir(val_annotations_path)
files.remove('Ambiguous_val.txt')

val_split = {'train': 0.8, 'val': 0.2}
trian_data = dict()
val_data = dict()

"""###################################
class index dictionary
"""
class_index = dict()
index_file = open('Class Index_Detection.txt','r')
lines = index_file.readlines()
for line in lines:
    item = line.strip().split()
    class_index[item[1]] = item[2]
index_file.close()

"""###################################
get the number of frames and duration 
an example of videoInfo dict:
-  video name                    #frames  duration
-{'video_validation_0000990': (3674, 122.46666666666667),...}
"""
import util
videoInfo = dict()
videos = os.listdir(val_videos_path)
for video in videos:
    videoPath = val_videos_path + video
    video = video.split('.')[0]
    videoInfo[video] = util.get_num_frames_and_duration(videoPath)
#print(str(videoInfo))


"""###################################
train and val data
"""
random.shuffle(files)
val_video_list_set = set()
for fileName in files:
    filepath = os.path.join(val_annotations_path, fileName)
    #print('Reading file: %s from %s'%(file, val_annotations_path))
    file = open(filepath, 'r')
    action = fileName.split('_')[0]
    actionLabel = int(class_index[action])
    lines = file.readlines() # action instances of one class (each line :videoName startTime endTime)
    
    video_list = list()
    #get the start and end frame 
    for line in lines:
        items = line.strip().split()
        assert len(items) == 3
        vid = items[0]  # video name
        startTime = float(items[1])
        endTime = float(items[2])
        totFrames, duration = videoInfo[vid] 
        startFrame = int(totFrames*(startTime/duration))
        endFrame = int(totFrames*(endTime/duration))
        if vid not in video_list:
            video_list.append(vid)
        if vid not in trian_data.keys():
            trian_data[vid] = {}
            trian_data[vid]['duration'] = duration
            trian_data[vid]['totFrames'] = totFrames
            trian_data[vid]['frameStamp'] = []
        trian_data[vid]['frameStamp'].append([startFrame, endFrame, actionLabel])
    file.close()
    
    #shuffle the video list, get the list of validation video set 
    random.shuffle(video_list)
    N = len(video_list)
    val_N = int(N*val_split['val'])
    trian_N = N - val_N
    val_video_list_set |= set(video_list[:val_N])

#extract the val data from train data
for videoName in val_video_list_set:
    val_data[videoName] = trian_data.pop(videoName)

print('Writing train json data ...')
with open('train.json', 'w') as f:
    json.dump(trian_data, f)

print('Writing val json data ...')
with open('validation.json', 'w') as f:
    json.dump(val_data, f)
    
print(str(len(val_video_list_set)))
print(str(len(val_data)))
print(str(len(trian_data)))
    #print(fileName.ljust(30) + str(N).ljust(4)+ str(val_N).ljust(4)+ str(trian_N))
