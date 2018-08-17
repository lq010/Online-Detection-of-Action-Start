"""
perform an 80%-20% split over the training examples, 
ensureing that the distribution of activity classes is approximately the same
"""
import os
import random
import json
import sys
sys.path.insert(0,'../')
import constantPaths as path

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
class_index = dict()

"""###################################
class index dictionary
"""
index_file = open('Class Index_Detection.txt','r')
lines = index_file.readlines()
for line in lines:
    item = line.strip().split()
    class_index[item[1]] = item[0]
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
for videoName,info in videoInfo.items():
    totFrames = info[0]
    duration = info[1]
    trian_data[videoName] = {}
    trian_data[videoName]['duration'] = duration
    trian_data[videoName]['totFrames'] = totFrames
    trian_data[videoName]['frameStamp'] = []
    for i in range(totFrames):
        trian_data[videoName]['frameStamp'].append([i,0,0,0]) # [i,0,0,0] => [#frame, isActionStart,isActionEnd, actionClass]
print(type(trian_data[videoName]['frameStamp']))
print(len(videoInfo))
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
        endTime = float(items[2]) #0123456789 

        totFrames, duration = videoInfo[vid] 
        startFrame = int(totFrames*(startTime/duration))
        endFrame = int(totFrames*(endTime/duration))
        d =endFrame - startFrame
        if d<=16:
            print(vid +' '+str(endTime - startTime))
#         if vid not in video_list:
#             video_list.append(vid)
#         # if vid not in trian_data.keys():
#         #     trian_data[vid] = {}
#         #     trian_data[vid]['duration'] = duration
#         #     trian_data[vid]['totFrames'] = totFrames
#         #     trian_data[vid]['frameStamp'] = []
#         trian_data[vid]['frameStamp'][startFrame] = [startFrame,1,0,actionLabel]
#         trian_data[vid]['frameStamp'][endFrame] = [endFrame,0,1,actionLabel]
#         for i in range(startFrame+1,endFrame):
#             trian_data[vid]['frameStamp'][i] = [i,0,0,actionLabel]

#     file.close()
    
#     #shuffle the video list, get the list of validation video set 
#     random.shuffle(video_list)
#     N = len(video_list)
#     val_N = int(N*val_split['val'])
#     trian_N = N - val_N
#     val_video_list_set |= set(video_list[:val_N])

# #extract the val data from train data
# for videoName in val_video_list_set:
#     val_data[videoName] = trian_data.pop(videoName)

# print('Writing train json data ...')
# with open('my_train.json', 'w') as f:
#     json.dump(trian_data, f)

# print('Writing val json data ...')
# with open('my_validation.json', 'w') as f:
#     json.dump(val_data, f)
    
# # print(str(len(val_video_list_set)))
# # print(str(len(val_data)))
# # print(str(len(trian_data)))
# #     #print(fileName.ljust(30) + str(N).ljust(4)+ str(val_N).ljust(4)+ str(trian_N))
