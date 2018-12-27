import os
import random
import json
import sys
import videoPaths as path

#path of the original val videos
test_videos_path = path.TEST_VIDEOS_PATH
#path of the val video annotations
test_annotations_path = 'TH14_Temporal_Annotations_Test/annotation'

files = os.listdir(test_annotations_path)
files.remove('Ambiguous_test.txt')

test_data = dict()




"""###################################
get the number of frames and duration 
an example of videoInfo dict:
-  video name                    #frames  duration
-{'video_validation_0000990': (3674, 122.46666666666667),...}
"""
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import util
videoInfo = dict()
videos = os.listdir(test_videos_path)
for video in videos:
    videoPath = test_videos_path + video
    video = video.split('.')[0]
    videoInfo[video] = util.get_num_frames_and_duration(videoPath)
#print(str(videoInfo))

""""""
for fileName in files:
    filepath = os.path.join(test_annotations_path, fileName)

    file = open(filepath, 'r')
    action = fileName.split('_')[0]
    
    lines = file.readlines()
    #get the start and end time 
    for line in lines:
        items = line.strip().split()
        assert len(items) == 3
        vid = items[0]  # video name
        try:
            startTime = float(items[1])
            endTime = float(items[2])
            totFrames, duration = videoInfo[vid]
        except ValueError:
            print ("error in file :" +fileName +", on line: ",line)

        if vid not in test_data.keys():
            test_data[vid] = {}
            test_data[vid]['duration'] = duration
            test_data[vid]['totFrames'] = totFrames
            test_data[vid]['annotations'] = []
        this_gt = [startTime, endTime]
        this_label = action
        this_anno = {'timeStamp':this_gt, 'label': this_label}
        test_data[vid]['annotations'].append(this_anno)
    file.close()

print('Writing train json data ...')
with open('test_ground_truth.json', 'w') as f:
    json.dump(test_data, f)