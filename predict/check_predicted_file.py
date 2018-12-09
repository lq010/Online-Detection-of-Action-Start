import h5py
import os
import numpy as np
import time
import json

output_dir = 'data/dataset'
output_path = os.path.join(output_dir, 'video_features.hdf5')
GT_file = 'data/test_ground_truth.json'

window_length = 16
threshold_score = 0
time_offset = 10
# def default():
#     with h5py.File(output_path, 'r+') as f:
#         print(f)
#         print("Keys: %s" % list(f.keys()))
#         a_group_key = list(f.keys())[0]

#         # Get the data
#         data = list(f)
#         data = np.array(data)
#         print(data.shape)

#         for a in f:
#             print(f[a])

frames = {}
from src.io_data import get_num_frames
dir ='/media/lq/C13E-1ED0/dataset/THUMOS/tmptest'

def number_of_frames():
    videos = os.listdir(dir)
    videos.sort()
    for v in videos:
        n = get_num_frames(dir+"/"+v)
        # print("{}:{}".format(v, n-16+1))
        frames[v] = n-16+1
    return frames

'''check the number of predictions, it should be == the number of frames'''
def check_prediction_number(name,obj):
    predictions = np.array(obj)  
    num = len(predictions)
    name = name+".mp4"
    if num != frames[name]:
        print ("Shape error: v-{}, t{}, p{}".format(name, frames[name],num))

def check_prediction_shape():
     with h5py.File(output_path, 'r+') as f:       
        f.visititems(check_prediction_number)



'''get the grount truth'''
with open(GT_file) as f:
    ground_truth = json.load(f)

'''
process the prediction. 
return:
    a dictionary:
        |-video name
          |-action class, time, score
          |...
        |... 
'''
result = dict()
def print_attrs(name, obj):
    predictions = np.array(obj)  
    Action_indexes = np.argmax(predictions,axis=2) 
    scores = predictions.max(axis =2)
    result[name] = []
    total_duration = ground_truth[name]['duration']
    total_frames = ground_truth[name]['totFrames']
    for i in range (1,len(Action_indexes)):
        Ct = int(Action_indexes[i]) # the out put of current window, C: class of an action
        Cp = int(Action_indexes[i-1]) #the out put of previous window
        if Ct > 0 and Cp != Ct and scores[i]>= threshold_score:
            t = (i+window_length-1)/total_frames * total_duration
            result[name].append([Ct, t, float(scores[i])])

def get_result():
    with h5py.File(output_path, 'r+') as f:       
        f.visititems(print_attrs)
    return result

# '''evaluate '''

# def evaluate(result):
#     correct_counter = 0
#     for vid,results in result.items():
#         print(vid)
#         gt_instances = ground_truth[vid]['timeStamp']       
#         for detected_AS in results:
#             for gt_AS in gt_instances:
#                 '''detected_AS:[class_label, start_time, score], gt_AS:[start_time, end_time, class_label]'''
#                 if detected_AS[0]==gt_AS[2] and abs(gt_AS[0]-detected_AS[1])<=time_offset:
#                     correct_counter += 1
#                     print(correct_counter)
#                     break # no duplicated detection
            

if __name__ == '__main__':
   
    r = get_result()
    print("number of AS:")
    for i in r:
        print("\t{}:{}".format(i,len(r[i])))

    f = number_of_frames()
    print("total number of vidoes:{}".format(len(f)))
    # for i in f :
    #     print( "{}: {}".format(i,f[i]))

    check_prediction_shape()
    # evaluate(r)
