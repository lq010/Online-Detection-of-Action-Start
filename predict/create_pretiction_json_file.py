import h5py
import os
import numpy as np
import time
import json


"""###################################
class index dictionary
"""
class_index = dict()
index_file = open('data/Class Index_Detection.txt','r')
lines = index_file.readlines()
for line in lines:
    item = line.strip().split()
    class_index[int(item[2])] = item[1]
index_file.close()


output_dir = 'data/dataset/sgd_momentum'
output_path = os.path.join(output_dir, 'video_features.hdf5')
GT_file = 'data/test_ground_truth.json'

window_length = 16
threshold_score = 0.5
time_offset = 10


'''get the grount truth'''
with open(GT_file) as f:
    ground_truth = json.load(f)


result = dict()
def print_attrs(name, obj):
    predictions = np.array(obj)
    max_scores_indexes = np.argmax(predictions,axis=2)
    max_scores = predictions.max(axis =2)
    result[name] = []
    total_duration = ground_truth[name]['duration']
    total_frames = ground_truth[name]['totFrames']
    for i in range (1,len(max_scores_indexes)):
        Ct = int(max_scores_indexes[i]) # the out put of current window, C: class of an action
        Cp = int(max_scores_indexes[i-1]) #the out put of previous window
        if Ct > 0 and Cp != Ct and max_scores[i]>= threshold_score: # 0-> background, 1-20->actions
            t = (i+window_length-1)/total_frames * total_duration

            result[name].append({'label':class_index[Ct], 'time':t , 'score':float(max_scores[i])})

def get_result():
    with h5py.File(output_path, 'r+') as f:
        f.visititems(print_attrs)
    return result

if __name__ == '__main__':
    result = get_result()
    save_path = './data/atest.json'
    with open(save_path, 'w') as f:
        json.dump(result, f)
    print('file {} created.'.format(save_path))


''''''

    # with open(save_path) as f:
    #     r = json.load(f)

    # for videoid, v in r.items():
    #     for result in v:
            
    #         print(videoid)
    #         print(result['time'])
    #         print()
    #         print(result['score'])
    #         exit()