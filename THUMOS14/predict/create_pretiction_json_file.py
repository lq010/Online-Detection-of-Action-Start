import h5py
import os
import numpy as np
import time
import json
import argparse

window_length = 16
time_offset = 10

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

'''#########
get the grount truth
'''
GT_file = 'data/test_ground_truth.json'
# GT_file = 'data/validation_ground_truth.json'
with open(GT_file) as f:
    ground_truth = json.load(f)


def get_result(rawPredictions, threshold_score):
    result = dict()
    def print_attrs(name, obj):
        predictions = np.array(obj)
        arg_axis = len(predictions.shape)-1
        max_scores_indexes = np.argmax(predictions,axis= arg_axis)
        max_scores = predictions.max(axis = arg_axis)
        result[name] = []
        try:
            total_duration = ground_truth[name]['duration']
            total_frames = ground_truth[name]['totFrames']
        except KeyError as e:
            print('video <{}> only exist in ambiguous list, skiped.'.format(name))
            return
        for i in range (1,len(max_scores_indexes)):
            Ct = int(max_scores_indexes[i]) # the out put of current window, C: class of an action
            Cp = int(max_scores_indexes[i-1]) #the out put of previous window
            if Ct > 0 and Cp != Ct and max_scores[i]>= threshold_score: # 0-> background, 1-20->actions
                t = (i+window_length-1)/total_frames * total_duration
                result[name].append({'label':class_index[Ct], 'time':t , 'score':float(max_scores[i])})

    with h5py.File(rawPredictions, 'r') as f:
        f.visititems(print_attrs)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='create the final result')
    parser.add_argument(
        '-p',
        '--raw-predictions]',
        type=str,
        dest='rawPredictions',
        help='the raw prediction file')
    parser.add_argument(
        '-d',
        '--prediction-dir',
        type=str,
        dest='dir',
        default='data/dataset',
        help=
        'directory where to store the final result (default: %(default)s)'
    )
    parser.add_argument(
        '-s',
        '--threshold-score',
        type=float,
        dest='threshold_score',
        default=0.5,
        help=
        'directory where to store the final result (default: %(default)s)'
    )
    args = parser.parse_args()
    rawPredictions =os.path.join(args.dir, args.rawPredictions)
    threshold_score = args.threshold_score
    save_name = args.rawPredictions.replace('.hdf5', '_score_' +str(threshold_score)+'.json')
    save_path = os.path.join(args.dir, save_name)

    result = get_result(rawPredictions, args.threshold_score)

    with open(save_path, 'w') as f:
        json.dump(result, f)
    print('file {} created.'.format(save_path))




    # with h5py.File(output_path, 'r+') as f:
    #     del f['video_test_0001235']

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