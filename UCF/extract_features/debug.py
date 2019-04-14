import h5py
import os
import numpy as np
import time
import json
import argparse
import cv2

window_length = 16
time_offset = 10

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

def get_result(rawPredictions, threshold_score):
    result = dict()
    def print_attrs(name, obj):
        predictions = np.array(obj)
        print(predictions.shape)
        arg_axis = len(predictions.shape)-1
        max_scores_indexes = np.argmax(predictions,axis= arg_axis)
        max_scores = predictions.max(axis = arg_axis)
        result[name] = []

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
        default='features.h5',
        help='the raw prediction file')
    parser.add_argument(
        '-d',
        '--prediction-dir',
        type=str,
        dest='dir',
        default='/media/lq/C13E-1ED0/dataset/UCF_Crimes/features',
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
    n =get_num_frames_and_duration('/media/lq/C13E-1ED0/dataset/UCF_Crimes/aaa/Abuse028_x264.mp4')
    print(n)
    result = get_result(rawPredictions, args.threshold_score)

    print(result)




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