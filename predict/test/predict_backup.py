from __future__ import absolute_import

import argparse
import multiprocessing
import os
import sys
import time
import traceback
import cv2

from models import c3d_model

import numpy as np
from progressbar import ProgressBar

import h5py
from src.data import VideoGenerator

from src.LR_Adam import Adam
from src.LR_SGD import SGD

def predict_videos(videos_dir, output_dir, batch_size, num_threads,
                     queue_size, num_gpus, weights_path):
    # Defining variables
    resize_size=(128,171)
    input_size = (112, 112)
    length = 16
    wait_time = 0.1

    output_path = os.path.join(output_dir, 'video_features.hdf5')
    mode = 'r+' if os.path.exists(output_path) else 'w'
    # Extract the ids of the videos already predicted 
    with h5py.File(output_path, mode) as output_file:
        predicted_videos = output_file.keys()
        videos_ids = [v[:-4] for v in os.listdir(videos_dir) if v[-4:] == '.mp4']
        # Lets remove from the list videos_ids, the ones already predicted
        videos_ids_to_predict = list(set(videos_ids) - set(predicted_videos))
        
        nb_videos = len(videos_ids_to_predict)
        print('Total number of videos: {}'.format(len(videos_ids)))
        print('Videos already predicted: {}'.format(len(predicted_videos)))
        print('Videos to predict: {}'.format(nb_videos))

    # Creating Parallel Fetching Video Data
    print('Creating {} process to fetch video data'.format(num_threads))
    data_gen_queue = multiprocessing.Queue(maxsize=queue_size)
    _stop_all_generators = multiprocessing.Event()
    _stop_all_extractors = multiprocessing.Event()

    def data_generator_task(index):
        generator = VideoGenerator(
            videos_ids_to_predict[index:nb_videos:num_threads], videos_dir,
            'mp4', length,resize_size)
        keep = True
        while keep:
            try:
                if data_gen_queue.qsize() < queue_size:
                    try:
                        generator_output = next(generator)
                    except ValueError:
                        continue
                    data_gen_queue.put(generator_output)
                else:
                    time.sleep(wait_time)
            except StopIteration:
                print('End')
                break
            except Exception:
                keep = False
                print('Something went wrong with generator_process. generator: {}'.format(index))
                print(traceback.print_exc())
        print('Generator <{}> stopped.'.format(index))
    generator_process = [
        multiprocessing.Process(target=data_generator_task, args=[i])
        for i in range(num_threads)
    ]
    for process in generator_process:
        process.daemon = True
        process.start()

    data_save_queue = multiprocessing.Queue()

    def predict_task():
        print("predict task running")
        # Loading the model
        print('Loading model')
        model = c3d_model.get_model()
        print('Compiling model')
            # Setting the Learning rate multipliers
        LR_mult_dict = {}
        LR_mult_dict['conv1']=1
        LR_mult_dict['conv2']=1
        LR_mult_dict['conv3a']=1
        LR_mult_dict['conv3b']=1
        LR_mult_dict['conv4a']=1
        LR_mult_dict['conv4b']=1
        LR_mult_dict['conv5a']=1
        LR_mult_dict['conv5b']=1
        LR_mult_dict['fc6']=1
        LR_mult_dict['fc7']=1
        LR_mult_dict['fc8']=10
        # Setting up optimizer
        base_lr = 0.00001
        adam = Adam(lr=base_lr, decay=0.00005, multipliers=LR_mult_dict)
        sgd = SGD(lr=base_lr, decay=0.00005, multipliers=LR_mult_dict)
        opt = sgd 
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
       
        print('Compiling done!')
        print(weights_path)
        model.load_weights(weights_path)
        print("wight <{}> loaded.".format(weights_path))
        print('Starting extracting features')


        while not (_stop_all_generators.is_set() and data_gen_queue.empty()):
            generator_output = None
            while True:
                if not data_gen_queue.empty():
                    generator_output = data_gen_queue.get()
                    if not generator_output:
                        continue
                    break
                else:
                    time.sleep(wait_time)
            video_id, X = generator_output
            print("predicting video :{}".format(video_id))
            if X is None:
                print('Could not be read the video {}'.format(video_id))
                continue
            # X = X - mean
            prediction = []
            for i in range(len(X)-length+1):
                inputs = X[i:i+length]
                inputs = inputs.astype(float) #TODO, float64  ??????
                inputs /=255.
                inputs = np.expand_dims(inputs, axis=0)
                Y = model.predict_on_batch(inputs)
                prediction.append(Y)

            data_save_queue.put((video_id, prediction))
            print("video :{} -done!".format(video_id))
        print('prediction task stopped')
    extractors_process = [
        multiprocessing.Process(target=predict_task)
        for i in range(num_gpus)
    ]
    for p in extractors_process:
        p.daemon = True
        p.start()

    # Create the process that will get all the extracted features from the data_save_queue and
    # store it on the hdf5 file.

    def saver_task():
        print("saver task runing")
        while not (_stop_all_extractors.is_set() and data_save_queue.empty()):
            extracted_output = None
            while True:
                if not data_save_queue.empty():
                    extracted_output = data_save_queue.get()
                    if not extracted_output:
                        continue
                    break
                else:
                    time.sleep(wait_time)
            video_id, predictions = extracted_output
            if predictions is None:
                print('Something went wrong')
                continue
           
            with h5py.File(output_path, 'r+') as f:
                f.create_dataset(video_id, data=predictions, dtype='float32')
            print('Saved video {}'.format(video_id))
        print("saver task stopped.")
    saver_process = multiprocessing.Process(target=saver_task)
    saver_process.daemon = True
    saver_process.start()

    # Joining processes
    for p in generator_process:
        p.join()
    _stop_all_generators.set()
    for p in extractors_process:
        p.join()
    _stop_all_extractors.set()
    saver_process.join()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='predict the output with the trained model')
    parser.add_argument(
        '-d',
        '--videos-dir',
        type=str,
        dest='directory',
        default='/media/lq/C13E-1ED0/dataset/THUMOS/tmptest',
        help='videos directory (default: %(default)s)')
    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        dest='output',
        default='data/dataset',
        help=
        'directory where to store the predictions (default: %(default)s)'
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        dest='batch_size',
        default=24,
        help='batch size when predicting (default: %(default)s)')
    parser.add_argument(
        '-t',
        '--num-threads',
        type=int,
        dest='num_threads',
        default=1,
        help='number of threads to fetch videos (default: %(default)s)')
    parser.add_argument(
        '-q',
        '--queue-size',
        type=int,
        dest='queue_size',
        default=2,
        help=
        'maximum number of elements at the queue when fetching videos (default %(default)s)'
    )
    parser.add_argument(
        '-g',
        '--num-gpus',
        type=int,
        dest='num_gpus',
        default=1,
        help=
        'number of gpus to use for predicting (default: %(default)s)')

    parser.add_argument(
        '-w',
        '--weight-path',
        type=str,
        dest = 'weights_path',
        help = 
        'the path of weight to be loaded.')
    args = parser.parse_args()

    predict_videos(args.directory, args.output, args.batch_size,
                     args.num_threads, args.queue_size, args.num_gpus,args.weights_path)
