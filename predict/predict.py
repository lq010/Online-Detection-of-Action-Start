from __future__ import absolute_import

import argparse
import threading
import os
import sys
import time
import traceback
import cv2

from models import c3d_model

import numpy as np

import h5py
from src.data import VideoGenerator

from LR_Adam import Adam
from LR_SGD import SGD
import queue

def predict_videos(videos_dir, output_file, weights_file, 
                        batch_size, num_threads, queue_size, num_gpus):
    
    # Defining variables
    resize_size=(128,171)
    input_size = (112, 112)
    length = 16
    wait_time = 0.1

    mode = 'r+' if os.path.exists(output_file) else 'w'
    # Extract the ids of the videos already predicted 
    with h5py.File(output_file, mode) as predictions_file:
        predicted_videos = predictions_file.keys()
        videos_ids = [v[:-4] for v in os.listdir(videos_dir) if v[-4:] == '.mp4']
        # Lets remove from the list videos_ids, the ones already predicted
        videos_ids_to_predict = list(set(videos_ids) - set(predicted_videos))
        
        nb_videos = len(videos_ids_to_predict)
        print('Total number of videos: {}'.format(len(videos_ids)))
        print('Videos already predicted: {}'.format(len(predicted_videos)))
        print('Videos to predict: {}'.format(nb_videos))
    if nb_videos == 0:
        print('all videos are predicted!!!')
        exit()

    # Creating Parallel Fetching Video Data
    print('Creating {} threads to fetch video data'.format(num_threads))

    data_gen_queue = queue.Queue(maxsize=queue_size)
    data_save_queue = queue.Queue()

    _stop_all_generators = threading.Event()
    _stop_all_extractors = threading.Event()

    class data_generator_task(threading.Thread):
        def __init__(self, threadID, videos_ids_to_predict, videos_dir, extension, length,resize_size):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.videos_ids_to_predict = videos_ids_to_predict
            self.videos_dir  = videos_dir 
            self.extension = extension
            self.length = length 
            self.resize_size = resize_size
        def run(self):
            generator = VideoGenerator(
                self.videos_ids_to_predict,
                self.videos_dir,
                self.extension,
                self.length,
                self.resize_size)
            keep = True
            while keep:
                try:
                    if data_gen_queue.qsize() < queue_size:
                        try:
                            generator_output = next(generator)
                        except ValueError:
                            continue
                        except MemoryError:
                            print("memory error while fetching video.".format())
                            print(traceback.print_exc())
                            continue
                        data_gen_queue.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except StopIteration:
                    print('End')
                    keep = False
                    break
                except Exception:
                    keep = False
                    print('Something went wrong with generator_process. generator: {}'.format(self.threadID))
                    print(traceback.print_exc())
            print('quesize :{}'.format(data_gen_queue.qsize()))
            print('Generator <{}> stopped.'.format(self.threadID))

    class predict_task(threading.Thread):
        def __init__(self, threadID):
            threading.Thread.__init__(self)
            self.threadID = threadID
        def run(self):
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
            LR_mult_dict['fc8']=5
            # Setting up optimizer
            base_lr = 0.00001
            adam = Adam(lr=base_lr, decay=0.00005, multipliers=LR_mult_dict)
            sgd = SGD(lr=base_lr, decay=0.00005, multipliers=LR_mult_dict)
            opt = adam 
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
            print('Compiling done!')
            print('laoding weight-------{}'.format(weights_file))
            model.load_weights(weights_file)
            print("wight <{}> loaded.".format(weights_file))
            print('Starting extracting features')

            while not (_stop_all_generators.is_set() and data_gen_queue.empty()):
                generator_output = None
                
                while True:
                    # print("----{} {} {}".format(_stop_all_extractors.is_set(),data_gen_queue.empty(),data_gen_queue.qsize()))
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
                
                tic = time.time()
                prediction = []
                '''1 window pre batch'''
                # for i in range(len(X)-length+1):
                #     inputs = X[i:i+length]
                #     inputs = inputs.astype(np.float32) 
                #     inputs /=255.
                #     inputs = np.expand_dims(inputs, axis=0)
                #     Y = model.predict_on_batch(inputs)
                #     prediction.append(Y)
                ''' > 1 window '''
                indexes = list(range(len(X)-length+1))
                batch_index = indexes[::batch_size]
                for index in batch_index[:-1]:
                    inputs = []
                    for j in range(batch_size):
                        window = X[index+j : index+j+length]
                        
                        window /= 255.
                        inputs.append(window)
                    inputs = np.array(inputs)
                    Y = model.predict_on_batch(inputs)
                    prediction.extend(Y)
                last_batch_index = batch_index[-1]
                inputs = []
                for index in range(last_batch_index,len(X)-length+1):
                    window = X[index : index+length]
                    
                    window /= 255.
                    inputs.append(window)
                inputs = np.array(inputs)
                Y = model.predict_on_batch(inputs)
                prediction.extend(Y)
                toc = time.time()

                data_save_queue.put((video_id, prediction))
                print("video :{} -done!  {:.2f} second(s)".format(video_id, toc-tic))
            print('prediction task stopped')

    class saver_task(threading.Thread):
        def __init__(self, nb_videos):
            threading.Thread.__init__(self)
            self.counter = 0
        def run(self):
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
            
                with h5py.File(output_file, 'r+') as f:
                    f.create_dataset(video_id, data=predictions, dtype='float32')
                self.counter += 1
                print('Save prediction: {} ({}/{})'.format(video_id,self.counter,nb_videos))
            if self.counter<nb_videos:
                print("{} videos are not processed, please run it again.".format(nb_videos-self.counter))
            print("saver task stopped.")


   
    generator_threads = [
        data_generator_task(i, videos_ids_to_predict[i:nb_videos:num_threads], videos_dir,
            'mp4', length,resize_size)
        for i in range(num_threads)
    ]
    for myThread in generator_threads:
        myThread.daemon = True
        myThread.start()

    
    extractors_threads = [
        predict_task(i)
        for i in range(num_gpus)
    ]
    for t in extractors_threads:
        t.daemon = True
        t.start()

    # Create the threads that will get all the extracted features from the data_save_queue and
    # store it on the hdf5 file.
    
    saver_thread = saver_task(nb_videos)
    saver_thread.daemon = True
    saver_thread.start()

    # Joining processes
    for t in generator_threads:
        t.join()
    _stop_all_generators.set()
    for t in extractors_threads:
        t.join()
    _stop_all_extractors.set()
    saver_thread.join()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='predict the output with the trained model')
    parser.add_argument(
        '-v',
        '--videos-dir',
        type=str,
        dest='video_dir',
        default='/media/lq/C13E-1ED0/dataset/THUMOS/tmptest',
        help='videos directory (default: %(default)s)')
    parser.add_argument(
        '-d',
        '--weight-dir',
        type=str,
        dest='weight_dir',
        default='data/dataset',
        help=
        'directory where to store the predictions (default: %(default)s)'
    )
    parser.add_argument(
        '-w',
        '--weight-name',
        type=str,
        dest='weight_name',
        help=
        'name of the weights'
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
        default=1,
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

    args = parser.parse_args()

    weights_file = os.path.join(args.weight_dir, args.weight_name)
    prediction_file_name = 'predictions_' + args.weight_name
    output_file = os.path.join(args.weight_dir, prediction_file_name)

    predict_videos(videos_dir = args.video_dir, 
                    output_file = output_file, 
                    weights_file = weights_file, 
                        batch_size = args.batch_size,
                        num_threads = args.num_threads,
                        queue_size = args.queue_size,
                        num_gpus = args.num_gpus)