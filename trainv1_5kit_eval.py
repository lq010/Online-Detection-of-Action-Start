import argparse
import json
import sys
import os
from models import c3d_model
import keras.backend as K
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint

from keras.utils import np_utils
from schedules import onetenth_4_8_12
import numpy as np
import random
import cv2
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import util

from my_callbacks import LearningRateTracker

from LR_Adam import Adam

def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()

        
def process_batch(windows, windows_length, img_path, isTrain):
    N = len(windows)
    X_s = np.zeros((N,windows_length,112,112,3),dtype='float32') #start windows
    X_s_labels = np.zeros(N,dtype='int')
    X_f =  np.zeros((N,windows_length,112,112,3),dtype='float32') #follow up windows
    # X_f_labels = np.zeros(N,dtype='int')
    for i in range(len(windows)): # len(windows) == batch size
        window = windows[i]
        path = window[0]
        start_frame = window[1] 
        label = window[2]
        # follow_frame = window[3]
        # follow_label = windows[4]

        imgs = os.listdir(img_path+path)
        imgs.sort(key=str.lower)  
        
        crop_x = random.randint(0, 15)
        crop_y = random.randint(0, 58)
        is_flip = random.randint(0, 1) # 1->flip
        for j in range(windows_length):
            '''start window'''
            img_s = imgs[start_frame + j]
            image_s = cv2.imread(img_path + path + '/' + img_s)
            image_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2RGB)
            image_s = cv2.resize(image_s, (171,128))
            if isTrain and is_flip == 1:
                image_s = cv2.flip(image_s, 1)
            X_s[i][j][:][:][:] = image_s[crop_x:crop_x + 112, crop_y:crop_y + 112, :]

        X_s_labels[i] = label
    return X_s, X_s_labels


def batch_generator(AS_windows, A_windows, BG_windows, windows_length, batch_size, N_iterations, N_classes, img_path):
    """
    input data generator
    """
    non_AS_windows = A_windows + BG_windows

    AS_size = batch_size//2
    non_AS_size = batch_size - AS_size

    random.shuffle(AS_windows)
    random.shuffle(non_AS_windows)
    
    N_AS = len(AS_windows)
    index_AS = 0
    index_non_AS = 0
    while True:
        for i in range(N_iterations):
            batch_windows = []
            # if i%2 == 0: # even, 1/2 AS, 1/4 A, 1/4 BG
            a_AS = index_AS
            b_AS = a_AS + AS_size
            a_non_AS = index_non_AS
            b_non_AS = a_non_AS + index_non_AS

            if b_AS > N_AS:
                print("\nAS windows, index out of range")
                index_AS = 0
                a_AS = 0 
                b_AS = a_AS+AS_size
                random.shuffle(AS_windows)
            batch_windows = AS_windows[a_AS:b_AS] +non_AS_windows[a_non_AS:b_non_AS]
            # else: #odd, 1/2 A, 1/2 BG
            #     a_A = index_A
            #     b_A = a_A + odd_A_size
            #     a_BG = index_BG 
            #     b_BG = a_BG + odd_BG_size
            #     batch_windows = A_windows[a_A:b_A] + BG_windows[a_BG:b_BG]

           
            index_AS = b_AS
            index_non_AS = b_non_AS
                       
            random.shuffle(batch_windows)
            
            X_s, X_s_labels = process_batch(batch_windows, windows_length, img_path, isTrain= True)

            X_s /= 255.
            Y = np_utils.to_categorical(np.array(X_s_labels), N_classes)
            yield X_s, Y

def val_batch_generator(windows, windows_length, batch_size, N_iterations, N_classes, img_path):

    while True:
        for i in range(N_iterations):
            a = i*batch_size
            b = a+batch_size
            batch_windows = windows[a:b]
            X_s, X_s_labels = process_batch(batch_windows, windows_length, img_path, isTrain= False)
            X_s /= 255.
            Y = np_utils.to_categorical(np.array(X_s_labels), N_classes)
            yield X_s, Y

def main(force_cpu):
    from data import videoPaths as path    
    img_path = path.VALIDATION_IMAGES_PATH
  


    if(force_cpu):
        print('using CPU')
        util.force_cpu()
    else:
        print('using GPU')

    N_classes = 20+1
    batch_size = 24
    epochs = 2
    input_shape = (16,112,112,3)
    windows_length = 16

    model = c3d_model.get_model(input_shape)
    
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

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    weights_dir = './weight'
    model_weight_filename = './tmp/1/weights.hdf5'
    model.load_weights(model_weight_filename, by_name = True, skip_mismatch=True, reshape=True)
######################    
    # for layer in model.layers:
    #     weights = layer.get_weights()
       
    #     print(layer)

    # l = model.get_layer(name='fc8')
    # list = l.get_weights()
    # for l in list:
    #     print(l.shape)
    #     print(l[10])
    # return 0
#####################

    model.summary()
       
    from dataUtil import load_train_data, load_val_data

    val_AS_windows, val_A_windows, val_BG_windows = load_val_data() # load val data
    
    N_val_samples = len(val_AS_windows)
    N_val_iterations = N_val_samples//batch_size
    scores = model.evaluate_generator(val_batch_generator( val_AS_windows,
                                                    windows_length, batch_size, N_val_iterations, N_classes,img_path),
                                        steps=N_val_iterations, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

    print("AS: los = {}, Accuracy = {}".format(scores[0], scores[1]))
    
    N_val_samples = len(val_A_windows)
    N_val_iterations = N_val_samples//batch_size
    scores = model.evaluate_generator(val_batch_generator( val_A_windows,
                                                    windows_length, batch_size, N_val_iterations, N_classes,img_path),
                                        steps=N_val_iterations, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

    print("A: los = {}, Accuracy = {}".format(scores[0], scores[1]))

    # N_val_samples = len(val_BG_windows)
    # N_val_iterations = N_val_samples//batch_size
    # scores = model.evaluate_generator(val_batch_generator( val_BG_windows,
    #                                                 windows_length, batch_size, N_val_iterations, N_classes,img_path),
    #                                     steps=N_val_iterations, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

    # print("trained weight BG: los = {}, Accuracy = {}".format(scores[0], scores[1]))


    weights_dir = './weight'
    model_weight_filename = os.path.join(weights_dir, 'sports1M_weights_tf.h5')
    model.load_weights(model_weight_filename, by_name = True, skip_mismatch=True, reshape=True)
        
    N_val_samples = len(val_AS_windows)
    N_val_iterations = N_val_samples//batch_size
    scores = model.evaluate_generator(val_batch_generator( val_AS_windows,
                                                    windows_length, batch_size, N_val_iterations, N_classes,img_path),
                                        steps=N_val_iterations, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

    print("AS: los = {}, Accuracy = {}".format(scores[0], scores[1]))
    
    N_val_samples = len(val_A_windows)
    N_val_iterations = N_val_samples//batch_size
    scores = model.evaluate_generator(val_batch_generator( val_A_windows,
                                                    windows_length, batch_size, N_val_iterations, N_classes,img_path),
                                        steps=N_val_iterations, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

    print("A: los = {}, Accuracy = {}".format(scores[0], scores[1]))

    # scores = model.evaluate_generator(val_batch_generator( val_BG_windows,
    #                                                 windows_length, batch_size, N_val_iterations, N_classes,img_path),
    #                                     steps=N_val_iterations, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

    # print("sports1M weight BG: los = {}, Accuracy = {}".format(scores[0], scores[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model ')

    parser.add_argument(
        '-c','--cpu',
        dest='force_cpu',
        action='store_true',
        default = False,
        help='Force Keras to use CPU (debug)')

    args = parser.parse_args()
    main(args.force_cpu)

    # util.send_email()
    # loss = sum( [ loss_function( output_true, output_pred ) for ( output_true, output_pred ) in zip( outputs_data, outputs_model ) ] )
