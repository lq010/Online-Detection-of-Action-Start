import argparse
import json
import sys
import os
from models.c3d_temporal_model import c3d_temportal_model
import keras.backend as K
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint

from LR_Adam import Adam
from LR_SGD import SGD

import keras.backend as K
from keras.utils import np_utils
from schedules import onetenth_4_8_12
import numpy as np
import random
import cv2
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import util
import time

import tensorflow as tf
def plot_history(history, result_dir):
    plt.plot(history.history['fc8_acc'], marker='.')
    plt.plot(history.history['val_fc8_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['fc8_acc', 'val_fc8_acc'], loc='lower right')
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
    acc = history.history['fc8_acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_fc8_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()

        
def process_batch(windows, windows_length, img_path, train=True):
    N = len(windows)
    X_s = np.zeros((N,windows_length,112,112,3),dtype='float32') #start windows
    X_s_labels = np.zeros(N,dtype='float32')
    X_f =  np.zeros((N,windows_length,112,112,3),dtype='float32') #follow up windows
    # X_f_labels = np.zeros(N,dtype='int')
    for i in range(len(windows)):
        window = windows[i]
        path = window[0]
        start_frame = window[1] 
        label = window[2]
        follow_frame = window[3]
        # follow_label = windows[4]

        imgs = os.listdir(img_path+path)
        imgs.sort(key=str.lower)
        
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1) # 1->flip
            for j in range(windows_length):
                global img_s
                global img_f

                '''start window'''
                img_s = imgs[start_frame + j]
                image_s = cv2.imread(img_path + path + '/' + img_s)                    
                image_s = cv2.resize(image_s, (171, 128))
                '''follow up window'''
                img_f = imgs[follow_frame + j]###                
                image_f = cv2.imread(img_path + path + '/' + img_f)
                image_f = cv2.resize(image_f, (171, 128))

                if is_flip == 1:
                    image_s = cv2.flip(image_s, 1)
                    image_f = cv2.flip(image_s, 1)
                X_s[i][j][:][:][:] = image_s[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
                X_f[i][j][:][:][:] = image_f[crop_x:crop_x + 112, crop_y:crop_y + 112, :]

            X_s_labels[i] = label
            # X_f_labels[i] = follow_label
        else:
            for j in range(windows_length):
                img = imgs[start_frame + j]
                image_s = cv2.imread(img_path + path + '/' + img)
                image_s = cv2.resize(image_s, (171, 128))
                X_s[i][j][:][:][:] = image_s[8:120, 30:142, :]
                img = imgs[follow_frame + j]
                image_f = cv2.imread(img_path + path + '/' + img)
                image_f = cv2.resize(image_f, (171, 128))
                X_f[i][j][:][:][:] = image_f[8:120, 30:142, :]
            X_s_labels[i] = label
            # X_f_labels[i] = follow_label
    return X_s, X_f,  X_s_labels



def batch_generator(AS_windows, A_windows, BG_windows, windows_length, batch_size, N_iterations, N_classes, img_path, isTrain= True):
    """
    input data generator
    """
    #1/2 AS, 1/4 A, 1/4 BG
    AS_size = batch_size >> 1
    A_size = AS_size >> 1
    BG_size = batch_size - AS_size - A_size

    random.shuffle(AS_windows)
    random.shuffle(A_windows)
    random.shuffle(BG_windows)

    N_AS = len(AS_windows)
    index_AS = 0
    index_A = 0
    index_BG = 0
    while True:
        for i in range(N_iterations):
            a_AS = index_AS
            b_AS = a_AS + AS_size
            a_A = index_A
            b_A = a_A + A_size
            a_BG = index_BG 
            b_BG = a_BG + BG_size
            if b_AS > N_AS:
                print("\nAS windows, index out of range")
                index_AS = 0
                a_AS = 0 
                b_AS = a_AS+AS_size
                random.shuffle(AS_windows)

            batch_windows = AS_windows[a_AS:b_AS] + A_windows[a_A:b_A] + BG_windows[a_BG:b_BG]
            index_A = b_A
            index_AS = b_AS
            index_BG = b_BG
            random.shuffle(batch_windows)
            
            X_s, X_f, X_s_labels = process_batch(batch_windows, windows_length, img_path, train=isTrain)
            X_s /= 255.
            X_f /= 255.
            Y = np_utils.to_categorical(np.array(X_s_labels), N_classes)
            
            inputs = [X_s, X_f]
            yield (inputs, [Y,Y])
            # yield X_s, Y

# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

def my_l2 (y_true,y_pred):
    loss = tf.nn.l2_loss(y_pred)
    return loss #tf.reduce_mean(loss)

def main():
    from data import videoPaths as path
    img_path = path.VALIDATION_IMAGES_PATH
    weights_dir = './weight'
    model_weight_filename = os.path.join(weights_dir, 'sports1M_weights_tf.h5')


    N_classes = 20+1
    batch_size = 16
    epochs = 8
    input_shape = (16,112,112,3)
   

    windows_length = 16

    
    model = c3d_temportal_model(input_shape,N_classes)

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
    opt = adam 

    model.compile(loss=['categorical_crossentropy',my_l2],loss_weights=[1,0.001], optimizer=opt, metrics=['accuracy'])
    model.load_weights(model_weight_filename, by_name = True, skip_mismatch=True, reshape=True)
    model.summary()

    from dataUtil import load_train_data, load_val_data
    train_AS_windows, train_A_windows, train_BG_windows = load_train_data() # load train data
    N_train_samples = len(train_AS_windows) * 2
    N_train_iterations = N_train_samples // batch_size

    val_AS_windows, val_A_windows, val_BG_windows = load_val_data() # load val data

    N_val_samples = len(val_A_windows)+len(val_AS_windows)*2
    # N_val_samples = len(val_AS_windows) << 1
    N_val_iterations = N_val_samples//batch_size

# ####################################   
    # a=batch_generator(train_AS_windows, train_A_windows, train_BG_windows, windows_length, batch_size, N_train_iterations, N_classes,img_path,isTrain=True)
    
    # for i in range(N_train_iterations):
    #     print("# " + str(i))
    #     length = next(a)
    #     try:    
    #         assert length == batch_size
    #     except AssertionError:
    #         print("error:{}".format(len(test_data)))
    #         return
    # test_data= next(a)
    # print(len(test_data))
    
    # print(type(test_data[0]))
    # exit()
#     # print(test_data[0])
#     # print(test_data.shape)
#     # print(test_data[0].shape)
#     # util.show_images(test_data[0])
#     # plt.imshow(test_data[0])
#     # plt.show()
    # val_AS_windows, val_A_windows, val_BG_windows = val_AS_windows[:16], val_A_windows[:8], val_BG_windows[:8]
# ##################################


    best_weight_dir = './tmp/'+ 'adam_temporal' 
    if not os.path.isdir(best_weight_dir):
        os.makedirs(best_weight_dir)
    checkpointer = ModelCheckpoint(filepath=best_weight_dir+'/'+'weights.hdf5', verbose=1, save_best_only=True)
    NAME = "THUMOS-{}".format(int(time.time()))
    tbCallBack = callbacks.TensorBoard(log_dir="./log/{}".format(NAME), histogram_freq=0, write_graph=True, write_images=True)
    val_generator  = batch_generator( val_AS_windows, val_A_windows, val_BG_windows,
                                                    windows_length, batch_size, N_val_iterations, N_classes,img_path,isTrain= True)
    train_generator = batch_generator(train_AS_windows, train_A_windows, train_BG_windows, windows_length, batch_size, N_train_iterations, N_classes,img_path,isTrain= True)
    history = model.fit_generator(train_generator,
                                  steps_per_epoch = 2,#N_train_iterations,
                                  epochs = epochs,
                                  callbacks=[tbCallBack,checkpointer],
                                  validation_data = next(val_generator),
                                  validation_steps = 2,#N_val_iterations,
                                  verbose=1)
    result_dir = 'results/'+ 'adam_temporal'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plot_history(history, result_dir)
    save_history(history, result_dir)
    model.save_weights(result_dir +'/'+'weights_c3d_temporal.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model ')

    parser.add_argument(
        '--id',
        dest='experiment_id',
        default=0,
        help='Experiment ID to track and not overwrite resulting models')


    main()
    # util.send_email()
    # loss = sum( [ loss_function( output_true, output_pred ) for ( output_true, output_pred ) in zip( outputs_data, outputs_model ) ] )
