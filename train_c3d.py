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

import time 

from my_callbacks import LearningRateTracker
from keras.callbacks import CSVLogger

from src.LR_Adam import Adam
from src.LR_SGD import SGD
from src.batch_generator_without_followup import batch_generator_AS_A_BG_2_1_1
from src.batch_generator_without_followup import val_batch_generator

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


def main(id):
    from data import videoPaths as path    
    img_path = path.VALIDATION_IMAGES_PATH
    weights_dir = './weight'
    model_weight_filename = os.path.join(weights_dir, 'sports1M_weights_tf.h5')


    N_classes = 20+1
    batch_size = 16
    epochs = 4
    input_shape = (16,112,112,3)
    windows_length = 16

    model = c3d_model.get_model()
    
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
    # sgd = SGD(lr=base_lr, decay=0.00005, multipliers=LR_mult_dict)
    opt = adam
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
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
    train_AS_windows, train_A_windows, train_BG_windows = load_train_data() # load train data
   
    # N_A_samples = len(train_A_windows)
    # N_batch_groups = N_A_samples // (batch_size//2 + batch_size//4)
    # N_train_iterations = N_batch_groups * 2

    #N_train_samples = len(train_AS_windows) *2 << 1 #  half AS, half non-AS
    N_train_samples = len(train_AS_windows) * 2
    N_train_iterations = N_train_samples // batch_size 

    val_AS_windows, val_A_windows, val_BG_windows = load_val_data() # load val data

    N_val_samples = len(val_AS_windows)*2
    # N_val_samples = len(val_AS_windows) << 1
    N_val_iterations = N_val_samples//batch_size
# ####################################   
    print("--#train AS windows: "+ str(len(train_AS_windows)) +" #train A windows: "+str(len(train_A_windows))+" #train BG windows: "+str(len(train_BG_windows)))
    print("-N_val_samples:"+str(N_val_samples)+ 
            "\n--#val AS windows: "+ str(len(val_AS_windows)) + " #val A windows: "+ str(len(val_A_windows))+ " #val BG windows: "+ str(len(val_BG_windows)))

    # a=batch_generator(train_AS_windows, train_A_windows, train_BG_windows, windows_length, batch_size, N_train_iterations, N_classes,img_path,isTrain=True)
    # for i in range(N_train_iterations):
    #     print("# " + str(i))
    #     length = next(a)
    #     try:    
    #         assert length == batch_size
    #     except AssertionError:
    #         print("error:{}".format(len(test_data)))
    #         return
        
    # return 
    # next(a)
    # test_data, y = next(a)
    # print(len(test_data))
    # print(len(y))
    # return
    # print(test_data[0])
    # print(test_data.shape)
    # print(test_data[0].shape)
    # util.show_images(test_data[0])
    # plt.imshow(test_data[0])
    # plt.show()
# ##################################
    result_dir = './results/adam_c3d_{}/'.format(id)
    best_weight_dir = result_dir+ 'weights' 
    best_weight_name = best_weight_dir + '/weights.{epoch:02d}-{val_loss:.3f}.hdf5'
    if not os.path.isdir(best_weight_dir):
        os.makedirs(best_weight_dir)
    if not os.path.exists(best_weight_dir):
        os.makedirs(result_dir)
    desp = result_dir+'desp.txt'
    with open(desp,'w') as f:
        f.write('batch size: {}\nbase_lr: {} \ntrain_samples:{} \nval_samples:{}\n '.format(batch_size,base_lr,N_train_samples,N_val_samples))
        f.write('init_weiht: {}'.format(model_weight_filename))
        f.write('batch_generator: {}'.format(batch_generator_AS_A_BG_2_1_1))
    # callbacks
    csv_logger = CSVLogger(result_dir +'/log.csv', separator=',')
    checkpointer = ModelCheckpoint(filepath=best_weight_name, verbose=1, save_best_only=False,save_weights_only=True)
    # NAME = "THUMOS-{}".format(int(time.time()))
    log_dir = os.path.join(result_dir,'log')
    tbCallBack = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    train_generator = batch_generator_AS_A_BG_2_1_1(train_AS_windows, train_A_windows, train_BG_windows,
                                                    windows_length, batch_size, N_train_iterations, N_classes,img_path)
    val_generator = val_batch_generator(val_AS_windows, val_A_windows, val_BG_windows,
                                                    windows_length, batch_size, N_val_iterations, N_classes,img_path)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch = 2,#N_train_iterations,
                                  epochs = epochs,
                                  callbacks=[csv_logger, tbCallBack, checkpointer],
                                  validation_data = val_generator,
                                  validation_steps = 2,#N_val_iterations,
                                  verbose=1)

    plot_history(history, result_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model ')

    parser.add_argument(
        '-id',
        dest='experiment_id',
        default=0,
        help='Experiment ID to track and not overwrite resulting models')

    args = parser.parse_args()
    main(args.experiment_id)

    # util.send_email()
    # loss = sum( [ loss_function( output_true, output_pred ) for ( output_true, output_pred ) in zip( outputs_data, outputs_model ) ] )
