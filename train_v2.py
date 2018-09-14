# -*- coding:utf-8 -*-
import json
import sys
import os
from models.modelv2 import c3d_model
from keras.optimizers import SGD,Adam
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

def use_cpu():
    import tensorflow as tf
    from keras import backend as K

    num_cores = 8

    # if GPU:
    #     num_GPU = 1
    #     num_CPU = 1
    # if CPU:
    num_CPU = 1
    num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)

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

        
def process_batch(windows, windows_length, img_path, train=True):
    N = len(windows)
    X_s = np.zeros((N,windows_length,112,112,3),dtype='float32') #start windows
    X_s_labels = np.zeros(N,dtype='int')
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
            for j in range(windows_length):
                global img_s
                global img
                try:
                    '''start window'''
                    img_s = imgs[start_frame + j]
                    image_s = cv2.imread(img_path + path + '/' + img_s)
                    image_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2RGB)
                    image_s = cv2.resize(image_s, (171, 128))
                    X_s[i][j][:][:][:] = image_s[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
                    '''follow up window'''
                    img = imgs[follow_frame + j]###                
                    image_f = cv2.imread(img_path + path + '/' + img)
                    image_f = cv2.cvtColor(image_f, cv2.COLOR_BGR2RGB)
                    image_f = cv2.resize(image_f, (171, 128))
                    X_f[i][j][:][:][:] = image_f[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print(img_path + path + '/' + img_s)
                    print(img_path + path + '/' + img)
                    print("total frame: " + str(len(imgs)))
                    print("start frame: " +str(start_frame))
                    print("follow frame: " + str(follow_frame))
                    print("j = " +str(j))
                    print(window)
                    break
            X_s_labels[i] = label
            # X_f_labels[i] = follow_label
        else:
            for j in range(windows_length):
                img = imgs[start_frame + j]
                image_s = cv2.imread(img_path + path + '/' + img)
                image_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2RGB)
                image_s = cv2.resize(image_s, (171, 128))
                X_s[i][j][:][:][:] = image_s[8:120, 30:142, :]
                img = imgs[follow_frame + j]
                image_f = cv2.imread(img_path + path + '/' + img)
                image_f = cv2.cvtColor(image_f, cv2.COLOR_BGR2RGB)
                image_f = cv2.resize(image_f, (171, 128))
                X_f[i][j][:][:][:] = image_f[8:120, 30:142, :]
            X_s_labels[i] = label
            # X_f_labels[i] = follow_label
    return X_s, X_f,  X_s_labels



def batch_generator(AS_windows, non_AS_windows, windows_length, batch_size, N_iterations, N_classes, img_path, isTrain=True):
    """
    input data generator
    """
    batch_size_AS = batch_size>>1
    batch_size_non_AS = batch_size - batch_size_AS
    random.shuffle(AS_windows)
    random.shuffle(non_AS_windows)
    while True:
        for i in range(N_iterations):
            a_AS = i*batch_size_AS
            b_AS = (i+1)*batch_size_AS
            a_non_AS = i*batch_size_non_AS
            b_non_AS = (i+1)*batch_size_non_AS
            
            batch_windows = AS_windows[a_AS:b_AS] + non_AS_windows[a_non_AS:b_non_AS]
            random.shuffle(batch_windows)
            
            X_s, X_f, X_s_labels = process_batch(batch_windows, windows_length, img_path, train=isTrain)
            

            X_s /= 255.
            X_f /=255.
            Y = np_utils.to_categorical(np.array(X_s_labels), N_classes)
            
            inputs = [X_s, X_f]
            yield inputs, Y
            # yield X_s, Y

# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


def main():
    from data import videoPaths as path    
    img_path = path.VALIDATION_IMAGES_PATH

    use_cpu()

    N_classes = 20+1
    batch_size = 16
    epochs = 16
    input_shape = (16,112,112,3)
    windows_length = 16

    model = c3d_model(input_shape)
    lr = 0.0001
    #sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    adam = Adam(lr=lr)

    model.compile(loss=['categorical_crossentropy',zero_loss], loss_weights = [1,0.1], optimizer=adam, metrics=['accuracy'])
    model.summary()
    from dataUtil import load_train_data, load_val_data
    train_AS_windows, train_non_AS_windows = load_train_data() # load train data
    N_train_samples = len(train_AS_windows) << 1 #  N_train_samples = len(train_AS_windows) * 2, half AS, half non-AS
    N_train_iterations = N_train_samples // batch_size 

    val_AS_windows, val_non_AS_windows = load_val_data() # load val data
    N_val_samples = len(val_AS_windows) << 1
    N_val_iterations = N_val_samples//batch_size
# ####################################   
#     print(len(train_AS_windows))
#     print(len(train_non_AS_windows))
#     print(len(val_AS_windows))
#     print(len(val_non_AS_windows))
    # a=batch_generator(train_AS_windows, train_non_AS_windows, windows_length, batch_size, N_train_iterations, N_classes,img_path,isTrain=True)
    # print(type(next(a)))
#     # test_data, y = next(a)
#     # print(len(test_data))
#     # print(len(y))
#     # print(test_data[0])
#     # print(test_data.shape)
#     # print(test_data[0].shape)
#     # util.show_images(test_data[0])
#     # plt.imshow(test_data[0])
#     # plt.show()
# ##################################


    history = model.fit_generator(batch_generator(train_AS_windows, train_non_AS_windows, 
                                                    windows_length, batch_size, N_train_iterations, N_classes,img_path,isTrain=True),
                                  steps_per_epoch= N_train_iterations,
                                  epochs=epochs,
                                  callbacks=[onetenth_4_8_12(lr)],
                                  validation_data=batch_generator(val_AS_windows, val_non_AS_windows, 
                                                    windows_length, batch_size, N_val_iterations, N_classes,img_path, isTrain=False),
                                  validation_steps= N_val_iterations,
                                  verbose=1)

    if not os.path.exists('results/'):
        os.mkdir('results/')
    plot_history(history, 'results/')
    save_history(history, 'results/')
    model.save_weights('results/weights_c3d.h5')


if __name__ == '__main__':
    main()
    # util.send_email()
    # loss = sum( [ loss_function( output_true, output_pred ) for ( output_true, output_pred ) in zip( outputs_data, outputs_model ) ] )
