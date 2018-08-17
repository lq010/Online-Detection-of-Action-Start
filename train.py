# -*- coding:utf-8 -*-
import json
import sys
import os
from models.model import c3d_model
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from schedules import onetenth_4_8_12
import numpy as np
import random
import cv2
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


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
    random.shuffle(windows)
    N = len(windows)
    # batch = np.zeros((num,16,128,171,3),dtype='float32')
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


def preprocess(inputs):
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    # inputs /=255.
    # inputs -= 0.5
    # inputs *=2.
    return inputs

def preprocess_input(input_dict, windows_length):
    """
    process the Action Start windows and non-Action Start windows
    input:
        input_dict: the dictionary of input data (train or val)
        windows_length: the length of the windows (16)
    return:
        the list of AS windows and non-AS windows
    """
    N_instance = 0
    AS_windows = [] #Action Start windows, the first frame number of the AS windows
    non_AS_windows = []

    for videoName, video in input_dict.items():
        N_instance += len(video['frameStamp']) # the number of action instances 
        N_frames_of_video = video['totFrames'] # the number of frames of each video
        leading_frame_of_last_window = N_frames_of_video - windows_length - windows_length # sub 2 window_length => 1 for s_window, 1 for f_window
        exclusive = []
        #Action Start windows
        for instance in video['frameStamp']:
            start_frame = instance[0]
            end_frame = instance[1]
            instance_label = instance[2]
            #action start(background + action)
            for n in range(max(start_frame - windows_length + 1 ,0), min(start_frame + 1 ,leading_frame_of_last_window)): #each start frame exist in 'windows_length' windows
                follow_start_frame = n + windows_length
                follow_instance_label = instance_label if (follow_start_frame+windows_length -1 ) <= end_frame else 0
                AS_windows.append([videoName, n, instance_label, follow_start_frame, follow_instance_label])#
                exclusive.append(n)
            #only action ,min(a,leading_frame_of_last_window)=> the annotation is out of range,
            for n in range(start_frame +1 , min(end_frame-windows_length +1,leading_frame_of_last_window)):
                follow_non_start_frame = n + windows_length
                follow_instance_label = instance_label if (follow_non_start_frame+windows_length -1 ) <= end_frame else 0
                non_AS_windows.append([videoName, n, instance_label, follow_non_start_frame, follow_instance_label]) #
                exclusive.append(n)
        
        #non-Action Start windows
        for n in range(leading_frame_of_last_window):
            if n in exclusive:
                continue
            non_AS_windows.append([videoName, n, 0, n+windows_length, 0]) # TODO

    random.shuffle(AS_windows)
    random.shuffle(non_AS_windows)
    return AS_windows, non_AS_windows

def batch_generator(AS_windows, non_AS_windows, windows_length, batch_size, N_iterations, N_classes, img_path, isTrain=True):
    """
    input data generator
    """
    batch_size_AS = batch_size>>1
    batch_size_non_AS = batch_size - batch_size_AS
    # print(AS_windows[:10])
    # print(non_AS_windows[:10])
    while True:
        for i in range(N_iterations):
            a_AS = i*batch_size_AS
            b_AS = (i+1)*batch_size_AS
            a_non_AS = i*batch_size_non_AS
            b_non_AS = (i+1)*batch_size_non_AS

            X_s, X_f, X_s_labels = process_batch(AS_windows[a_AS:b_AS] + non_AS_windows[a_non_AS:b_non_AS], windows_length, img_path, train=isTrain)
            
            X_s = preprocess(X_s)
            X_f = preprocess(X_f)
            Y = np_utils.to_categorical(np.array(X_s_labels), N_classes)
            # X_s = np.transpose(X_s, (0,2,3,1,4))
            # X_f = np.transpose(X_f, (0,2,3,1,4))
            yield X_s, Y
            # yield X_s, X_f, Y



def main():
    import constantPaths as path    
    img_path = path.VALIDATION_IMAGES_PATH
    train_file = 'data/train.json'
    val_file = 'data/validation.json'

    with open(train_file) as f:
        train_anno = json.load(f)
    with open(val_file) as f:
        val_anno = json.load(f)
  
    N_classes = 20+1
    batch_size = 2#16
    epochs = 2
    input_shape = (16,112,112,3)
    windows_length = 16

    model = c3d_model(input_shape)
    lr = 0.0001
    #sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    adam = Adam(lr=lr)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

####################################   
    # a=batch_generator(train_anno, windows_length, batch_size, N_classes,img_path,isTrain=True)
    # next(a)
    # print(len(a))
    # test_data, y = next(a)
    # print(len(test_data))
    # print(len(y))
##################################
    train_AS_windows, train_non_AS_windows = preprocess_input(train_anno, windows_length)
    N_train_samples = len(train_AS_windows) << 1 # half AS, half non-AS
    N_train_iterations = N_train_samples // batch_size
    val_AS_windows, val_non_AS_windows = preprocess_input(val_anno, windows_length)
    N_val_samples = len(val_AS_windows) << 1
    N_val_iterations = N_val_samples//batch_size

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
