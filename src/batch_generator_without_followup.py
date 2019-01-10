from keras.utils import np_utils
import numpy as np
import random
import cv2
import os

def process_batch(windows, windows_length, img_path, isTrain):
    N = len(windows)
    X_s = np.zeros((N,windows_length,112,112,3),dtype=np.float32) #start windows
    X_s_labels = np.zeros(N,dtype='int')
    X_f =  np.zeros((N,windows_length,112,112,3),dtype=np.float32) #follow up windows
    # X_f_labels = np.zeros(N,dtype='int')
    for i in range(len(windows)): # len(windows) == batch size
        window = windows[i]
        path = window[0]
        start_frame = window[1] 
        label = window[2]

        imgs = os.listdir(img_path+path)
        imgs.sort(key=str.lower)  
        
        crop_x = random.randint(0, 15)
        crop_y = random.randint(0, 58)
        is_flip = random.randint(0, 1) # 1->flip
        for j in range(windows_length):
            '''start window'''
            img_s = imgs[start_frame + j]
            image_s = cv2.imread(img_path + path + '/' + img_s)
            image_s = cv2.resize(image_s, (171,128))
            if isTrain and is_flip == 1:
                image_s = cv2.flip(image_s, 1)
            X_s[i][j][:][:][:] = image_s[crop_x:crop_x + 112, crop_y:crop_y + 112, :]

        X_s_labels[i] = label
    return X_s, X_s_labels


def batch_generator_AS_nonAS_1_1(AS_windows, A_windows, BG_windows, windows_length, batch_size, N_iterations, N_classes, img_path):
    """
    input data generator
    """
    #1/2 AS, 1/2 nonAS
    non_AS_windows = A_windows + BG_windows

    AS_size = batch_size//2
    non_AS_size = batch_size - AS_size
    N_AS = len(AS_windows)

    while True:
        random.shuffle(AS_windows)
        random.shuffle(non_AS_windows)
        index_AS = 0
        index_non_AS = 0
        for i in range(N_iterations):
            batch_windows = []
            a_AS = index_AS
            b_AS = a_AS + AS_size
            a_non_AS = index_non_AS
            b_non_AS = a_non_AS + non_AS_size

            if b_AS > N_AS:
                # print("\nAS windows, index out of range {}".format(i))
                index_AS = 0
                a_AS = 0 
                b_AS = a_AS+AS_size
                random.shuffle(AS_windows)
            # print("{}as    {}:{}".format(i,a_AS,b_AS))
            # print("{}nonas {}:{}".format(i,a_non_AS,b_non_AS))
            batch_windows = AS_windows[a_AS:b_AS] +non_AS_windows[a_non_AS:b_non_AS]
         
            index_AS = b_AS
            index_non_AS = b_non_AS
                       
            random.shuffle(batch_windows)
            
            X_s, X_s_labels = process_batch(batch_windows, windows_length, img_path, isTrain= True)

            X_s /= 255.
            Y = np_utils.to_categorical(np.array(X_s_labels), N_classes)
            yield X_s, Y

def batch_generator_AS(AS_windows, windows_length, batch_size, N_iterations, N_classes, img_path):
    """
    input data generator
    """
    AS_size = batch_size
    N_AS = len(AS_windows)

    while True:
        random.shuffle(AS_windows)    
        index_AS = 0
        for i in range(N_iterations):
            batch_windows = []
            a_AS = index_AS
            b_AS = a_AS + AS_size

            if b_AS > N_AS:
                # print("\nAS windows, index out of range {}".format(i))
                index_AS = 0
                a_AS = 0 
                b_AS = a_AS+AS_size
                random.shuffle(AS_windows)
            batch_windows = AS_windows[a_AS:b_AS]
         
            index_AS = b_AS
                       
            random.shuffle(batch_windows)
            
            X_s, X_s_labels = process_batch(batch_windows, windows_length, img_path, isTrain= True)

            X_s /= 255.
            Y = np_utils.to_categorical(np.array(X_s_labels), N_classes)
            yield X_s, Y