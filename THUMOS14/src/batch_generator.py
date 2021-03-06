from keras.utils import np_utils
import numpy as np
import random
import cv2
import os

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
            is_flip = random.randint(0, 1) # 1->flip
            for j in range(windows_length):
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

def val_batch_generator(AS_windows, A_windows, BG_windows, windows_length, batch_size, N_iterations, N_classes, img_path):
    N = (len(AS_windows)+batch_size)//2    
    windows = AS_windows + A_windows[:N] + BG_windows[:N]
    # windows = AS_windows + A_windows + BG_windows
    while True:
        for i in range(N_iterations):
            a = i*batch_size
            b = a+batch_size
            batch_windows = windows[a:b]
            X_s, X_f, X_s_labels = process_batch(batch_windows, windows_length, img_path, train=False)
            X_s /= 255.
            X_f /= 255.
            Y = np_utils.to_categorical(np.array(X_s_labels), N_classes)
            inputs = [X_s, X_f]
            yield (inputs, [Y, Y])# the second 'Y' is useless



def train_batch_generator_AS_A_BG_2_1_1(AS_windows, A_windows, BG_windows, windows_length, batch_size, N_iterations, N_classes, img_path):
    """
    input data generator
    """
    #1/2 AS, 1/4 A, 1/4 BG
    AS_size = batch_size >> 1
    A_size = AS_size >> 1
    BG_size = batch_size - AS_size - A_size
    N_AS = len(AS_windows)

    while True:
        random.shuffle(AS_windows)
        random.shuffle(A_windows)
        random.shuffle(BG_windows)   
        index_AS = 0
        index_A = 0
        index_BG = 0
        for i in range(N_iterations):
            a_AS = index_AS
            b_AS = a_AS + AS_size
            a_A = index_A
            b_A = a_A + A_size
            a_BG = index_BG 
            b_BG = a_BG + BG_size
            if b_AS > N_AS:
                # print("\nAS windows, index out of range")
                index_AS = 0
                a_AS = 0 
                b_AS = a_AS+AS_size
                random.shuffle(AS_windows)

            batch_windows = AS_windows[a_AS:b_AS] + A_windows[a_A:b_A] + BG_windows[a_BG:b_BG]
            index_A = b_A
            index_AS = b_AS
            index_BG = b_BG
            random.shuffle(batch_windows)
            
            X_s, X_f, X_s_labels = process_batch(batch_windows, windows_length, img_path, train=True)
            X_s /= 255.
            X_f /= 255.
            Y = np_utils.to_categorical(np.array(X_s_labels), N_classes)
            
            inputs = [X_s, X_f]
            yield (inputs, [Y,Y])# the second 'Y' is useless
            # yield X_s, Y

def train_batch_generator_AS_nonAS_1_1(AS_windows, A_windows, BG_windows, windows_length, batch_size, N_iterations, N_classes, img_path):
    """
    input data generator
    """
    #1/2 AS, 1/2 non AS
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
            
            X_s, X_f, X_s_labels = process_batch(batch_windows, windows_length, img_path, train=True)
            X_s /= 255.
            X_f /= 255.
            Y = np_utils.to_categorical(np.array(X_s_labels), N_classes)
            
            inputs = [X_s, X_f]
            yield (inputs, [Y,Y])# the second 'Y' is useless
            # yield X_s, Y

