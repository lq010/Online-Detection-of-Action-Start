import argparse
import json
import sys
import os
from models import c3d_model
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
    for i in range(len(windows)):
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
        is_flip = random.randint(0, 1) #add flip
        for j in range(windows_length):
            '''start window'''
            img_s = imgs[start_frame + j]
            image_s = cv2.imread(img_path + path + '/' + img_s)
            image_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2RGB)
            image_s = cv2.resize(image_s, (171, 128))
            if isTrain and is_flip == 1:
                image_s = cv2.flip(image_s, 1)
            X_s[i][j][:][:][:] = image_s[crop_x:crop_x + 112, crop_y:crop_y + 112, :]

        X_s_labels[i] = label
    return X_s, X_s_labels


def batch_generator(AS_windows, non_AS_windows, windows_length, batch_size, N_iterations, N_classes, img_path, isTrain= True):
    """
    input data generator
    """
    batch_size_AS = batch_size>>1
    # if isTrain:
        # batch_size_AS = batch_size//3
    # else:
    #     batch_size_AS = batch_size>>2

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
            
            X_s, X_s_labels = process_batch(batch_windows, windows_length, img_path, isTrain)

            X_s /= 255.
            Y = np_utils.to_categorical(np.array(X_s_labels), N_classes)
            yield X_s, Y


def main(force_cpu):
    from data import videoPaths as path    
    img_path = path.VALIDATION_IMAGES_PATH
    weights_dir = './weight'
    model_weight_filename = os.path.join(weights_dir, 'sports1M_weights_tf.h5')

    if(force_cpu):
        print('using CPU')
        util.force_cpu()
    else:
        print('using GPU')

    N_classes = 20+1
    batch_size = 16 #24
    epochs = 16
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
    LR_mult_dict['fc8']=10

    # Setting up optimizer
    base_lr = 0.00001
    adam = Adam(lr=base_lr, decay=0.00005, multipliers=LR_mult_dict)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
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
    N_train_samples = len(train_AS_windows) << 1 #  N_train_samples = len(train_AS_windows) * 2, half AS, half non-AS
    # N_train_samples = len(train_AS_windows) * 3
    N_train_iterations = N_train_samples // batch_size 

    val_AS_windows, val_A_windows, val_BG_windows = load_val_data() # load val data
    N_val_samples = len(val_AS_windows) << 1
    N_val_iterations = N_val_samples//batch_size
# ####################################   
    print("#train samples:" + str(N_train_samples) 
         +"\n --#train AS windows: "+ str(len(train_AS_windows)) +" #train A windows: "+str(len(train_A_windows))+" #train BG windows: "+str(len(train_BG_windows)))
    print("#val samples:" + str(N_val_samples) 
         +"\n --#val AS windows: "+ str(len(val_AS_windows)) + " #val non_A windows: "+ str(len(val_A_windows))+ " #val non_BG windows: "+ str(len(val_BG_windows)))

    # a=batch_generator(train_AS_windows, train_non_AS_windows, windows_length, batch_size, N_train_iterations, N_classes,img_path,isTrain=True)
    # next(a)
    # test_data, y = next(a)
    # print(len(test_data))
    # print(len(y))
    # print(test_data[0])
    # print(test_data.shape)
    # print(test_data[0].shape)
    # util.show_images(test_data[0])
    # plt.imshow(test_data[0])
    # plt.show()
# ##################################

    history = model.fit_generator(batch_generator(train_AS_windows, train_non_AS_windows, 
                                                    windows_length, batch_size, N_train_iterations, N_classes,img_path),
                                  steps_per_epoch = N_train_iterations,
                                  epochs = epochs,
                                  validation_data = batch_generator(val_AS_windows, val_non_AS_windows, 
                                                    windows_length, batch_size, N_val_iterations, N_classes,img_path, isTrain = False),
                                  validation_steps = N_val_iterations,
                                  verbose=1)

    if not os.path.exists('results/'):
        os.mkdir('results/')
    plot_history(history, 'results/')
    save_history(history, 'results/')
    model.save_weights('results/weights_c3d.h5')


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
