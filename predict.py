# coding=utf8
from models import c3d_model
import keras.backend as K
from keras.utils import np_utils
import numpy as np
import cv2
import os
from LR_Adam import Adam

"""###################################
class index dictionary
"""
class_index = dict()
index_file = open('data/Class Index_Detection.txt','r')
lines = index_file.readlines()
for line in lines:
    item = line.strip().split()
    class_index[int(item[2])] = item[1]
index_file.close()

print(class_index)
def main():

    input_shape = (16,112,112,3)
    windows_length = 16

    model = c3d_model.get_model(input_shape)

    lr = 0.0001
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
    model.summary()
    weights_dir = './weight'
    model_weight_filename = os.path.join(weights_dir, 'sports1M_weights_tf.h5')
    model_weight_filename = './results/weights_c3d.h5'
    model.load_weights(model_weight_filename, by_name = True, skip_mismatch=True, reshape=True)
    # model.load_weights('results/weights_c3d.h5')

    # read video
    bike = '/home/lq/Documents/Thesis/C3D-keras-master/videos/v_Biking_g05_c02.avi'
    video = bike
    cap = cv2.VideoCapture(video)

    clip = []
    counter = 0 
    while True:
        ret, frame = cap.read()
        counter += 1
        # if counter < 8250:
        #     continue
        if ret:
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(tmp, (171, 128)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs = np_utils.normalize(inputs)
                inputs = inputs[:,:,8:120,30:142,:]
                # inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
                pred = model.predict(inputs)
                print(pred)
                label = np.argmax(pred[0])
                print('#############' + str(label))
                cv2.putText(frame, class_index[label], (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                cv2.putText(frame, "prob: %.4f" % pred[0][label], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                clip.pop(0)
            cv2.imshow('result', frame)
            cv2.waitKey(10)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()