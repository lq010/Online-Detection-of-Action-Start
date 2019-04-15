# coding=utf8
from models import  c3d_model
import keras.backend as K
from keras.utils import np_utils
import numpy as np
import cv2
import os
from ucf_src.LR_Adam import Adam
import time 
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
class_index[21] = 'fake'
print(class_index)
def main():

    
    windows_length = 16

    model = c3d_model.get_model(2)



    model.summary()
    weights_dir = './weight'
    # model_weight_filename = os.path.join(weights_dir, 'sports1M_weights_tf.h5')
  
    
    gan= '/media/lq/C13E-1ED0/dataset/UCF_Crimes/results/gan_f1/weights/c3d_TC_GAN_1_outputs_it200.hdf5'
    tc= '/media/lq/C13E-1ED0/dataset/UCF_Crimes/results/thesis/adam_temporal_f1/weights/weights.03.hdf5'
    c3d='/media/lq/C13E-1ED0/dataset/UCF_Crimes/results/thesis/adam_c3d_c3d_as_211/weights/weights.04.hdf5'
    model.load_weights('/media/lq/C13E-1ED0/dataset/UCF_Crimes/results/adam_c3d_final_1_retrain_march21/weights/weights.09-1.660.hdf5')#, by_name = True, skip_mismatch=True, reshape=True)
    # model.load_weights('results/weights_c3d.h5')

    # read video




    crime = '/media/lq/C13E-1ED0/dataset/UCF_Crimes/Videos/Arson/Arson009_x264.mp4'
    fight = '/media/lq/C13E-1ED0/dataset/UCF_Crimes/Videos/Fighting/Fighting047_x264.mp4'
    accident = '/media/lq/C13E-1ED0/dataset/UCF_Crimes/Videos/RoadAccidents_test/RoadAccidents002_x264.mp4'
    cap = cv2.VideoCapture(accident)

    clip = []
    counter = 0 
    while True:
        ret, frame = cap.read()
        counter += 1
        # if counter < 2300:
        #     continue
        if ret:
            #tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(frame, (160, 120)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                
                # inputs = np_utils.normalize(inputs)
                # inputs = inputs[:,8:120,30:142,:]
                inputs /= 255.
                # inputs = np.expand_dims(inputs, axis=0)
                # inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
                input_window = []
                input_window.append(inputs)
                input_window = np.array(input_window).astype(np.float32)
                print('----')
                print(input_window.shape)
                pred = model.predict(input_window)
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
            # cv2.imwrite( "/media/lq/C13E-1ED0/dataset/c3d_ucf/img_{}.jpg".format(counter), frame );

            # time.sleep(0.1)
            cv2.waitKey(10)
        else:
            print ('err')
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()