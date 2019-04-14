# coding=utf8
from models import c3d_model
import keras.backend as K
from keras.utils import np_utils
import numpy as np
import cv2
import os
from src.LR_Adam import Adam
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

    input_shape = (16,112,112,3)
    windows_length = 16

    model = c3d_model.get_model()



    model.summary()
    weights_dir = './weight'
    model_weight_filename = os.path.join(weights_dir, 'sports1M_weights_tf.h5')
    c3d = '/media/lq/C13E-1ED0/dataset/THUMOS/result/adam_2epoches/best_weight/weights.hdf5'
    c3d_TC_weights ='/home/lq/Documents/Thesis/Thesis/results/adam_temporal_8/weights/weights.02-2.111.hdf5'
    c3d_TC_GAN_weights = '/home/lq/Documents/Thesis/Thesis/results/gan_9/weights/c3d_TC_GAN_21_outputs_it1500.hdf5'
    model.load_weights(c3d)
    # model.load_weights('results/weights_c3d.h5')

    # read video

    train_baseball = '/media/lq/C13E-1ED0/dataset/THUMOS/validation/video_validation_0000690.mp4'
    train_basketball = '/media/lq/C13E-1ED0/dataset/THUMOS/validation/video_validation_0000901.mp4' #this video used as train data
    train_billiards ='/media/lq/C13E-1ED0/dataset/THUMOS/validation/video_validation_0000055.mp4'
    
    val_baseball = '/media/lq/C13E-1ED0/dataset/THUMOS/validation/video_validation_0000687.mp4'
    val_basketball = '/media/lq/C13E-1ED0/dataset/THUMOS/validation/video_validation_0000907.mp4'
    val_billiards = '/media/lq/C13E-1ED0/dataset/THUMOS/validation/video_validation_0000057.mp4'
    val_golf = '/media/lq/C13E-1ED0/dataset/THUMOS/validation/video_validation_0000282.mp4'

    test_firsbeeCatch= '/media/lq/C13E-1ED0/dataset/THUMOS/test/video_test_0000413.mp4'
    test_baseball = '/media/lq/C13E-1ED0/dataset/THUMOS/test/video_test_0000664.mp4'
    test_basketball= '/media/lq/C13E-1ED0/dataset/THUMOS/test/video_test_0000179.mp4'
    test_billiards = '/media/lq/C13E-1ED0/dataset/THUMOS/test/video_test_0001146.mp4'
    
    bike = '/home/lq/Documents/Thesis/C3D-keras-master/videos/v_Biking_g05_c02.avi'
    golf = '/media/lq/C13E-1ED0/dataset/THUMOS/test/video_test_0000238.mp4'
    basketball = '/media/lq/C13E-1ED0/dataset/THUMOS/tmptest/video_test_0000179.mp4'
    
    CleanandJerk ='/media/lq/C13E-1ED0/dataset/THUMOS/111/video_test_0000698.mp4'
    large = '/media/lq/C13E-1ED0/dataset/THUMOS/test_large/video_test_0000793.mp4'
    test_cleanandjerk = '/media/lq/C13E-1ED0/dataset/THUMOS/test/video_test_0001307.mp4'
    video = test_basketball
    
    cap = cv2.VideoCapture(test_cleanandjerk)

    clip = []
    counter = 0 
    while True:
        ret, frame = cap.read()
        counter += 1
        # if counter < 8250:
        #     continue
        if ret:
            #tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(frame, (171, 128))[8:120,30:142,:])
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
                text = class_index[label]
                color = (0, 0, 255) if not text == 'background' else (0,0,0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale=0.6
                (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
                text_offset_x = 20
                text_offset_y = 20
                # set the rectangle background to white
                rectangle_bgr = (255, 255, 255)
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width+2, text_offset_y - text_height -5))

                cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
                cv2.putText(frame, class_index[label], (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 1)

                text2 = "prob: %.4f" % pred[0][label]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale=0.6
                (text_width, text_height) = cv2.getTextSize(text2, font, fontScale=font_scale, thickness=1)[0]
                text_offset_x = 20
                text_offset_y = 40
                # set the rectangle background to white
                rectangle_bgr = (255, 255, 255)
                box_coords = ((text_offset_x, text_offset_y+2), (text_offset_x + text_width +2, text_offset_y - text_height -5))
                cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

                cv2.putText(frame, text2, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 1)
                clip.pop(0)
            cv2.imshow('result', frame)
            # cv2.imwrite( "/media/lq/C13E-1ED0/dataset/clean_example/img_{}.jpg".format(counter), frame );
            # time.sleep(0.2)
            cv2.waitKey(10)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()