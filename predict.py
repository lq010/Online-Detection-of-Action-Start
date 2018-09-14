# coding=utf8
from models.model import c3d_model
from keras.optimizers import SGD,Adam
import keras.backend as K
from keras.utils import np_utils
import numpy as np
import cv2


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

    model = c3d_model(input_shape)
    # init model
    model = c3d_model(input_shape)
    lr = 0.005
    adam = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    model.load_weights('results/weights_c3d.h5')

    # read video
    video = 'videos/video_test_0000007.mp4'
    cap = cv2.VideoCapture(video)

    clip = []
    counter = 0 
    while True:
        ret, frame = cap.read()
        counter += 1
        if counter < 8250:
            continue
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
            # cv2.imshow('result', frame)
            # cv2.waitKey(10)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()