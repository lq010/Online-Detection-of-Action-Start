'''Test the input data, to make sure there's no error'''
from train import preprocess_input
import json
import sys
import os
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
input_shape = (112,112,16,3)
windows_length = 16

train_AS_windows, train_non_AS_windows = preprocess_input(train_anno, windows_length)
N_train_samples = len(train_AS_windows) << 1 # half AS, half non-AS
N_train_iterations = N_train_samples // batch_size
val_AS_windows, val_non_AS_windows = preprocess_input(val_anno, windows_length)
N_val_samples = len(val_AS_windows) << 1
N_val_iterations = N_val_samples//batch_size


import threading
import time

def process_batch(windows, windows_length, img_path, threadId,train=True):
    # batch = np.zeros((num,16,128,171,3),dtype='float32')
    counter = 0
    for i in range(len(windows)):
        
        if (counter%1000 == 0):
            print('Thread ' + str(threadId)+"   "+ str(counter))
        counter += 1
        window = windows[i]
        path = window[0]
        start_frame = window[1] 
        label = window[2]
        follow_frame = window[3]
        # follow_label = windows[4]

        imgs = os.listdir(img_path+path)
        imgs.sort(key=str.lower)
        
        if train:
            for j in range(windows_length):
                global img_s
                global img      
                try:
                    '''start window'''
                    img_s = imgs[start_frame + j]
                    '''follow up window'''
                    img = imgs[follow_frame + j]###                
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print('Thread-' + str(threadId))
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print(img_path + path + '/' + img_s)
                    print(img_path + path + '/' + img)
                    print("total frame: " + str(len(imgs)))
                    print("start frame: " +str(start_frame))
                    print("follow frame: " + str(follow_frame))
                    print("j = " +str(j))
                    print(window)
                    print('--------')
                    break
    return 0

class myThread (threading.Thread):
    def __init__(self, threadID, name, windows, windows_length, img_path):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.windows = windows
    def run(self):
        print ("Starting " + self.name)
        process_batch(self.windows, windows_length, img_path, self.threadID, train=True)
        print ("Exiting " + self.name)

# print(train_AS_windows[:10])
N_thread = 10
N = len(train_non_AS_windows)//N_thread +1
print(len(train_non_AS_windows[0:N]))
threads = list()
s = 0
e = N
print('creating threads')
for i in range(N_thread):
    print('Thread-'+str(i) + ": " + str(s)+'-'+str(e))
    t = myThread(i, "Thread-"+str(i), train_non_AS_windows[s:e], windows_length, img_path )
    threads.append(t)
    s += N
    e += N

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
print ("Exiting Main Thread")



