import json
import pickle
import random

train_file = 'data/train_accidents.json'

train_windows_file = 'data/train_windows.pickle'

val_file = '/home/lq/Documents/Thesis/Thesis/UCF/data/validation_accidents.json'
val_windows_file = 'data/val_windows.pickle'

windows_length = 16

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
    A_windows = [] #Action
    BG_windows = [] #back ground 

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
            for n in range(max(start_frame - windows_length + 1 ,0), min(start_frame + 1 ,leading_frame_of_last_window)): #
                follow_start_frame = n + windows_length
                # follow_instance_label = instance_label if (follow_start_frame+windows_length -1 ) <= end_frame else 0
                AS_windows.append([videoName, n, instance_label, follow_start_frame]) # , follow_instance_label])#
                exclusive.append(n)
            #only action ,min(a,leading_frame_of_last_window)=> the annotation is out of range,
            for n in range(start_frame +1 , min(end_frame-windows_length +1,leading_frame_of_last_window)):
                follow_non_start_frame = n  #the follow window of action window is itself, same for BG
                # follow_instance_label = instance_label if (follow_non_start_frame+windows_length -1 ) <= end_frame else 0
                A_windows.append([videoName, n, instance_label, follow_non_start_frame]) # , follow_instance_label]) #
                exclusive.append(n)
            #the end of the action, contains the action and background frames(remove noise?)
            for n in range(end_frame - windows_length +1, min(end_frame+1,leading_frame_of_last_window)):
                exclusive.append(n)
        #non-Action Start windows
        for n in range(leading_frame_of_last_window):
            if n in exclusive:
                continue
            BG_windows.append([videoName, n, 0, n]) #

    # random.shuffle(AS_windows)
    random.shuffle(A_windows)
    random.shuffle(BG_windows)
    return AS_windows, A_windows, BG_windows

def init_train_data():
    with open(train_file) as f:
        train_anno = json.load(f)

    t = preprocess_input(train_anno, windows_length)
    with open(train_windows_file, 'wb') as handle:
        pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return t

def init_val_data():
    with open(val_file) as f:
        val_anno = json.load(f)

    v = preprocess_input(val_anno, windows_length)
    with open(val_windows_file, 'wb') as handle:
        pickle.dump(v, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return v



def load_train_data():
    '''return train sets, AS_windows, non_AS_windows'''
    try:
        with open(train_windows_file, 'rb') as handle:
            train_windows = pickle.load(handle)
            assert type(train_windows) is tuple
            assert len(train_windows) == 3
        return train_windows
    except :
        return init_train_data()
    

def load_val_data():
    '''return validation sets, AS_windows, non_AS_windows'''
    try:
        with open(val_windows_file, 'rb') as handle:
            val_windows = pickle.load(handle)
            assert type(val_windows) is tuple
            assert len(val_windows) == 3
        return val_windows
    except :
        return init_val_data()



if __name__ == "__main__":
    train_AS_windows,  train_A_windows,  train_BG_windows = load_train_data() 
    print(len( train_AS_windows))
    print(len( train_A_windows))
    print(len( train_BG_windows))
    # import time
    # start = time.time()
    # with open(train_file) as f:
    #     train_anno = json.load(f)
    # a = preprocess_input(train_anno, windows_length)

    # end = time.time()
    # print(end - start)
    
    # start = time.time()
    
    # b = load_train_data()
    # end = time.time()
    # print(end - start)

    # print(type(a))
    # print(type(b))
    # print(a==b)
    # print(len(a[0]))
    # print(len(b[0]))