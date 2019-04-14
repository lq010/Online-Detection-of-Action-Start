import h5py
import os
import numpy as np
import time
import json

output_dir = 'data/dataset'
# output_path = '/media/lq/C13E-1ED0/dataset/THUMOS/result/adam_temporal_2nd/best_weight/video_features_adam_float64.hdf5'#
# os.path.join(output_dir, 'video_features.hdf5')
prediction_path = '/home/lq/Documents/Thesis/Thesis/results/adam/forEva/predictions_weights.hdf5'
output_file = '/home/lq/Documents/Thesis/Thesis/results/adam/forEva/predictions_val_set.hdf5'
json_path ='/home/lq/Documents/Thesis/Thesis/data/validation.json'

v_list = []
with open(json_path) as f:
    video_list = json.load(f)
    v_list = video_list.keys()




'''
process the prediction. 
return:
    a dictionary:
        |-video name
          |-action class, time, score
          |...
        |... 
'''
result = dict()
i = 0
def print_attrs(name, obj):
    # print(name)
    global i
    if name not in v_list:
        return
    i +=1
    predictions = np.array(obj)
    with h5py.File(output_file, 'r+') as f:
        f.create_dataset(name, data=predictions, dtype='float32')


def get_result():
    global i
    with h5py.File(prediction_path, 'r+') as f:       
        f.visititems(print_attrs)
    print(i)
    return result
         

if __name__ == '__main__':
 
    r = get_result()
   
    # print("number of AS:")
    # for i in r:
    #     print("\t{}:{}".format(i,len(r[i])))

    # f = number_of_frames()
    # print("total number of vidoes:{}".format(len(f)))
    # # for i in f :
    # #     print( "{}: {}".format(i,f[i]))

    # check_prediction_shape()

    
    # # evaluate(r)
