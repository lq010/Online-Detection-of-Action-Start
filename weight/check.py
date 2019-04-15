import h5py
import os
import numpy as np
file = 'sports1M_weights_tf.h5'
file = os.path.join( os.pardir,'results/weights_c3d.h5')
f = h5py.File(file, 'r')
print(list(f.keys()))
print(list(f.items()))

# fc8 = f['fc8']
# print(fc8.name)

# read group

gf8 = f.get('dense_3').get('dense_3')
f8_keys = list(gf8.keys())
f8_items = list(gf8.items())
print('fc8 keys:')
print(f8_keys)
print('items in fc8:')
print(f8_items)

'''Keras expects the layer weights to be a list of length 2. First element is the kernel weights and the second is the bias.'''

bias = np.array(gf8.get('bias:0'))
kernel = np.array(gf8.get('kernel:0'))
print(bias)
print(kernel)
