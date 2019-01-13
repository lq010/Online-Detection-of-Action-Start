import c3d_model
import numpy as np
from keras.layers import Input, Dense
from keras.models import Sequential, Model

c3d_model = c3d_model.get_model(nb_classes=22)

weights_path = '/home/lq/Documents/Thesis/Thesis/results/gan_1/weights/c3d_TC_GAN_22_outputs_it5000.hdf5'
save_path = weights_path.replace('_22_outputs_', '_21_outputs_')
n_classes =21

c3d_model.load_weights(weights_path)

w = c3d_model.get_layer('fc8').get_weights()
print(w)
w[0] = np.delete(w[0], np.s_[-1], axis=1)
w[1] = np.delete(w[1], np.s_[-1])

#Deleting the old output layer
c3d_model.layers.pop()
last_layer = c3d_model.get_layer('dropout_2').output
#New output layer
out = Dense(n_classes, activation='softmax', name='fc8')(last_layer)
c3d_model = Model(inputs=c3d_model.input, outputs=out)
c3d_model.get_layer('fc8').set_weights(w)
print(c3d_model.get_layer('fc8').get_weights())
c3d_model.summary()

c3d_model.save_weights(save_path)