# Custom loss layer
from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation,TimeDistributed,Permute,merge,Conv2D, Layer
from keras.layers.convolutional import ZeroPadding3D
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K
import tensorflow as tf

class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def myl2_loss(self, x_s, x_f):
        x = x_s - x_f
        return tf.nn.l2_loss(x)
    def categorical_crossentropy(self, y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred)

    def call(self, inputs, weight = [1, 0.1]):
        x_s = inputs[0]
        x_f = inputs[1]
        y_true = inputs[2]
        y_pred = inputs[3]

        loss = self.myl2_loss(x_s, x_f) * weight[0] + self.categorical_crossentropy(y_true,y_pred) * weight[1]
        self.add_loss(loss, inputs=inputs)
        # we don't use this output, but it has to have the correct shape:
        return K.ones_like(x_s)

def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

import numpy as np
true = np.zeros((2,5,10,10,3))
pred = np.ones((2,5,10,10,3))
true = tf.convert_to_tensor(true, dtype='float32')
pred = tf.convert_to_tensor(pred, dtype='float32')
with tf.Session() as sess:
    a = categorical_crossentropy(true,pred)
    print (a.eval())