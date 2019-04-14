import numpy as np

from keras.models import Model
from keras.layers import Input

import keras.backend as K
from keras.engine.topology import Layer
from keras.layers.core import  Dense

from keras import objectives

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


class CustomRegularization(Layer):
    def __init__(self, **kwargs):
        super(CustomRegularization, self).__init__(**kwargs)

    def call(self ,x ,mask=None):
        ld=x[0]
        rd=x[1]
        bce = objectives.binary_crossentropy(ld, rd)
        loss2 = K.sum(bce)
        self.add_loss(loss2,x)
        #you can output whatever you need, just update output_shape adequately
        #But this is probably useful
        print("####"+str(bce))
        return bce

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0],1)

input_size= 10
output_dim = 1
x1 = Input(shape=(input_size,))
ld = Dense(128, activation='relu')(x1)
out1 = Dense(output_dim, activation='sigmoid')(ld)

x2 = Input(shape=(input_size,))
rd = Dense(128, activation='relu')(x2)
out2 = Dense(output_dim, activation='sigmoid')(rd)
import tensorflow as tf
import keras.backend as K

cr = CustomRegularization()([ld,rd])


m = Model( [x1,x2], [out1,out2,cr])
m.compile( loss=[K.binary_crossentropy,K.binary_crossentropy,zero_loss], optimizer="adam")
m.summary()
nb_examples = 32
pre = m.predict( [np.random.randn(nb_examples,input_size),np.random.randn(nb_examples,input_size)] )
result = m.fit(  [np.random.randn(nb_examples,input_size),np.random.randn(nb_examples,input_size)], [np.random.randn(nb_examples,output_dim),np.random.randn(nb_examples,output_dim), np.random.randn(nb_examples,1) ]  )
print(pre)

