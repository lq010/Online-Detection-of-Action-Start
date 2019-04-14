from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras import backend as K
import numpy as np

m = Sequential([
    Dropout(rate=0.5, input_shape=(1,), noise_shape=(1,1),seed = 1)
])
m.summary()

input_a = Input(shape=(1,))
input_b = Input(shape=(1,))

processed_a = m(input_a)
processed_b = m(input_b)

def l1_distance((x1, x2)):
    return K.sum(K.abs(x1-x2), axis=1)

c = Lambda(l1_distance, output_shape=(1,))([processed_a, processed_b])
s = Model([input_a, input_b], c)
s.compile(optimizer='sgd', loss='mse')
s.summary()

x0 = np.array([1])
x1 = np.array([1])
x  = [x0,x1]
y  = np.array([0])

s.fit(x, y, verbose=1, epochs=10)

print(s.evaluate(x,y), s.predict(x))