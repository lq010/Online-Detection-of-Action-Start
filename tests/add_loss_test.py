import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.losses import mse
import numpy as np

# Some random training data
features = np.random.rand(100,20)
labels_1 = np.random.rand(100,4)
labels_2 = np.random.rand(100,1)

# Input layer, one hidden layer
input_layer = Input((20,))
dense_1 = Dense(128)(input_layer)

# Two outputs
output_1 = Dense(4)(dense_1)
output_2 = Dense(1)(dense_1)

# Two additional 'inputs' for the labels
label_layer_1 = Input((4,))
label_layer_2 = Input((1,))

# Instantiate model, pass label layers as inputs
model = Model(inputs=[input_layer, label_layer_1, label_layer_2], outputs=[output_1, output_2])

# Construct your custom loss as a tensor
loss = K.mean(mse(output_1, label_layer_2) * mse(output_2, label_layer_2))

# Add loss to model
model.add_loss(loss)

# Compile without specifying a loss
model.compile(optimizer='sgd')

model.fit([features, labels_1, labels_2], epochs=2)