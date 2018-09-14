from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation,TimeDistributed,Permute,merge,Conv2D
from keras.layers.convolutional import ZeroPadding3D
from keras.regularizers import l2
from keras.models import Model


nb_classes = 20+1
dropout_ratio = 0.5
input_shape = (16,112,112,3)

#layer Conv1
Conv1 = Conv3D(64, (3, 3, 3), activation='relu',padding='same', name='conv1', input_shape=input_shape)
Pool1 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2),padding='valid', name='pool1')

#layer Conv2
Conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', name='conv2')
Pool2 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),padding='valid', name='pool2')

#layer Conv3
Conv3a = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3a')
Conv3b = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='conv3b')
Pool3 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')

#layer Conv4
Conv4a = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4a')
Conv4b = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv4b')
Pool4 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),padding='valid', name='pool4')

#layer Conv5
Conv5a = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5a')
Conv5b = Conv3D(512, (3, 3, 3), activation='relu', padding='same', name='conv5b')

zPadding5 = ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5')
Pool5 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2),padding='valid', name='pool5')
Flatten5 = Flatten()

#FC6
FC6 = Dense(4096, activation='relu', name='fc6')
Drop6 = Dropout(dropout_ratio, name='dropout_6')

#FC7
FC7 = Dense(4096, activation='relu', name='fc7')
Drop7 = Dropout(dropout_ratio ,name='dropout_7')


''''''
def conv_fc7(X_input):
    #layer Conv1
    x = Conv1(X_input)
    x = Pool1(x)

    #layer Conv2
    x = Conv2(x)
    x = Pool2(x)

    #layer Conv3
    x = Conv3a(x)
    x = Conv3b(x)
    x = Pool3(x)

    #layer Conv4
    x = Conv4a(x)
    x = Conv4b(x)
    x = Pool4(x)

    #layer Conv5
    x = Conv5a(x)
    x = Conv5b(x)
    x = zPadding5(x)
    x = Pool5(x)
    x = Flatten5(x)

    #FC6
    x = FC6(x)
    x = Drop6(x)

    #FC7
    x = FC7(x)
    x = Drop7(x)
    return x


def c3d_model(input_shape):
    """
    Implementation of the C3D
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    # define the input placeholder as a tensor with shape input_shape. 
    X_s_input = Input(input_shape, name = 'input_s')
    # X_f_input = Input(input_shape, name = 'input_f')

    x_s_7 = conv_fc7(X_s_input)
    # x_f_7 = conv_fc7(X_f_input)
 
    #FC8
    x_s = Dense(nb_classes, name = 'fc_8')(x_s_7)
    # x_s = Activation('softmax', name='softmax_8')(x_s)

    # x_s_7,x_f_7

    # model = Model(inputs = [X_s_input, X_f_input], outputs = [x_s, x_s_7, x_f_7])
    model = Model(inputs = X_s_input, outputs = x_s)
    return model
    
if __name__ == '__main__':
    X_s_input = Input(input_shape)
    model = c3d_model(input_shape)
    model.summary()
