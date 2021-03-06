from keras import Input
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import  Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.regularizers import l2

def get_model(nb_classes=21):
    input_shape=(16, 112, 112, 3) # l, h, w, c
    X_input = Input(input_shape)

    
    #1st layer group 
    x = Convolution3D(64, (3, 3, 3), activation='relu',
                            padding='same', name='conv1',
                            input_shape=input_shape)(X_input)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1')(x)
    # 2nd layer group
    x = Convolution3D(128, (3, 3, 3), activation='relu',
                            padding='same',  name='conv2')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2')(x)
    # 3rd layer group
    x = Convolution3D(256, (3, 3, 3), activation='relu',
                            padding='same',  name='conv3a')(x)
    x = Convolution3D(256, (3, 3, 3), activation='relu',
                            padding='same',  name='conv3b')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3')(x)
    # 4th layer group
    x = Convolution3D(512, (3, 3, 3), activation='relu',
                            padding='same',  name='conv4a')(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu',
                            padding='same',  name='conv4b')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4')(x)
    # 5th layer group
    x = Convolution3D(512, (3, 3, 3), activation='relu',
                            padding='same',  name='conv5a')(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu',
                            padding='same',  name='conv5b')(x)
    x = ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5')(x)
    x = Flatten(name='flatten_1')(x)
    # FC layers group
    x = Dense(4096, name='fc6')(x)
    x = Activation('relu', name='fc6_relu')(x)
    x = Dropout(.5)(x)
    x = Dense(4096, name='fc7')(x)
    x = Activation('relu', name='fc7_relu')(x)
    x = Dropout(.5)(x)
    x = Dense(nb_classes, activation='softmax', name='fc8')(x)

    model = Model(X_input, x)
    return model


if __name__ == '__main__':
    model = get_model()
    model.summary()
    # from keras.utils import plot_model
    # plot_model(model, to_file='model_v2.png', show_shapes=True)