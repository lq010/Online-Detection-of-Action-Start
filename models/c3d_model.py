from keras import Input
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import  Convolution3D, MaxPooling3D, ZeroPadding3D


def get_model(s = False, backend = 'tf'):
    input_shape=(16, 112, 112, 3) # l, h, w, c
    X_input = Input(input_shape)

    nb_classes = 20+1
    #1st layer group 
    x = Convolution3D(64, (3, 3, 3), activation='relu',
                            padding='same', name='conv1',
                            input_shape=input_shape)(X_input)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1')(x)
    # 2nd layer group
    x = Convolution3D(128, (3, 3, 3), activation='relu',
                            padding='same', name='conv2')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2')(x)
    # 3rd layer group
    x = Convolution3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3a')(x)
    x = Convolution3D(256, (3, 3, 3), activation='relu',
                            padding='same', name='conv3b')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3')(x)
    # 4th layer group
    x = Convolution3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv4a')(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv4b')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4')(x)
    # 5th layer group
    x = Convolution3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv5a')(x)
    x = Convolution3D(512, (3, 3, 3), activation='relu',
                            padding='same', name='conv5b')(x)
    x = ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5')(x)
    x = Flatten()(x)
    # FC layers group
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(.5)(x)
    x = Dense(nb_classes, activation='softmax', name='fc8')(x)

    model = Model(X_input, x)
    return model


if __name__ == '__main__':
    model = get_model()
    model.summary()