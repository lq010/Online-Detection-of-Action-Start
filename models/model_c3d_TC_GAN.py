# -*- coding: utf-8 -*-
""" Simple implementation of Generative Adversarial Neural Network """

import numpy as np
import keras.backend as K

from IPython.core.debugger import Tracer

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, concatenate, Lambda
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
plt.switch_backend('agg')   # allows code to run without a system DISPLAY

import c3d_model
from dataUtil import load_train_data, load_val_data
from src.batch_generator_without_followup import batch_generator_AS_nonAS_1_1, batch_generator_AS
from data import videoPaths as path

class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, latent_dim=100, length=16, width=112, height=112, channels=3, optimizer=None, c3d_weight=None):
        self.image_path = path.VALIDATION_IMAGES_PATH
        self.n_classes = 20+1
        self.latent_dim = latent_dim
        self.length = length
        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)
        
        self.optimizer = optimizer
        #init the c3d model
        self.c3d_model = c3d_model.get_model()
        if c3d_weight == None:
            raise Exception('weights is requited!')
        self.c3d_model.load_weights(c3d_weight)
        convLayers = ['conv1','conv2','conv3a','conv3b','conv4a','conv4b','conv5a','conv5b']
        for layer in convLayers:
            self.c3d_model.get_layer(layer).trainable = False
        self.add_outputs(1)
        #fixed c3d (conv1 - pool5)
        self.fixed_c3d = Model(inputs=self.c3d_model.input,
                                outputs=self.c3d_model.get_layer('flatten_1').output)
        #generator    
        self.G = self.__generator()
        # self.G.compile(loss='', optimizer=self.optimizer)
        #discriminator
        self.D = self.__discriminator()
        self.D.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.GAN = self.__stacked_generator_discriminator()
        self.GAN.compile(loss=self.loss_matching, optimizer=self.optimizer)

    def loss_matching(self, y_true, y_pred):
        loss = K.mean(K.abs(y_pred))
        return loss

    def __generator(self):
        """ Declare generator """
        #FC1
        generator_input = Input(shape=(self.latent_dim,))
        x = Dense(8192, name='fc1')(generator_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        #FC2
        x = Dense(8192, name='fc2')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        model_gen = Model(generator_input, x)

        return model_gen

    def __discriminator(self):
        """ Declare discriminator """
        FC6 = self.c3d_model.get_layer('fc6')
        FC7 = self.c3d_model.get_layer('fc7')
        FC8 = self.c3d_model.get_layer('fc8')

        discriminator_input = Input(shape=(8192,))
        x = FC6(discriminator_input)
        x = LeakyReLU()(x)
        x = FC7(x)
        x = LeakyReLU()(x)
        x = Dropout(0.4)(x)
        x = FC8(x)
        model_disc = Model(discriminator_input, x)
        return model_disc

    def __stacked_generator_discriminator(self):

        self.D.trainable = False
        #output from generator
        gan_input_fake = Input(shape=(self.latent_dim,))
        fake_feature = self.G(gan_input_fake)

        #putput from ConvNets (pool5)
        # gan_input_real = self.fixed_c3d.input
        # real_feature = self.fixed_c3d.output
        real_feature = Input((8192,))

        #FC6 and FC7 layers
        intermediate_layer_model = Model(inputs=self.D.input,
                                 outputs=self.D.get_layer('fc7').output)       
        out_fake = intermediate_layer_model()(fake_feature)
        out_real = intermediate_layer_model()(real_feature)

        matching = Lambda(lambda x: K.mean(x[1], axis=0) - K.mean(x[2], axis=0), 
                                 name='loss_matching')([out_fake, out_real])

        model_gan = Model(inputs=[gan_input_fake, real_feature],outputs=[matching])
        return model_gan

    def train(self, X_train, iterations=5000, batch_size=16, save_interval=100):
        #load the real data
        train_AS_windows, train_A_windows, train_BG_windows = load_train_data() # load train data
        val_AS_windows, val_A_windows, val_BG_windows = load_val_data() # load val data

        #real image generator for discriminator (AS + non AS)
        disc_batch_generator = batch_generator_AS_nonAS_1_1(AS_windows=train_AS_windows,
                                                                A_windows=train_A_windows,
                                                                BG_windows=train_BG_windows,
                                                                windows_length=self.length,
                                                                batch_size=batch_size//2,
                                                                N_iterations=iterations,
                                                                N_classes=self.n_classes+1,
                                                            img_path=self.image_path)
        #real image generator for generator (AS only)
        gen_batch_generator = batch_generator_AS(AS_windows=train_AS_windows,
                                                    windows_length=self.length,
                                                    batch_size=batch_size//2,
                                                    N_iterations=iterations,
                                                    N_classes=self.n_classes+1,
                                                    img_path=self.image_path)
        for cnt in range(iterations):
            '''discriminator'''
            #Sample random points in the latent space
            random_latent_vectors = np.random.normal(size=(batch_size//2, self.latent_dim))
            #Decode them to fake images
            generated_features = self.G.predict(random_latent_vectors)
            
            #real images
            real_images, real_labels = next(disc_batch_generator)
            real_features = self.fixed_c3d.predict(real_images)
            combined_features = np.concatenate([generated_features, real_features])

            fake_labels = np.ones(batch_size//2)*(self.n_classes) #n_classes=21, 0=>BG, 1-20=>actions, 21=>fake

            labels = np.concatenate([fake_labels, real_labels])
            # Add random noise to the labels - important trick!
            # labels += 0.05 * np.random.random(labels.shape)
            d_loss = self.D.train_on_batch(combined_features, labels)

            '''generator (via the gan model, where the discriminator weights are frozen)'''
            random_latent_vectors = np.random.normal(size=(batch_size, self.latent_dim))

            real_AS_images, real_AS_labels = next(gen_batch_generator)
            real_AS_features = self.fixed_c3d.predict(real_AS_images)
            
            g_loss = self.GAN.train_on_batch([random_latent_vectors, real_AS_features],
                                                 [real_AS_labels, real_AS_labels]) #the labels are not used


            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

            if cnt % save_interval == 0:
                self.plot_images(save2file=True, step=cnt)


    def plot_images(self, save2file=False, samples=16, step=0):
        ''' Plot and generated images '''
        filename = "./images/mnist_%d.png" % step
        noise = np.random.normal(0, 1, (samples, 100))

        images = self.G.predict(noise)

        plt.figure(figsize=(10, 10))

        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.height, self.width])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

    def add_outputs(self, n_new_outputs):
        #Increment the number of outputs
        self.n_classes += n_new_outputs
        weights = self.c3d_model.get_layer('fc8').get_weights()
        #Adding new weights, weights will be 0 and the connections random
        shape = weights[0].shape[0]
        weights[1] = np.concatenate((weights[1], np.zeros(n_new_outputs)), axis=0)
        weights[0] = np.concatenate((weights[0], -0.0001 * np.random.random_sample((shape, n_new_outputs)) + 0.0001), axis=1)
        #Deleting the old output layer
        self.c3d_model.layers.pop()
        last_layer = self.c3d_model.get_layer('dropout_7').output
        #New output layer
        out = Dense(self.n_classes, activation='softmax', name='fc8')(last_layer)
        self.model = Model(input=self.c3d_model.input, output=out)
        #set weights to the layer
        self.c3d_model.get_layer('fc8').set_weights(weights)
        # print(weights[0])

if __name__ == '__main__':
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)


    gan = GAN()
    gan.train(X_train)