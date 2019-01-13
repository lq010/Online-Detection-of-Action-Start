import numpy as np
import os
from tqdm import tqdm
from IPython.core.debugger import Tracer

import tensorflow as tf

import keras.backend as K
import keras.callbacks as callbacks
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, concatenate, Lambda
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils

import matplotlib.pyplot as plt
plt.switch_backend('agg')   # allows code to run without a system DISPLAY

from models import c3d_model
from dataUtil import load_train_data, load_val_data
from src.batch_generator_without_followup import batch_generator_AS_nonAS_1_1, batch_generator_AS
from data import videoPaths as path


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, 
                                             simple_value=value), ])
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, latent_dim=100, length=16, width=112, height=112, channels=3, c3d_weights=None):
        self.image_path = path.VALIDATION_IMAGES_PATH
        self.n_classes = 20+1
        self.latent_dim = latent_dim
        self.length = length
        self.width = width
        self.height = height
        self.channels = channels
        self.c3d_weights = c3d_weights
        self.shape = (self.width, self.height, self.channels)
        self.optimizer = Adam(lr=1e-5, decay=0.00005)
       
        #init the c3d model
        self.c3d_model = c3d_model.get_model()
        if self.c3d_weights == None:
            raise Exception('weights is requited!')
        try:
            self.c3d_model.load_weights(self.c3d_weights)
        except OSError:
            print("the pretrained weights doesn't exist, please use <-h> to check usage")
            exit() 
        convLayers = ['conv1','conv2','conv3a','conv3b','conv4a','conv4b','conv5a','conv5b']
        for layer in convLayers:
            self.c3d_model.get_layer(layer).trainable = False
        self.add_outputs(1)
        
        #fixed c3d (conv1 - pool5), extracting real features
        self.fixed_c3d = Model(inputs=self.c3d_model.input,
                                outputs=self.c3d_model.get_layer('flatten_1').output)
        
        #discriminator,
        self.D = self.__discriminator()
        self.D.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        #generator    
        self.G = self.__generator()      
        # self.G.compile(loss='', optimizer=self.optimizer)
        self.GAN = self.__stacked_generator_discriminator()   
        self.GAN.compile(loss=self.loss_matching, optimizer=self.optimizer)
    
        self.c3d_model.summary()
        self.fixed_c3d.summary()
        self.G.summary()
        self.D.summary()
        self.GAN.summary()
        
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

        #output from generator
        gan_input_fake = Input(shape=(self.latent_dim,))
        fake_feature = self.G(gan_input_fake)

        #output from ConvNets (pool5)
        real_feature = Input((8192,))

        #FC6 and FC7 layers
        intermediate_layer_model = Model(inputs=self.D.input,
                                 outputs=self.D.get_layer('fc7').get_output_at(1))

        # set the discriminator weights to non-trainable                           
        intermediate_layer_model.trainable = False  
        out_fake = intermediate_layer_model(fake_feature)
        out_real = intermediate_layer_model(real_feature)

        matching = Lambda(lambda x: K.mean(x[0], axis=0) - K.mean(x[1], axis=0), 
                                 name='loss_matching')([out_fake, out_real])

        model_gan = Model(inputs=[gan_input_fake, real_feature],outputs=[matching])
        return model_gan

    def train(self, iterations=5000, batch_size=16, save_interval=2500, id=0):
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
                                                    batch_size=batch_size,
                                                    N_iterations=iterations,
                                                    N_classes=self.n_classes+1,
                                                    img_path=self.image_path)

        #logs
        result_dir = './results/gan_{}/'.format(id)
        weight_dir = os.path.join(result_dir, 'weights')
        if not os.path.isdir(weight_dir):
            os.makedirs(weight_dir)
        if not os.path.exists(weight_dir):
            os.makedirs(result_dir)
        log_dir = os.path.join(result_dir, 'logs')
        desp = os.path.join(result_dir,'desp.txt')
        with open(desp, 'w') as f:
            f.write("c3d weights: {}\n".format(self.c3d_weights))
            f.write("optimizer: {}\n".format(self.optimizer.get_config()))
        callback = callbacks.TensorBoard(log_dir=log_dir, batch_size=batch_size, histogram_freq=0, write_graph=True, write_images=True)
        callback.set_model(self.GAN)
        loss_names = ['disc_train_loss', 'disc_train_acc', 'gen_train_loss']
        for cnt in tqdm(range(iterations)):         
            '''discriminator'''
            #Sample random points in the latent space
            random_latent_vectors = np.random.standard_normal(size=(batch_size//2, self.latent_dim))
            #Decode them to fake images
            generated_features = self.G.predict(random_latent_vectors)
            
            #real images
            real_images, real_labels = next(disc_batch_generator)
            real_features = self.fixed_c3d.predict(real_images)
            combined_features = np.concatenate([generated_features, real_features])

            fake_labels = np.ones(batch_size//2)*(self.n_classes) #n_classes=21, 0=>BG, 1-20=>actions, 21=>fake

            fake_labels = np_utils.to_categorical(fake_labels, self.n_classes+1)

            labels = np.concatenate([fake_labels, real_labels])
        
            # Add random noise to the labels - important trick!
            # labels += 0.05 * np.random.random(labels.shape)
            d_loss, d_acc = self.D.train_on_batch(combined_features, labels)
        
            '''generator (via the gan model, where the discriminator weights are frozen)'''
            random_latent_vectors = np.random.standard_normal(size=(batch_size, self.latent_dim))

            real_AS_images, real_AS_labels = next(gen_batch_generator)
            real_AS_features = self.fixed_c3d.predict(real_AS_images)

            g_loss = self.GAN.train_on_batch([random_latent_vectors, real_AS_features],
                                                 [real_AS_labels]) #the labels are not used

            #tensorboard log                    
            logs = [d_loss, d_acc, g_loss]          
            write_log(callback, loss_names, logs, cnt)
            if cnt==save_interval:
                self.save_weights(weight_dir, cnt)
            tqdm.write('iteration: {}, [Discriminator :: d_loss: {}, d_acc: {}], [ Generator :: loss: {}]'
                                                                                .format(cnt, d_loss, d_acc, g_loss))
        self.save_weights(weight_dir, iterations)
        print('done')
        
    def save_weights(self, weight_dir, iteration):
        #save weights
        out_22 = 'c3d_TC_GAN_22_outputs_it{}.hdf5'.format(iteration)
        self.c3d_model.save_weights(os.path.join(weight_dir,out_22))
        self.GAN.save_weights(os.path.join(weight_dir,'GAN.hdf5'))
        #remove the last node from output layer
        # self.remove_last_output()
        # out_21 = 'c3d_TC_GAN_21_outputs_it{}.hdf5'.format(iteration)
        # self.c3d_model.save_weights(os.path.join(weight_dir,out_21))

    def add_outputs(self, n_new_outputs):
        #Increment the number of outputs
        new_n_classes = self.n_classes + n_new_outputs
        weights = self.c3d_model.get_layer('fc8').get_weights()
        #Adding new weights, weights will be 0 and the connections random
        shape = weights[0].shape[0]

        weights[1] = np.concatenate((weights[1], np.zeros(n_new_outputs)), axis=0)
        weights[0] = np.concatenate((weights[0], -0.0001 * np.random.random_sample((shape, n_new_outputs)) + 0.0001), axis=1)
        #Deleting the old output layer
        self.c3d_model.layers.pop()
        last_layer = self.c3d_model.get_layer('dropout_2').output
        #New output layer
        out = Dense(new_n_classes, activation='softmax', name='fc8')(last_layer)
        self.c3d_model = Model(input=self.c3d_model.input, output=out)
        #set weights to the layer
        self.c3d_model.get_layer('fc8').set_weights(weights)
        # print(weights[0])

    def remove_last_output(self):
        w = self.c3d_model.get_layer('fc8').get_weights()

        w[0] = np.delete(w[0], np.s_[-1], axis=1)
        w[1] = np.delete(w[1], np.s_[-1])

        #Deleting the old output layer
        self.c3d_model.layers.pop()
        last_layer = self.c3d_model.get_layer('dropout_2').output
        #New output layer
        out = Dense(self.n_classes, activation='softmax', name='fc8')(last_layer)
        self.c3d_model = Model(inputs=self.c3d_model.input, outputs=out)
        self.c3d_model.get_layer('fc8').set_weights(w)