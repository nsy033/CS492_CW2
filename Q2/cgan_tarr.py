from __future__ import division, print_function
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean, cov
from numpy import exp
from numpy import std
from math import floor
import os
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Dropout, Input,  Reshape, multiply
from keras.layers import Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Nadam, Adam, SGD
from keras.datasets import mnist
import tensorflow as tf
import argparse
from scipy.linalg import sqrtm
import utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
if len(X_train.shape) == 4:
    channels = X_train.shape[3]
else:
    channels = 1

img_shape = (img_rows, img_cols, channels)
num_classes = 10
latent_dim = 100
optimizer = Adam(0.0002, 0.5)
'''
def generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    #model.summary()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))
    model_input = multiply([noise, label_embedding])
    img = model(model_input)
    return Model([noise, label], img)
'''

def generator():
    vbm=2
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8, virtual_batch_size=vbm))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8, virtual_batch_size=vbm))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8, virtual_batch_size=vbm))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))
    model_input = multiply([noise, label_embedding])
    img = model(model_input)
    return Model([noise, label], img)

def discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=np.prod(img_shape)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    #model.summary()

    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])
    validity = model(model_input)
    return Model([img, label], validity)

# Build the generator

generator = generator()
discriminator = discriminator()
# The generator takes noise and the target label as input
# and generates the corresponding digit of that label

# generator_path = '../Q2/saved_model_weights/version1/generator_weights_99000.h5'
generator_path = '../Q2/saved_model_weights/version10/generator_weights_29000.h5'
discriminator_path = '../Q2/saved_model_weights/version7/discriminator_weights_29000.h5'
generator.load_weights(generator_path)
discriminator.load_weights(discriminator_path)

batch_size=10000
# idx = np.random.randint(0, X_test.shape[0], batch_size)
label = []
for i in range(10):
    label+=[i]*1000
print(len(label))
label = np.array(label)
path_save_model = 'save_weight_classifier/version_4_ver2.h5'

model = tf.keras.models.load_model(path_save_model)

noise = np.random.normal(0, 1, (batch_size, 100))
gen_imgs = generator.predict([noise, label])
gen_imgs = 0.5 * gen_imgs + 0.5
print(gen_imgs.shape)
pred = np.argmax(model.predict(gen_imgs), axis=1)
print(pred.shape)
pred_correct_idx = np.where((pred-label)==0)
print(pred_correct_idx[0].shape)
# pred = model.predict(gen_imgs)
# res = np.argmax(pred, axis=1)-y_test
# print(1-np.count_nonzero(res)/10000)



# discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])


# noise = Input(shape=(latent_dim,))
# label = Input(shape=(1,))
# img = generator([noise, label])

# discriminator.trainable = False
# # The discriminator takes generated image as input and determines validity
# # and the label of that image
# valid = discriminator([img, label])
# # The combined model  (stacked generator and discriminator)
# # Trains generator to fool discriminator
# combined = Model([noise, label], valid)
# combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

# valid = np.ones((batch_size, 1))
# fake = np.zeros((batch_size, 1))
# # Train the discriminator

# noise = np.random.normal(0, 1, (batch_size, 100))
# # Generate a half batch of new images
# gen_imgs = generator.predict([noise, labels])
# d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
# d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
# d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

# # additional storage
# D_real_image_loss, D_real_image_acc = d_loss_real
# D_fake_image_loss, D_fake_image_acc = d_loss_fake

# print(d_loss[1])