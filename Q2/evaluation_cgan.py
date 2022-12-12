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

# Parse argument
parser = argparse.ArgumentParser()
    
# Argument lists
parser.add_argument('--all', action ='store_true', default=False, help="turn on for inception score")
parser.add_argument('--ins', action ='store_true', default=False, help="turn on for inception score")
parser.add_argument('--fid', action ='store_true', default=False, help="turn on for inception distance")
parser.add_argument('--mmd', action ='store_true', default=False, help="turn on for inception distance")
parser.add_argument('--ms', action ='store_true', default=False, help="turn on for inception distance")
parser.add_argument('--nn', action ='store_true', default=False, help="turn on for inception distance")

# Read the arguments
args = parser.parse_args()
all_ = args.all
ins = args.ins
fid = args.fid
mmd = args.mmd
ms = args.ms
nn = args.nn
# Image shape information

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
    model.summary()

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
'''

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
# The generator takes noise and the target label as input
# and generates the corresponding digit of that label

# generator_path = '../Q2/saved_model_weights/version1/generator_weights_99000.h5'
generator_path = '../Q2/saved_model_weights/version10/generator_weights_29000.h5'
generator.load_weights(generator_path)

# classifier model
path_save_model = 'save_weight_classifier/version_4_ver2.h5'
model = tf.keras.models.load_model(path_save_model)

if fid or all_:
    fid = utils.calculate_inception_distance_generated_img(generator, model, latent_dim, 1000, X_test)
    print("FID score is ", fid)
if ins or all_:
    is_avg, is_std = utils.calculate_inception_score_generated_img(generator, model, latent_dim, 1000)
    print("IS score is ", is_avg, ", ", is_std)
if mmd or all_:
    mmd = utils.calculate_mmd_generated_img(generator, model, latent_dim, 100, X_test)
    print('MMD score', mmd)
if ms or all_:
    ms = utils.calculate_mode_score_generated_img(generator, model, latent_dim, 1000, X_test)
    print('Mode score', ms)
if nn or all_:
    ms = utils.calculate_nn_score_generated_img(generator, model, latent_dim, 100, X_test)
    print('1NN score', ms)

