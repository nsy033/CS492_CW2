from __future__ import division, print_function
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
from numpy import std
from math import floor
import os
import keras
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Dropout, Input,  Reshape, multiply
from keras.layers import Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Nadam, Adam, SGD
from keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras import layers
import seaborn as sns


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
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
    #model.summary()

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
# The generator takes noise and the target label as input
# and generates the corresponding digit of that label
generator_path = '../Q2/saved_model_weights/version7/generator_weights_29000.h5'
generator.load_weights(generator_path)

# classifier model
#path_save_model = '../Q4/save_weight_classifier/normal/120_100.0_0.0.h5'
path_save_model = '../Q3/save_weight_classifier/version_4_ver2.h5'
model = tf.keras.models.load_model(path_save_model)
print(model.layers)
# GAN
num_imgs_each_digit = 1000
noise = np.random.normal(0, 1, (num_imgs_each_digit * 10, latent_dim))
tmp = []
for i in range(num_imgs_each_digit):
    for digit in range(10):
        tmp.append(digit)
sampled_labels = np.array(tmp)
gen_imgs = generator.predict([noise, sampled_labels])
# Rescale images 0 - 1
gen_imgs = 0.5 * gen_imgs + 0.5

# Random sampling 100 images
num_tsne = 1000
idx = np.random.choice(10000, num_tsne, replace=False)
print(idx)
x_test = X_test[idx]
x_gen = gen_imgs[idx]
y_test = y_test[idx]
y_gen = sampled_labels[idx]

# sess = tf.Session()
# Extract embedding
extractor = tf.keras.Model(inputs=model.inputs,
                        outputs=model.layers[-2].output)
features = extractor(tf.concat([x_test, x_gen], axis = 0))
# with sess.as_default(): 
    # sess.run(tf.global_variables_initializer())
# features = features.eval()
inp = model.inputs
print(features.shape)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
emb_tsne = tsne.fit_transform(features)
print(emb_tsne.shape)
plt.figure(figsize=(8, 8))
plt.scatter(x = emb_tsne[:num_tsne,0], y=emb_tsne[:num_tsne,1], c=y_test, marker = "o", label = "Real")
plt.scatter(x = emb_tsne[num_tsne:,0], y=emb_tsne[num_tsne:,1], c=y_gen, marker = "x", label = "Fake")
plt.legend(loc=4)
plt.savefig('scatter.png')
plt.close()

preds = model.predict(x_test)
confidence= np.amax(preds, axis = 1)
print("Real confidence", np.mean(confidence))
sns.set_style('darkgrid')
sns.distplot(confidence, kde=False, bins = 20, hist_kws = {'range': (0,1)})
plt.savefig('real.png')
plt.close()
preds = model.predict(x_gen)
confidence= np.amax(preds, axis = 1)
print("Fake confidence", np.mean(confidence))
sns.set_style('darkgrid')
sns.distplot(confidence, kde=False, bins = 20, hist_kws = {'range': (0,1)})
plt.savefig('fake.png')
plt.close()
