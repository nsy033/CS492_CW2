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
from math import floor, ceil
import os
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Dropout, Input,  Reshape, multiply
from keras.layers import Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Nadam, Adam, SGD
from keras.datasets import mnist
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--real', type = float, default=0, help="ratio of real images")
parser.add_argument('--fake', type = float, default=0, help="ratio of fake images")
parser.add_argument('--size', type = int, default=60000, help="train size")

args = parser.parse_args()
perc_real = args.real
perc_fake = args.fake
num_train = args.size

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
img_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_test /= 255

num_fake = int(num_train*perc_fake)
num_real = int(num_train*perc_real)
# # Image shape information

# img_rows = X_train.shape[1]
# img_cols = X_train.shape[2]
# if len(X_train.shape) == 4:
#     channels = X_train.shape[3]
# else:
#     channels = 1

# img_shape = (img_rows, img_cols, channels)
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
generator.load_weights('../Q2/saved_model_weights/version7/generator_weights_29000.h5')

# generate 10 imgages for each digit:
num_imgs_each_digit = 6000
noise = np.random.normal(0, 1, (num_imgs_each_digit * 10, latent_dim))
tmp = []
for i in range(num_imgs_each_digit):
    for digit in range(10):
            tmp.append(digit)
sampled_labels = np.array(tmp)
gen_imgs = generator.predict([noise, sampled_labels])
# Rescale images 0 - 1
gen_imgs = 0.5 * gen_imgs + 0.5
# gen_imgs is a list of generated image with each pixel a value range form 0 to 1



# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
def classifier():
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=img_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    # model.add(Dense(128, activation=tf.nn.relu))
    # model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    return model

clf = classifier()
clf.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
clf.load_weights('../Q3/save_weight_classifier/version_4_ver2.h5')
preds = clf.predict(gen_imgs)
confidence= np.amax(preds, axis = 1)
idx = np.argpartition(confidence, -num_fake)[-num_fake:]
# idx = range(num_fake)
x_train = np.concatenate([X_train[:num_real], gen_imgs[idx]], axis = 0)
y_train = np.concatenate([y_train[:num_real], sampled_labels[idx]], axis = 0)
x_test = X_test

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

clf.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
clf.fit(x=x_train,y=y_train, epochs=10)

clf.evaluate(x_test, y_test)

# image_index = 4444
# plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
# pred = clf.predict(x_test[image_index].reshape(1, 28, 28, 1))
# print(clf.argmax())

existing_models = os.listdir('save_weight_classifier')
new_path_save_model = 'save_weight_classifier/train_with_high_confidence/' + str(100*perc_real) + '_' + str(100*perc_fake) + '.h5'
clf.save(new_path_save_model)

# model1 = tf.keras.models.load_model(new_path_save_model)
# model1.evaluate(x_test,y_test)
# image_index = 4444
# pred1 = model1.predict(x_test[image_index].reshape(1, 28, 28, 1))
# print(pred1.argmax())
