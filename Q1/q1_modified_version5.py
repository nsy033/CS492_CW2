from __future__ import division, print_function
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, AveragePooling2D, MaxPooling2D, UpSampling2D, Input, Reshape, Lambda, Concatenate, Subtract, Reshape, multiply
from keras.layers import Embedding, ZeroPadding2D
from keras.regularizers import l2
from keras.layers import LeakyReLU
from keras import backend as K
from keras.optimizers import Nadam, Adam, SGD
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.callbacks import Callback, History
from keras.datasets import mnist


# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# Image shape information

img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)
num_classes = 10
latent_dim = 100
optimizer = Adam(0.0008)


def generator():
    model = Sequential()
    dropout = 0.4
    depth = 64+64+64+64
    dim = 7
    model.add(Dense(dim*dim*depth, input_dim=100))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Reshape((dim, dim, depth)))
    model.add(Dropout(dropout))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(1, 5, padding='same'))
    model.add(Activation('sigmoid'))
    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)
    return Model(noise, img)

def discriminator():
    model = Sequential()
    depth = 64
    dropout = 0.4
    model.add(Conv2D(depth*1, 5, strides=2, input_shape=(28, 28, 1),
                      padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))
    model.add(Conv2D(depth*2, 5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))
    model.add(Conv2D(depth*4, 5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))
    model.add(Conv2D(depth*8, 5, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)

discriminator = discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Build the generator

generator = generator()
# The generator takes noise and the target label as input
# and generates the corresponding digit of that label

noise = Input(shape=(latent_dim,))
img = generator(noise)

# For the combined model we will only train the generator
discriminator.trainable = False
# The discriminator takes generated image as input and determines validity
# and the label of that image
valid = discriminator(img)
# The combined model  (stacked generator and discriminator)
# Trains generator to fool discriminator
combined = Model(noise, valid)

optimizer = Adam(0.0004)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# print(img.shape.as_list())

# def save_imgs(epoch, parent_save_path, version):
#     r, c = 2, 5
#     noise = np.random.normal(0, 1, (r * c, 100))
#     sampled_labels = np.arange(0, 10).reshape(-1, 1)
#     gen_imgs = generator.predict([noise, sampled_labels])
#     # Rescale images 0 - 1
#     gen_imgs = 0.5 * gen_imgs + 0.5
#     fig, axs = plt.subplots(r, c)
#     cnt = 0
#     for i in range(r):
#         for j in range(c):
#             axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
#             axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
#             axs[i,j].axis('off')
#             cnt += 1
#     fig.savefig(parent_save_path + "/version_" + str(version) + "_epoch_" + str(epoch))
#     plt.close()


def save_imgs(parent_path,epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(parent_path + "/" + "mnist_" + str(epoch) + ".png")
    plt.close()
batch_size=100


# (X_train, _), (_, _) = mnist.load_data()

# X_train = (X_train.astype(np.float32) - 127.5) / 127.5
# X_train = np.expand_dims(X_train, axis=3)

# # Adversarial ground truths
# valid = np.ones((batch_size, 1))
# fake = np.zeros((batch_size, 1))

# Declaring empty lists to save the losses for plotting
d_loss_plot = []
g_loss_plot = []
acc_plot = []
discriminator_loss_on_real_image = []
discriminator_loss_on_fake_image = []
discriminator_acc_on_real_image = []
discriminator_acc_on_fake_image = []

# def plot_training_process(save_path=None):
#     plt.plot(acc_plot)
#     plt.title("D_acc")
#     if save_path!= None:
#         plt.savefig(save_path + "/D_acc.png")
#     plt.show()

#     plt.plot(list(range(1, len(discriminator_loss_on_real_image) + 1)), discriminator_loss_on_real_image, 'b')
#     plt.plot(list(range(1, len(discriminator_loss_on_fake_image) + 1)), discriminator_loss_on_fake_image, 'r')
#     plt.title("D_loss_on_real_and_fake_image")
#     if save_path!= None:
#         plt.savefig(save_path + "/D_loss_on_real_and_fake_image.png")
#     plt.show()

#     plt.plot(d_loss_plot)
#     plt.title("D_loss")
#     if save_path!= None:
#         plt.savefig(save_path + "/D_loss.png")
#     plt.show()

#     plt.plot(g_loss_plot)
#     plt.title("G_loss")
#     if save_path!= None:
#         plt.savefig(save_path + "/G_loss.png")
#     plt.show()

#     from numpy import save
#     save(save_path + '/d_loss.npy', d_loss_plot)
#     save(save_path + '/d_loss_on_real_image.npy', discriminator_loss_on_real_image)
#     save(save_path + '/d_loss_on_fake_image.npy', discriminator_loss_on_fake_image)
#     save(save_path + '/d_acc_on_real_image.npy', discriminator_acc_on_real_image)
#     save(save_path + '/d_acc_on_fake_image.npy', discriminator_acc_on_fake_image)
#     save(save_path + '/g_loss.npy', g_loss_plot)


def train(save_image_interval, save_model_interval,epochs, batch_size=128):
    os.makedirs('images', exist_ok=True)
    sub_images = os.listdir("images")
    new_version_sub_path = "images/version" + str(len(sub_images) + 1)
    os.makedirs(new_version_sub_path)

    os.makedirs('saved_model_weights', exist_ok=True)
    num_existing_version_saved = len(os.listdir("saved_model_weights"))
    new_save_model_path = "saved_model_weights/version" + str(num_existing_version_saved + 1)
    os.makedirs(new_save_model_path)


    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = X_train.astype('float32') / 255.0
    X_train = X_train.reshape((X_train.shape[0],)+(28, 28, 1))

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    list_G_loss = []
    list_D_loss = []
    list_D_acc = []
    list_D_real_image_loss = []
    list_D_fake_image_loss = []

    for epoch in range(epochs):
        # Training the Discriminator
        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, 100))
        # Generate a half batch of new images
        fake_imgs = generator.predict(noise)

        # Train the discriminator
        D_loss_real = discriminator.train_on_batch(real_imgs, valid)
        D_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        D_real_image_loss, D_real_image_acc = D_loss_real
        D_fake_image_loss, D_fake_image_acc = D_loss_fake


        D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)

        # Training the Generator
        # Condition on labels
        sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
        # Train the generator
        g_loss = combined.train_on_batch(noise, valid)
        list_D_acc.append(100*D_loss[1])
        list_D_loss.append(D_loss[0])
        list_D_real_image_loss.append(D_real_image_loss)
        list_D_fake_image_loss.append(D_fake_image_loss)
        list_G_loss.append(g_loss)

        # Saving generated image samples at every sample interval
        if epoch % save_image_interval== 0:
            print("%d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, epochs, D_loss[0], 100 * D_loss[1], g_loss))
            save_imgs(new_version_sub_path, epoch)
        if epoch % save_model_interval == 0:
            print("save model of epoch ", epoch)
            generator.save_weights(new_save_model_path + "/generator_weights_" + str(epoch) + ".h5")
            discriminator.save_weights(new_save_model_path + "/discriminator_weights_" + str(epoch) + ".h5")
            combined.save_weights(new_save_model_path + "/combined_weights_"+ str(epoch) + ".h5")
    
    path_save_plot_training = "plot_history_training/version" + str(len(sub_images) + 1)
    os.makedirs(path_save_plot_training)
    plot_training_process(list_D_acc, list_D_real_image_loss, list_D_fake_image_loss ,list_G_loss, path_save_plot_training +"/plot_training_")


def plot_training_process(D_acc_list, D_real_image_loss_list, D_fake_image_loss_list, G_loss_list, save_path=None):
    plt.plot(D_acc_list)
    plt.title("D_acc")
    if save_path!= None:
        plt.savefig(save_path + "D_acc.png")
    plt.show()

    plt.plot(list(range(1, len(D_real_image_loss_list) + 1)), D_real_image_loss_list, 'b')
    plt.plot(list(range(1, len(D_fake_image_loss_list) + 1)), D_fake_image_loss_list, 'r')
    plt.title("D_loss_on_real_and_fake_image")
    if save_path!= None:
        plt.savefig(save_path + "D_loss_on_real_and_fake_image.png")
    plt.show()

    plt.plot(G_loss_list)
    plt.title("G_loss")
    if save_path!= None:
        plt.savefig(save_path + "G_loss.png")
    plt.show()
t0 = time.time()
train(save_image_interval=200, save_model_interval=1000, epochs = 30000)
t1 = time.time()
print("elapsed time: ", t1-t0)



