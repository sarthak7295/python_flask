import numpy as np
# import pandas as pd
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2
from keras.layers import Activation
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2


def get_mnist_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(units=120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(10, W_regularizer=l2(0.01)))
    model.add(Activation('softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.load_weights('mnist_fin.h5')
    return model

def get_result(image_name,my_model):
    img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    res = my_model.predict(img.reshape(1, 32, 32, 1))
    # print(np.argmax(my_model.predict(img.reshape(1, 32, 32, 1))))
    return res











    # img = resize_img(img)
    # my_model.predict(img)
# m = get_mnist_model()
# get_result('output.png',m)

# img = cv2.imread('output.png',cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
# plt.title('Example %d. Label: %d' % (1, 1))
# plt.imshow(img, cmap=plt.cm.gray_r)
# plt.show()
#
# my_model = get_mnist_model()
# # my_model.load_weights('mnist_fin.h5')
#
# (train_x, train_y) , (test_x, test_y) = mnist.load_data()
# train_x = train_x.reshape(-1, 28, 28, 1)
# test_x = test_x.reshape(-1, 28, 28, 1)
#
# train_y = to_categorical(train_y)
# test_y =  to_categorical(test_y)
#
#
# train_x= np.pad(train_x, ((0,0),(2,2),(2,2),(0,0)), 'constant')
# test_x= np.pad(test_x, ((0,0),(2,2),(2,2),(0,0)), 'constant')
#
# my_model.summary()
# score = my_model.evaluate(test_x, test_y, verbose=0)
# print("%s: %.2f%%" % (my_model.metrics_names[1], score[1]*100))
# print(np.argmax(my_model.predict(img.reshape(1,32,32,1))))
#
#
# def display_image(position):
#     image = test_x[position].squeeze()
#     plt.title('Example %d. Label: %d' % (position, np.argmax(test_y[position])))
#     plt.imshow(image, cmap=plt.cm.gray_r)
#     plt.show()


# display_image(50)