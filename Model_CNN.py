import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Flatten, Dense
from keras.layers import MaxPooling2D
from keras.models import *
import numpy as np
from sklearn.model_selection import train_test_split
from Evaluation import evaluation


def Model(X, Y, test_x, test_y):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.summary()
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10))
    model.compile(optimizer='adam', loss=tf.compat.v1.losses.sparse_softmax_cross_entropy, metrics=['accuracy'])
    model.fit(X, Y.astype('int'), epochs=10, validation_data=(test_x, test_y.astype('int')))
    pred = model.predict(test_x)
    return pred


def Model_CNN(Feature,Video,Target):
    Data= np.concatenate((Feature,Video),axis=0)
    Activation_function = round(Data.shape[0] * 0.75)
    train_data = Data[:Activation_function, :]
    train_target = Target[:Activation_function, :]
    test_data = Data[Activation_function:, :]
    test_target = Target[Activation_function:, :]
    IMG_SIZE = 32
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    pred = Model(Train_X, train_target, Test_X, test_target)
    Eval = evaluation(pred, test_target)
    return Eval


