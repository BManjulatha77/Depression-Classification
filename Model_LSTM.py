import numpy as np
from keras import Sequential
from keras.src.layers import Bidirectional, TimeDistributed, Dense
import cv2 as cv
from Evaluation import evaluation

def LSTM(train_data, train_target, test_data, sol):
    if sol is None:
        sol = [1, 1, 1, 1]
    model = Sequential()
    model.add(Bidirectional(LSTM(1, return_sequences=True), input_shape=(train_target.shape[1], 1)))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))  # activation= activation[int(sol[0])] activation='sigmoid'
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # optimizer = optimizer[int(sol[1])]  optimizer='adam'
    # # train BI-LSTM
    #     # fit model for one epoch on this sequence
    for i in range(train_data.shape[0]):
        d=cv.resize(train_data[i],[1, train_target.shape[1]]).ravel()
        data = d.reshape((1, d.shape[0], 1))
        tar = train_target[i].reshape((1, train_target.shape[1], 1))
        model.fit(data, tar, epochs= int(sol[2]), batch_size= int(sol[3]), verbose=2)
    # # evaluate BI-LSTM
    predict = np.zeros((test_data.shape[0], train_target.shape[1]))#.astype('int')
    for i in range(test_data.shape[0]):
        d = cv.resize(test_data[i], [1, train_target.shape[1]]).ravel()
        data = d.reshape((1, d.shape[0], 1))
        predict[i] = model.predict(data, verbose=0).ravel()
    return predict, model

def Model_LSTM(Feature,Video,Target, sol=None):
    Data = np.concatenate((Feature, Video), axis=0)
    Activation_function = round(Data.shape[0] * 0.75)
    train_data = Data[:Activation_function, :]
    train_target = Target[:Activation_function, :]
    test_data = Data[Activation_function:, :]
    test_target = Target[Activation_function:, :]
    if sol is None:
        sol = [1, 1, 1, 1]
    out, model = LSTM(train_data, train_target, test_data, sol)
    pred = np.asarray(out)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred.astype('int'), test_target)
    return np.asarray(Eval).ravel()



