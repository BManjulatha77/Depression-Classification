import numpy as np
from Evaluation import evaluation
from Neural_Network import NeuralNetwork

def Model_DNN(Feature,Video,Target):
    Data = np.concatenate((Feature, Video), axis=0)
    Activation_function = round(Data.shape[0] * 0.75)
    data = Data[:Activation_function, :]
    labels = Target[:Activation_function, :]
    test_data = Data[Activation_function:, :]
    test_target = Target[Activation_function:, :]
    simple_network = NeuralNetwork(no_of_in_nodes=data.shape[1],
                                   no_of_out_nodes=labels.shape[1],
                                   no_of_hidden_nodes=200,
                                   learning_rate=0.1,
                                   bias=None)

    for _ in range(20):
        for i in range(len(data)):
            simple_network.train(data[i, :], labels[i])

    pred = simple_network.run(test_data)
    predict = np.zeros((pred.shape[1])).astype('int')
    for i in range(pred.shape[1]):
        if pred[0, i] > 0.5:
            predict[i] = 1
        else:
            predict[i] = 0
    Pred = np.round(predict)
    Eval = evaluation(Pred.reshape(-1, 1), test_target)
    return np.asarray(Eval).ravel()


