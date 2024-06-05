import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import seaborn as sn
import pandas as pd

def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a


def Plot_Results():

    # for a in range(3):
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR','NPV', 'FDR', 'F1-Score',
             'MCC']
    Graph_Term = [0,1,2,3,4,5,6,7,8,9]
    Algorithm = ['TERMS', 'BWO-HCARDNet', 'CO-HCARDNet', 'MTBO-HCARDNet', 'OOA-HCARDNet', 'MRVOO-HCARDNet']
    Classifier = ['TERMS', 'LSTM', 'RNN', 'TCN', 'HCRDNet', 'MRVOO-HCRDNet']

    value = Eval[4, :, 4:]
    value[:, :-1] = value[:, :-1] * 100
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value[j, :])
    print('--------------------------------------------------Algorithm Comparison - ',
          '--------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
    print('---------------------------------------------------Classifier Comparison - ',
          '--------------------------------------------------')
    print(Table)

    Eval = np.load('Eval_all.npy', allow_pickle=True)
    BATCH = [1, 2, 3, 4, 5]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[1]):
                if Graph_Term[j] == 9:
                    Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]
                else:
                    Graph[k, l] = Eval[k, l, Graph_Term[j] + 4] * 100
        X = np.arange(5)
        plt.plot(BATCH, Graph[:, 0], '-.',color='b', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                 label="BWO-HCARDNet")
        plt.plot(BATCH, Graph[:, 1],'-.', color='#ef4026', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                 label="CO-HCARDNet")
        plt.plot(BATCH, Graph[:, 2],'-.', color='lime', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                 label="MTBO-HCARDNet")
        plt.plot(BATCH, Graph[:, 3],'-.', color='y', linewidth=3, marker='*', markerfacecolor='k', markersize=16,
                 label="OOA-HCARDNet")
        plt.plot(BATCH, Graph[:, 4], '-.',color='k', linewidth=3, marker='*', markerfacecolor='white', markersize=16,
                 label="MRVOO-HCARDNet")
        plt.xlabel('Activation Function')
        plt.xticks(X + 1, ('Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid'))
        # plt.xticks(BATCH, ('4', '8', '16', '32', '48'))
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/%s_line_1.png" % ( Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

        fig = plt.figure()
        # ax = plt.axes(projection="3d")
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Graph[:, 5], color='b', edgecolor='k', width=0.15, hatch="*", label="LSTM")
        ax.bar(X + 0.15, Graph[:, 6], color='#ef4026', edgecolor='k', width=0.15, hatch="*", label="RNN")
        ax.bar(X + 0.30, Graph[:, 7], color='lime', edgecolor='k', width=0.15, hatch='*', label="TCN")
        ax.bar(X + 0.45, Graph[:, 8], color='y', edgecolor='k', width=0.15, hatch="*", label="HCRDNet")
        ax.bar(X + 0.60, Graph[:, 9], color='k', edgecolor='w', width=0.15, hatch="o", label="MRVOO-HCARDNet")
        plt.xticks(X + 0.25, ('Linear', 'ReLU','Tanh',  'Softmax','Sigmoid'))
        plt.xlabel('Activation Function')
        plt.ylabel(Terms[Graph_Term[j]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
        path1 = "./Results/_Batch_%s_bar.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

def Confusion_matrix():
    # Confusion Matrix
    Eval = np.load('Eval_all.npy', allow_pickle=True)
    value = Eval[3, 4, :5]
    val = np.asarray([0, 1, 1])
    data = {'y_Actual': [val.ravel()],
            'y_Predicted': [np.asarray(val).ravel()]
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'], colnames=['Predicted'])
    value = value.astype('int')

    confusion_matrix.values[0, 0] = value[1]
    confusion_matrix.values[0, 1] = value[3]
    confusion_matrix.values[1, 0] = value[2]
    confusion_matrix.values[1, 1] = value[0]

    sn.heatmap(confusion_matrix, annot=True).set(title='Accuracy = ' + str(Eval[3, 4, 4] * 100)[:5] + '%')
    sn.plotting_context()
    path1 = './Results/Confusion.png'
    plt.savefig(path1)
    plt.show()

def plot_Fitness():

    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['BWO-HCARDNet', 'CO-HCARDNet', 'MTBO-HCARDNet', 'OOA-HCARDNet', 'MRVOO-HCARDNet']

    conv = np.load('Fitness.npy', allow_pickle=True)

    Value = np.zeros((conv.shape[0], 5))
    for j in range(conv.shape[0]):
        Value[j, 0] = np.min(conv[j, :])
        Value[j, 1] = np.max(conv[j, :])
        Value[j, 2] = np.mean(conv[j, :])
        Value[j, 3] = np.median(conv[j, :])
        Value[j, 4] = np.std(conv[j, :])

    Table = PrettyTable()
    Table.add_column("ALGORITHMS", Statistics)
    for j in range(len(Algorithm)):
        Table.add_column(Algorithm[j], Value[j, :])
    print('-------------------------------------------------- Statistical Analysis--------------------------------------------------')
    print(Table)

    iteration = np.arange(conv.shape[1])
    plt.plot(iteration, conv[0, :], color='#7ebd01', linewidth=3, marker='>', markerfacecolor='blue', markersize=12,
             label="BWO-HCARDNet")
    plt.plot(iteration, conv[1, :], color='#ef4026', linewidth=3, marker='>', markerfacecolor='red', markersize=12,
             label="CO-HCARDNet")
    plt.plot(iteration, conv[2, :], color='#12e193', linewidth=3, marker='>', markerfacecolor='green', markersize=12,
             label="MTBO-HCARDNet")
    plt.plot(iteration, conv[3, :], color='#ff0490', linewidth=3, marker='>', markerfacecolor='yellow',
             markersize=12,
             label="OOA-HCARDNet")
    plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='>', markerfacecolor='cyan', markersize=12,
             label="MRVOO-HCARDNet")
    plt.xlabel('No. of Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    path1 = "./Results/convergence.jpg"
    plt.savefig(path1)
    plt.show()

if __name__ == '__main__':
    # Plot_Results()
    # Confusion_matrix()
    plot_Fitness()
