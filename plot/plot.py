import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.core import series
import os
import time

MAX_EPOCH = 5


def compute_mean_index(index, end):
    indexes = []
    start = 0
    for i in range(0, len(index)):
        indexes.append((start+index[i])/2)
        start = index[i]
    indexes.append((index[-1]+end)/2)
    return indexes


def __plot__(loss, acc, title, epoch_indexes):

    indexes = np.arange(0, len(loss[1]))
    mean_epoch_indexes = compute_mean_index(epoch_indexes, len(loss[1]))
    fig, ax = plt.subplots()

    ax.plot(indexes, loss[1], color="red",
            linewidth=0.3, label="training loss")

    ax.set_ylabel("loss", color="red", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(indexes, acc[1], color="blue",
             linewidth=0.3, label="training accuracy")
    ax2.set_ylabel("accuracy", color="blue", fontsize=14)
    ax.plot(mean_epoch_indexes,
            loss[0], color="orange", linewidth=1, label="test loss")
    ax.plot(mean_epoch_indexes,
            acc[0], color="green", linewidth=1, label="test accuracy")
    [plt.axvline(x=epoch_indexes[i], color='black', linestyle='-', label='epoch {}'.format(i))
     for i in range(0, len(epoch_indexes))]
    plt.legend()
    plt.savefig(title+".png")


def plot_loss_accuracy(path):
    data_logs = [pd.read_csv(
        path+"/test_logs.csv", encoding='latin'), pd.read_csv(
        path+"/train_logs.csv", encoding='latin')]

    list_loss_logs_smooth = []
    list_acc_logs_smooth = []
    list_loss = []
    list_accuracy = []
    cpt = 0
    acc_loss = 0.0
    acc_acc = 0.0

    epoch = 0
    vec_loss_epoch = []
    vec_acc_epoch = []

    acc_loss = 0.0
    acc_acc = 0.0

    epoch_indexes = []
    # loop for test and train
    for j in range(0, 2):
        epoch = 0
        for i in range(len(data_logs[j]['loss'])):
            # if test
            if j == 0:
                if data_logs[j]['Epoch'][i] == epoch:
                    vec_loss_epoch.append(data_logs[j]['loss'][i])
                    vec_acc_epoch.append(data_logs[j]['accuracy'][i])
                    # take into account last epoch
                    if epoch == MAX_EPOCH-1:
                        list_loss_logs_smooth += [np.mean(vec_loss_epoch)]
                        list_acc_logs_smooth += [np.mean(vec_acc_epoch)]
                        vec_loss_epoch = []
                        vec_acc_epoch = []
                        break
                else:
                    list_loss_logs_smooth += [np.mean(vec_loss_epoch)]
                    list_acc_logs_smooth += [np.mean(vec_acc_epoch)]
                    epoch += 1
                    vec_loss_epoch = []
                    vec_acc_epoch = []
            # if train
            if j == 1:
                # smooth
                if cpt == 20:
                    if data_logs[j]['Epoch'][i] != epoch:
                        epoch += 1
                        epoch_indexes.append(i/20)

                    list_loss_logs_smooth.append(acc_loss/20)
                    list_acc_logs_smooth.append(acc_acc/20)
                    acc_loss = 0.0
                    acc_acc = 0.0
                    cpt = 0
                acc_loss += data_logs[j]['loss'][i]
                acc_acc += data_logs[j]['accuracy'][i]
                cpt += 1

        list_loss.append(list_loss_logs_smooth)

        list_accuracy.append(list_acc_logs_smooth)
        list_loss_logs_smooth = []
        list_acc_logs_smooth = []
        acc_loss = 0.0
        acc_acc = 0.0

    __plot__(list_loss, list_accuracy, os.path.split(path)[1], epoch_indexes)


if __name__ == '__main__':
    models = ["model/GRU_fasttext", "model/GRU_glove",
              "model/LSTM_fasttext", "model/LSTM_glove"]
    for model in models:
        plot_loss_accuracy(model)
