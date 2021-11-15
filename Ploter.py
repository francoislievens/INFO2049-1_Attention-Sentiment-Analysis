import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.core import series
import os
import time

SHOW_FIG = True
SAVE_FIG = False
MAX_EPOCH = 3

def plot_model(model_name, show_name):

    # Check if length evaluation data available:
    if os.path.exists('model/{}/lngts_logs.csv'.format(model_name)):
        plot_length_influence(model_name, show_name)

    # Chek if training data available:
    if os.path.exists('model/{}/train_logs.csv'.format(model_name)):
        plot_train_logs(model_name, show_name)



def plot_train_logs(model_name, show_name):

    train_logs = pd.read_csv('model/{}/train_logs.csv'.format(model_name), encoding='latin')
    test_logs = pd.read_csv('model/{}/test_logs.csv'.format(model_name), encoding='latin')
    data_logs = [test_logs, train_logs]
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

    __plot__(list_loss, list_accuracy, model_name, show_name, epoch_indexes)

def compute_mean_index(index, end):
    indexes = []
    start = 0
    for i in range(0, len(index)):
        indexes.append((start+index[i])/2)
        start = index[i]
    indexes.append((index[-1]+end)/2)
    return indexes

def __plot__(loss, acc, model_name, show_name, epoch_indexes):

    indexes = np.arange(0, len(loss[1]))
    mean_epoch_indexes = compute_mean_index(epoch_indexes, len(loss[1]))
    fig, ax = plt.subplots()

    ax.plot(indexes, loss[1], color="red",
            linewidth=0.3, label="training loss")

    ax.set_ylabel("Loss", color="red")
    ax.set_ylim([0, 0.3])
    #ax.set_ylim([0.6, 1])
    ax.plot(mean_epoch_indexes,
            loss[0], color="orange", linewidth=1, label="test loss")
    ax2 = ax.twinx()
    ax2.plot(indexes, acc[1], color="blue",
             linewidth=0.3, label="training accuracy")
    ax2.set_ylabel("Accuracy", color="blue")

    ax2.plot(mean_epoch_indexes,
            acc[0], color="green", linewidth=1, label="test accuracy")
    ax2.set_ylim([0.6, 0.95])
    [plt.axvline(x=epoch_indexes[i], color='black', linestyle='-', label='epoch {}'.format(i+2))
     for i in range(0, len(epoch_indexes))]
    ax.legend(loc='center right')
    ax2.legend(loc='lower right')
    plt.title('{}:\n Training curves'.format(show_name))
    if SAVE_FIG:
        plt.savefig('model/{}/train_perfs_{}.png'.format(model_name, model_name))
    if SHOW_FIG:
        plt.show()
    plt.close()

def plot_length_influence(model_name, show_name):

    step_size = 1
    windows_size = 400
    # Load data:
    df = pd.read_csv('model/{}/lngts_logs.csv'.format(model_name), sep=',').to_numpy()

    # sort the array
    df = df[np.argsort(df[:, 0])]

    acc = []
    loss = []
    size = []

    start_idx = 0
    end_idx = windows_size

    while end_idx < df.shape[0]:

        acc.append(np.mean(df[start_idx:end_idx, 2]))
        loss.append(np.mean(df[start_idx:end_idx, 1]))
        size.append(np.mean(df[start_idx:end_idx, 0]) / 5)      # 5 is the average number of characters in a work
        start_idx += step_size
        end_idx += step_size

    acc.append(np.mean(df[start_idx:, 2]))
    loss.append(np.mean(df[start_idx:, 1]))
    size.append(np.mean(df[start_idx:, 0]) / 5)

    fig, ax = plt.subplots()

    ax.plot(size, loss, label='MSE', color='red', linewidth=0.5)
    ax.set_ylabel('Loss', color='red')
    ax.set_ylim([0.1, 0.25])
    ax.legend(loc='center right')
    ax2 = ax.twinx()
    ax2.plot(size, acc, label='Accuracy', color='blue', linewidth=0.5)
    ax2.set_ylabel('Accuracy', color='blue')
    ax2.set_xlabel('Number of words')
    ax2.legend()


    plt.title('{}:\n Performances according to sequence length'.format(show_name))
    if SHOW_FIG:
        plt.show()
    if SAVE_FIG:
        plt.savefig('model/{}/{}_length_evaluator.png'.format(model_name, model_name))
    plt.close()



if __name__ == '__main__':

    model_list = [
        ['LSTM_w2v_a', 'LSTM with Word2Vec and Attention'],
        ['LSTM_glove_a', 'LSTM with Glove and Attention']
    ]

    for md in model_list:
        plot_model(model_name=md[0], show_name=md[1])