import torch
from CreateVocab import *
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from Train import train
import matplotlib.pyplot as plt
import pandas as pd

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

    ax.plot(size, acc, label='Accuracy', color='blue', linewidth=0.5)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Number of words')
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(size, loss, label='MSE', color='red', linewidth=0.5)
    ax2.set_ylabel('Loss')
    ax2.legend(loc='center right')
    plt.title('{}:\n Performances according to sequence length'.format(show_name))
    plt.show()
    plt.savefig('model/{}/{}_length_evaluator.png'.format(model_name, model_name))
    plt.close()


if __name__ == '__main__':

    model_list = [
        ['LSTM_w2v_a', 'LSTM with Word2Vec and Attention'],
        ['LSTM_glove_a', 'LSTM with Glove and Attention']
    ]
    plot_length_influence(model_name='LSTM_w2v_a', show_name='LSTM with Word2Vec and Attention')

