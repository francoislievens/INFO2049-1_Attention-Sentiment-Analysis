"""
This file contain the implementation that we made
in order to generate all data array and plots for
evaluating model's performances and the report
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

SHOW_FIG = False
SAVE_FIG = True
MAX_EPOCH = 3
LAST_EPOCH_IDX = 2


def plot_model(model_name, show_name):

    # # Check if length evaluation data available:
    if os.path.exists('model/{}/lngts_logs.csv'.format(model_name)):
        plot_length_influence(model_name, show_name)

    # Chek if training data available:
    if os.path.exists('model/{}/train_logs.csv'.format(model_name)):
        plot_train_logs(model_name, show_name)


def plot_train_logs(model_name, show_name):

    train_logs = pd.read_csv(
        'model/{}/train_logs.csv'.format(model_name), encoding='latin')
    test_logs = pd.read_csv(
        'model/{}/test_logs.csv'.format(model_name), encoding='latin')
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
                if i == len(data_logs[j]['loss'])-1:
                    list_loss_logs_smooth += [np.mean(vec_loss_epoch)]
                    list_acc_logs_smooth += [np.mean(vec_acc_epoch)]
                    vec_loss_epoch = []
                    vec_acc_epoch = []
                    break
                if data_logs[j]['Epoch'][i] == epoch:

                    vec_loss_epoch.append(data_logs[j]['loss'][i])
                    vec_acc_epoch.append(data_logs[j]['accuracy'][i])
                    # take into account last epoch
                elif epoch == MAX_EPOCH-1:
                    vec_loss_epoch.append(data_logs[j]['loss'][i])
                    vec_acc_epoch.append(data_logs[j]['accuracy'][i])

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
        plt.savefig(
            'model/{}/train_perfs_{}.png'.format(model_name, model_name))
    if SHOW_FIG:
        plt.show()
    plt.close()


def plot_length_influence(model_name, show_name):

    step_size = 1
    windows_size = 400
    # Load data:
    df = pd.read_csv(
        'model/{}/lngts_logs.csv'.format(model_name), sep=',').to_numpy()

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
        # 5 is the average number of characters in a work
        size.append(np.mean(df[start_idx:end_idx, 0]) / 5)
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
        plt.savefig(
            'model/{}/{}_length_evaluator.png'.format(model_name, model_name))
    plt.close()

def compare_all_models(models):
    # Store average data for the last epoch of all models
    avg_train_loss = []
    std_train_loss = []
    avg_test_loss = []
    std_test_loss = []
    avg_train_acc = []
    std_train_acc = []
    avg_test_acc = []
    std_test_acc = []
    avg_ls_loss = []
    std_ls_loss = []
    avg_ls_acc = []
    std_ls_acc = []

    for mdl in models:
        df_train = pd.read_csv('model/{}/train_logs.csv'.format(mdl[0]))
        df_test = pd.read_csv('model/{}/test_logs.csv'.format(mdl[0]))
        # Select last epoch values
        df_train = df_train[df_train['Epoch'] >= LAST_EPOCH_IDX]
        df_test = df_test[df_test['Epoch'] >= LAST_EPOCH_IDX]
        # Get average and standard deviation
        avg_train_loss.append(np.mean(df_train['loss'].to_numpy()))
        std_train_loss.append(np.std(df_train['loss'].to_numpy())/2)
        avg_train_acc.append(np.mean(df_train['accuracy'].to_numpy()))
        std_train_acc.append(np.std(df_train['accuracy'].to_numpy())/2)
        avg_test_loss.append(np.mean(df_test['loss'].to_numpy()))
        std_test_loss.append(np.std(df_test['loss'].to_numpy())/2)
        avg_test_acc.append(np.mean(df_test['accuracy'].to_numpy()))
        std_test_acc.append(np.std(df_test['accuracy'].to_numpy())/2)

        # For long sequences evaluation
        df_lgt = pd.read_csv('model/{}/lngts_logs.csv'.format(mdl[0]))
        # Sort by length
        df_lgt = df_lgt.sort_values(by=['length']).to_numpy()
        # Get 1000 longest
        df_lgt = df_lgt[-1000:, :]

        # Get average and std values
        avg_ls_loss.append(np.mean(df_lgt[:, 1]))
        std_ls_loss.append(np.std(df_lgt[:, 1])/2)
        avg_ls_acc.append(np.mean(df_lgt[:, 2]))
        std_ls_acc.append(np.std(df_lgt[:, 2])/2)

    # Barplot
    mod_names = []
    for mdl in models:
        mod_names.append(mdl[2])

    # accuracy bar plot
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(avg_train_acc))
    values = avg_train_acc
    error = std_train_acc

    ax.barh(y_pos, values, xerr=error, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mod_names)
    ax.invert_yaxis()
    ax.set_xlabel('Average train accuracy')
    ax.set_title('Average train accuracy \n'
                 'on epoch 3')
    ax.set_xlim([0.7, 1])
    plt.grid()
    if SHOW_FIG:
        plt.show()
    if SAVE_FIG:
        plt.savefig('ComparisonPlots/train_acc.png')
    plt.close()

    # test accuracy bar plot
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(avg_train_acc))
    values = avg_test_acc
    error = std_test_acc

    ax.barh(y_pos, values, xerr=error, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mod_names)
    ax.invert_yaxis()
    ax.set_xlabel('Average test accuracy')
    ax.set_title('Average test accuracy \n'
                 'on epoch 3')
    ax.set_xlim([0.7, 1])
    plt.grid()
    if SHOW_FIG:
        plt.show()
    if SAVE_FIG:
        plt.savefig('ComparisonPlots/test_acc.png')
    plt.close()

    # loss test bar plot
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(avg_train_acc))
    values = avg_test_loss
    error = std_test_loss

    ax.barh(y_pos, values, xerr=error, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mod_names)
    ax.invert_yaxis()
    ax.set_xlabel('Average test loss')
    ax.set_title('Average test loss \n'
                 'on epoch 3')
    ax.set_xlim([0.06, 0.165])
    plt.grid()
    if SHOW_FIG:
        plt.show()
    if SAVE_FIG:
        plt.savefig('ComparisonPlots/test_loss.png')
    plt.close()

    # loss train bar plot
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(avg_train_acc))
    values = avg_train_loss
    error = std_train_loss

    ax.barh(y_pos, values, xerr=error, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mod_names)
    ax.invert_yaxis()
    ax.set_xlabel('Average train loss')
    ax.set_title('Average train loss \n'
                 'on epoch 3')
    ax.set_xlim([0.06, 0.165])
    plt.grid()
    if SHOW_FIG:
        plt.show()
    if SAVE_FIG:
        plt.savefig('ComparisonPlots/train_loss.png')
    plt.close()

    # long sec loss test bar plot
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(avg_train_acc))
    values = avg_ls_loss
    error = std_ls_loss

    ax.barh(y_pos, values, xerr=error, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mod_names)
    ax.invert_yaxis()
    ax.set_xlabel('Average loss')
    ax.set_title('Average loss on long sequences')
    ax.set_xlim([0, 0.35])
    plt.grid()
    if SHOW_FIG:
        plt.show()
    if SAVE_FIG:
        plt.savefig('ComparisonPlots/ls_loss.png')
    plt.close()

    # long sec acc test bar plot
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(avg_train_acc))
    values = avg_ls_acc
    error = std_ls_acc

    ax.barh(y_pos, values, xerr=error, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mod_names)
    ax.invert_yaxis()
    ax.set_xlabel('Average accuracy')
    ax.set_title('Average accuracy on long sequences')
    ax.set_xlim([0, 1])
    plt.grid()
    if SHOW_FIG:
        plt.show()
    if SAVE_FIG:
        plt.savefig('ComparisonPlots/ls_acc.png')
    plt.close()

def plot_words_attention(tokens, attention, idx, save_fig=False, show_fig=True):

    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(tokens))
    values = attention.flatten()
    error = np.zeros(len(tokens))

    ax.barh(y_pos, values, xerr=error, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens)
    ax.invert_yaxis()
    ax.set_xlabel('Attention score')
    ax.set_title('Word attention')
    ax.set_xlim([0, 1])
    plt.grid()
    if show_fig:
        plt.show()
    if save_fig:
        plt.savefig('ComparisonPlots/sentence_{}.png'.format(idx))
    plt.close()

def plot_length_influence(model_name, show_name):

    step_size = 1
    windows_size = 400
    # Load data:
    df = pd.read_csv(
        'model/{}/lngts_logs.csv'.format(model_name), sep=',').to_numpy()

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
        # 5 is the average number of characters in a work
        size.append(np.mean(df[start_idx:end_idx, 0]) / 5)
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
        plt.savefig(
            'model/{}/{}_length_evaluator.png'.format(model_name, model_name))
    plt.close()


def generate_performances_array(model_list):

    outputs = np.zeros((len(model_list), 12))

    for i in range(0, len(model_list)):
        avg_train_loss = []
        std_train_loss = []
        avg_test_loss = []
        std_test_loss = []
        avg_train_acc = []
        std_train_acc = []
        avg_test_acc = []
        std_test_acc = []
        avg_ls_loss = []
        std_ls_loss = []
        avg_ls_acc = []
        std_ls_acc = []

        df_train = pd.read_csv('model/{}/train_logs.csv'.format(model_list[i][0]))
        df_test = pd.read_csv('model/{}/test_logs.csv'.format(model_list[i][0]))

        # Select last epoch values
        df_train = df_train[df_train['Epoch'] >= LAST_EPOCH_IDX]
        df_test = df_test[df_test['Epoch'] >= LAST_EPOCH_IDX]
        # Get average and standard deviation

        avg_train_loss.append(np.mean(df_train['loss'].to_numpy()))
        std_train_loss.append(np.std(df_train['loss'].to_numpy()))
        avg_train_acc.append(np.mean(df_train['accuracy'].to_numpy()))
        std_train_acc.append(np.std(df_train['accuracy'].to_numpy()))
        avg_test_loss.append(np.mean(df_test['loss'].to_numpy()))
        std_test_loss.append(np.std(df_test['loss'].to_numpy()))
        avg_test_acc.append(np.mean(df_test['accuracy'].to_numpy()))
        std_test_acc.append(np.std(df_test['accuracy'].to_numpy()))

        # For long sequences evaluation
        df_lgt = pd.read_csv('model/{}/lngts_logs.csv'.format(mdl[0]))
        # Sort by length
        df_lgt = df_lgt.sort_values(by=['length']).to_numpy()
        # Get 1000 longest
        df_lgt = df_lgt[-1000:, :]

        # Get average and std values
        avg_ls_loss.append(np.mean(df_lgt[:, 1]))
        std_ls_loss.append(np.std(df_lgt[:, 1])/2)
        avg_ls_acc.append(np.mean(df_lgt[:, 2]))
        std_ls_acc.append(np.std(df_lgt[:, 2])/2)

    # Append in data array
    outputs[:, 0] = avg_train_loss
    outputs[:, 1] = std_train_loss
    outputs[:, 2] = avg_train_acc
    outputs[:, 3] = std_train_acc
    outputs[:, 4] = avg_test_loss
    outputs[:, 5] = std_test_loss
    outputs[:, 6] = avg_test_acc
    outputs[:, 7] = std_test_acc
    outputs[:, 8] = avg_ls_loss
    outputs[:, 9] = std_ls_loss
    outputs[:, 10] = avg_ls_acc
    outputs[:, 11] = std_ls_acc

    file = open('ComparisonPlots/results_tab.csv')
    for i in range(0, len(model_list)):
        row_name = model_list[i][1]
        row = outputs[i, :].tolist()
        row_txt = ','.join(row)
        final_row = '{},{}\n'.format(row_name, row_txt)
        file.write(final_row)


if __name__ == '__main__':

    model_list = [
        ['LSTM_w2v_a', 'LSTM with Word2Vec and Attention', 'LSTM\nW2V\nAttention'],
        ['LSTM_glove_a', 'LSTM with Glove and Attention', 'LSTM\nGlove\nAttention'],
        ['LSTM_glove_na', 'LSTM with Glove and no Attention', 'LSTM\nGlove\nNo Att'],
        ['GRU_glove_a', 'GRU with Glove and Attention', 'GRU\nGlove\nAttention'],
        ['GRU_glove_na', 'GRU with Glove and no Attention', 'GRU\nGlove\nNo Att'],
        ['GRU_fasttext_a', 'GRU with FastText and Attention', 'GRU\nFT\nAttention'],
        ['GRU_w2v_a', 'GRU with Word2Vec and Attention', 'GRU\nGlove\nAttention'],
        ['LSTM_fasttext_a', 'LSTM with FastText and Attention', 'LSTM\nFT\nAttention']
    ]

    for md in model_list:
        plot_model(model_name=md[0], show_name=md[1])




    compare_all_models(model_list)

