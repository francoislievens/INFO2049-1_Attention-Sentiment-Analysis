import pandas as pd
import numpy as np
import os
import pickle
import sys
import torchtext
from torchtext.legacy.data import Field, TabularDataset, BucketIterator



DATA_PATH = 'data/aclImdb/'
TRAIN_SPLIT = 0.8

def prepare_csv():

    data_text = []
    data_sentiment = []
    sub_dir_lst = ['test/neg', 'test/pos', 'train/neg', 'train/pos']

    # Run the list of sub directory to load all data
    idx = 0
    print('Input files reading...')
    for sub in sub_dir_lst:
        # Get the sentiment of the folder
        sentiment = 0  # Negative sentiment
        if idx == 1 or idx == 3:
            sentiment = 1

        # Get list of files in the folder
        sub_lst = os.listdir('{}/{}'.format(DATA_PATH, sub))

        # Read and store all files in the list
        for itm in sub_lst:
            f = open('{}{}/{}'.format(DATA_PATH, sub, itm), 'r', encoding='utf8')
            readed = f.read()
            data_text.append(readed)
            f.close()
            data_sentiment.append(sentiment)

        idx += 1
        print('   ... Done.')

    # Shuffle the dataset using fix seed
    shuf_idx = np.arange(len(data_text))
    np.random.shuffle(shuf_idx)
    shuf_idx = shuf_idx.tolist()
    tmp_txt = [data_text[i] for i in shuf_idx]
    tmp_sent = [data_sentiment[i] for i in shuf_idx]
    reviews = tmp_txt
    sentiments = tmp_sent

    # Tokenize data and concat with sentiment
    tmp = []
    for i in range(len(reviews)):
        tmp_sentence = reviews[i]
        tmp_sentence = tmp_sentence.replace(',', ' ')
        tmp.append([tmp_sentence, sentiments[i]])

    # To split in train and test
    split_idx = int(len(reviews) * TRAIN_SPLIT)

    # Put in the dataframe
    df_train = pd.DataFrame(tmp[0:split_idx], columns=['text', 'sentiments'])
    df_test = pd.DataFrame(tmp[split_idx:], columns=['text', 'sentiments'])

    # Save both in csv
    df_train.to_csv('data/train.csv', sep=',', header=True, index=False)
    df_test.to_csv('data/test.csv', sep=',', header=True, index=False)



if __name__ == '__main__':

    prepare_csv()








