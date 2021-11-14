import pandas as pd
import numpy as np
import os
import pickle
import sys
import torchtext
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from tqdm import tqdm


DATA_PATH = 'data/aclImdb/'
TRAIN_SPLIT = 0.8


def shuffle_data(text, sentiments, datasetName):
    # Shuffle the dataset using fix seed
    shuf_idx = np.arange(len(text))
    np.random.shuffle(shuf_idx)
    shuf_idx = shuf_idx.tolist()
    tmp_txt = [text[i] for i in shuf_idx]
    tmp_sent = [sentiments[i] for i in shuf_idx]
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
    df_train.to_csv('data/'+datasetName+'_train.csv',
                    sep=',', header=True, index=False)
    df_test.to_csv('data/'+datasetName+'_test.csv',
                   sep=',', header=True, index=False)


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
            f = open('{}{}/{}'.format(DATA_PATH, sub, itm),
                     'r', encoding='utf8')
            readed = f.read()
            data_text.append(readed)
            f.close()
            data_sentiment.append(sentiment)

        idx += 1
        print('   ... Done.')

    shuffle_data(data_text, data_sentiment, "")


def tweete_preparation(path="data/tweete/training.1600000.processed.noemoticon.csv"):
    header = ["sentiments", "The id of the tweet",
              "date", "query", "the user that tweeted", "tweet"]
    data_tweet = pd.read_csv(path, names=header, encoding='latin')

    data_tweet['sentiments'] = data_tweet['sentiments'] / 4

    data_tweet = data_tweet.astype({'sentiments': int})

    sentiments = np.array(data_tweet["sentiments"])
    tweet = np.array(data_tweet["tweet"])
    shuffle_data(tweet, sentiments, "tweet")

def prepare_large_movie_data():

    # Load the serialized dataset
    with open('data/LargeMovieV1/largemovie.pkl', 'rb') as f:
        text, encoded_batch, sentiments = pickle.load(f)

    # Build a length array:
    lgt = []
    for i in range(len(text)):
        lgt.append(len(text[i]))
        if i > 10000:
            break
    lgt = np.array(lgt)
    idx_array = np.arange(len(lgt))

    # Shuffle index
    idx_len = np.zeros((len(idx_array), 2))
    idx_len[:, 0] = idx_array
    idx_len[:, 1] = lgt

    # Sort by length
    idx_len = idx_len[np.argsort(idx_len[:, 1])]

    # Build a data csv file sorted by length
    outputs = []
    for i in range(0, idx_len.shape[0]):
        outputs.append('{},{},{}\n'.format(text[int(idx_len[i, 0])].replace(',', ' '),
                                         int(sentiments[int(idx_len[i, 0])]),
                                         int(idx_len[i, 1])))

    # Write in a file
    f = open('data/LargeMovieV1/by_length.csv', 'w', encoding='utf-8')
    f.write('text,sentiments,length\n')
    for i in tqdm(range(0, len(outputs))):
        f.write(outputs[i])
    f.close()

if __name__ == '__main__':
    # tweete_preparation()

    # prepare_csv()
    prepare_large_movie_data()
