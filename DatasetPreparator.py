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

    # If csv file not already prepared:
    if not os.path.exists('data/train.csv'):
        prepare_csv()


    # ================================== #
    #       For Fast Text                #
    # ================================== #

    # Prepare fields
    text_field = Field(
        tokenize='basic_english',
        lower=True
    )
    label_field = Field(sequential=False, use_vocab=False)

    ft_fields = {'text': ('t', text_field), 'sentiments': ('s', label_field)}

    train, test = TabularDataset.splits(
        path='data',
        train='train.csv',
        test='test.csv',
        format='csv',
        fields=ft_fields
    )

    # Build the vocabulary embedding vectors from data for fast text and glove
    ft_voc_vec = text_field.build_vocab(train, max_size=10000, min_freq=1, vectors='fasttext.en.300d')
    gl_voc_vec = text_field.build_vocab(train, max_size=10000, min_freq=1, vectors='glove.6B.300d')

    print('check')
    sys.exit(0)







"""
    # ================================== #
    #       For Glove                    #
    # ================================== #

    # Use fields
    go_text_field = Field(
        tokenize='basic_english',
        lower=True
    )
    go_label_field = Field(sequential=False, use_vocab=False)

    # Preprocess data:
    go_preprocessed = df['text'].apply(lambda x: go_text_field.preprocess(x))

    # Load fast text 300d
    go_text_field.build_vocab(
        go_preprocessed,
        vectors='glove.6B.300d'
    )

    # Get the vocab
    go_vocab = go_text_field.vocab      # Can be used as a dic with the word we search as key

"""
