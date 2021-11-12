import torch
import numpy as np
import os
import sys
import torch.nn as nn
import math
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from DatasetPreparator import prepare_csv
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

DATA_PATH = 'data/aclImdb/'
TRAIN_SPLIT = 0.8
EMBEDDING = 'glove'         # glove or fasttext
BATCH_SIZE = 1
DEVICE = 'cuda'
NB_EPOCH = 10
HIDDEN_SIZE = 1024

class SentimentModel(torch.nn.Module):

    def __init__(self,
                 input_size,
                 embed_size=300,
                 name='Test',
                 device='cpu'):
        super(SentimentModel, self).__init__()
        self.name = name
        self.embed_size = embed_size
        self.device = device
        self.hidden_size = 2000
        self.nb_heads = 4  # Number of heads for multi head self attention

        # The embedding layer
        self.embed_layer = torch.nn.Embedding(input_size, embed_size)

        # Bi directional LSTM layer
        self.rnn_1 = torch.nn.LSTM(embed_size,
                                   self.hidden_size,
                                   num_layers=1,
                                   bidirectional=True,
                                   batch_first=True)

        # Energy FC layer
        self.energy = torch.nn.Linear(in_features=self.hidden_size * 4,
                                      out_features=1)
        self.energy_sm = torch.nn.Softmax(dim=0)

        # output layer
        self.output_fc = torch.nn.Linear(in_features=self.hidden_size * 2,
                                         out_features=1)
        self.output_sm = torch.nn.Sigmoid()

    def forward(self, x):
        # Get shapess
        N = x.size()[0]  # Batch size
        lng = x.size()[1]  # length of sequences
        # Get the embedding of inputs
        # embed_x = self.embed_layer.forward(x, get_embed=True)
        with torch.no_grad():
            embed_x = self.embed_layer(x)

        print(embed_x.shape)
        sys.exit(0)
        # ==================================== #
        #             Encoder part             #
        # ==================================== #

        # Apply the bidirectional LSTM
        hid_states, (final_h_states, final_c_state) = self.rnn_1(embed_x)
        # WARNING shapes:
        #   For hid_state: Batch - seq length - 2 * hidden size
        #   For final states: 2 - batch_size - hidden size

        # ==================================== #
        #             Decoder part             #
        # ==================================== #

        # concatenate the two final states (since bi-directional lstm)
        final_state = torch.cat((final_h_states[0:1, :, :], final_h_states[1:2, :, :]),
                                dim=2).reshape(N, -1, self.hidden_size * 2)
        # Repeat this sequence to math the sequence length
        final_state = final_state.repeat(1, lng, 1)
        # concatenate with all hidden states
        att_states = torch.cat((hid_states, final_state), dim=2)

        # Apply the attention fc layer
        att = self.energy(att_states.reshape(N * lng, self.hidden_size * 4))
        # Apply the softmax to get attention weights vector
        att = self.energy_sm(att)
        # Reshape the outputs per sequences length
        att = att.reshape(N, lng, 1)

        # Apply attention weights on hidden states
        context = torch.einsum('nsk,nsl->nkl', att, hid_states).reshape(N, self.hidden_size * 2)

        # Apply the prediction layer
        output = self.output_fc(context)
        output = self.output_sm(output)

        return output



def train():
    # If csv file not already prepared:
    if not os.path.exists('data/train.csv'):
        prepare_csv()

    # Prepare fields
    text_field = Field(
        tokenize='basic_english',
        lower=True
    )
    label_field = Field(sequential=False, use_vocab=False)

    ft_fields = {'text': ('t', text_field), 'sentiments': ('s', label_field)}

    # Get datasets objects
    train, test = TabularDataset.splits(
        path='data',
        train='train.csv',
        test='test.csv',
        format='csv',
        fields=ft_fields
    )

    # Build the vocabulary embedding vectors from data for fast text and glove
    print('Building vocab...')
    if EMBEDDING == 'fasttext':
        text_field.build_vocab(train, max_size=100000, min_freq=1, vectors='fasttext.en.300d')
    if EMBEDDING == 'glove':
        text_field.build_vocab(train, max_size=100000, min_freq=1, vectors='glove.6B.300d')
    print('... Done')

    # Get iterators
    train_iterator, test_iterator = BucketIterator.splits(
        (train, test), batch_size=BATCH_SIZE, device=DEVICE
    )

    # Get parameters for the model
    input_size = len(text_field.vocab)
    # Instanciate the model
    model = SentimentModel(input_size=input_size,
                           embed_size=300,
                           name='Test',
                           device=DEVICE).to(DEVICE)
    # Load pre-trained embedding parameters
    model.embed_layer.weight.data.copy_(text_field.vocab.vectors).to(DEVICE)



    start_epoch = 0

    # Epoch loop
    for i in range(start_epoch, NB_EPOCH):
        for step, batch in enumerate(train_iterator):
            text = batch.t.to(DEVICE)
            sentiment = batch.s.to(DEVICE)

            pred = model(text)

            print(pred)
            print(text.shape)
            print(sentiment)
            sys.exit(0)


if __name__ == '__main__':

    train()