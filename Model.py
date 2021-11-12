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
MODEL_PATH = 'model'
TRAIN_SPLIT = 0.8
EMBEDDING = 'glove'         # glove or fasttext
BATCH_SIZE = 10
DEVICE = 'cuda'
NB_EPOCH = 10
HIDDEN_SIZE = 1024
LEARNING_RATE = 1e-4
MODEL_NAME = 'Glove_Test'

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
        self.hidden_size = HIDDEN_SIZE
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

        # Get the embedding of inputs
        # embed_x = self.embed_layer.forward(x, get_embed=True)
        embed_x = self.embed_layer(x)
        # Get batch on the first dim
        embed_x = embed_x.permute(1, 0, 2)

        # Get shapess
        N = embed_x.size()[0]  # Batch size
        lng = embed_x.size()[1]  # length of sequences

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
                                dim=2)
        final_state = final_state.reshape(N, -1, self.hidden_size * 2)
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


def load_model(model, optimizer, model_path, model_name):
    epoch_idx = 0
    if not os.path.exists('{}/{}'.format(model_path, model_name)):
        os.mkdir('{}/{}'.format(model_path, model_name))
        file = open('{}/{}/train_logs.csv'.format(model_path, model_name), 'w')
        file.write('Epoch,train_loss\n')
        file.close()
        file = open('{}/{}/test_logs.csv'.format(model_path, model_name), 'w')
        file.write('Epoch,train_loss\n')
        file.close()
    else:
        # Try to load model's weights
        try:
            model.load_state_dict((torch.load('{}/{}/model_weights.pt'.format(model_path, model_name))))
            print('Previous model loaded')
        except:
            print('Impossible to load existing model: Maybe an existing empty folder exists')
            sys.exit(1)
        # Try to load optimizer weights
        try:
            optimizer.load_state_dict(torch.load('{}/{}/optimizer_weights.pt'.format(model_path, model_name)))
            print('Optimizer weights loaded')
        except:
            print('Fail to load optimizer weights')
        # Load epoch index if the model exists
        train_logs = pd.read_csv('{}/{}/train_logs.csv'.format(model_path, model_name), sep=',')
        epoch_idx = int(train_logs.iloc[-1]['Epoch']) + 1
    return epoch_idx

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
                           name=MODEL_NAME,
                           device=DEVICE).to(DEVICE)
    # Load pre-trained embedding parameters
    model.embed_layer.weight.data.copy_(text_field.vocab.vectors).to(DEVICE)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    # Tensorboard
    tb = SummaryWriter()

    # Load exiting model
    start_epoch = load_model(model=model,
                             optimizer=optimizer,
                             model_path=MODEL_PATH,
                             model_name=MODEL_NAME)

    # Epoch loop
    for i in range(start_epoch, NB_EPOCH):

        print('Training epoch {} / {}'.format(i, NB_EPOCH))
        loop = tqdm(train_iterator, leave=True)

        train_loss = []
        for step, batch in enumerate(loop):
            # Get data
            text = batch.t.to(DEVICE)
            sentiment = batch.s.to(DEVICE)

            # Makes predictions
            pred = model(text)

            # Compute loss
            loss = loss_fn(pred.flatten(), sentiment.float())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logs
            tb.add_scalar('{}_Train'.format(MODEL_NAME), loss.item(), i)
            train_loss.append(loss.item())
            loop.set_postfix(loss=loss.item())

            if step % 500 == 0:
                torch.save(model.state_dict(), '{}/{}/model_weights.pt'.format(MODEL_PATH, MODEL_NAME))
                torch.save(optimizer.state_dict(), '{}/{}/optimizer_weights.pt'.format(MODEL_PATH, MODEL_NAME))
                f = open('{}/{}/train_logs.csv'.format(MODEL_PATH, MODEL_NAME), 'a')
                for itm in train_loss:
                    f.write('{},{}\n'.format(i, itm))
                f.close()
                train_loss = []

        torch.save(model.state_dict(), '{}/{}/model_weights.pt'.format(MODEL_PATH, MODEL_NAME))
        torch.save(optimizer.state_dict(), '{}/{}/optimizer_weights.pt'.format(MODEL_PATH, MODEL_NAME))
        f = open('{}/{}/train_logs.csv'.format(MODEL_PATH, MODEL_NAME), 'a')
        for itm in train_loss:
            f.write('{},{}\n'.format(i, itm))
        f.close()

        # Test loop
        print('Testing epoch {} / {}'.format(i, NB_EPOCH))
        loop = tqdm(test_iterator, leave=True)
        test_loss = []
        for step, batch in enumerate(loop):

            with torch.no_grad():
                # Get data
                text = batch.t.to(DEVICE)
                sentiment = batch.s.to(DEVICE)

                # Makes predictions
                pred = model(text)

                # Compute loss
                loss = loss_fn(pred, sentiment)

            # Logs
            tb.add_scalar('{}_Test'.format(MODEL_NAME), loss.item(), i)
            test_loss.append(loss.item())
            loop.set_postfix(loss=loss.item())

            if step % 500 == 0:
                f = open('{}/{}/test_logs.csv'.format(MODEL_PATH, MODEL_NAME), 'a')
                for itm in test_loss:
                    f.write('{},{}\n'.format(i, itm))
                f.close()
                test_loss = []
        f = open('{}/{}/test_logs.csv'.format(MODEL_PATH, MODEL_NAME), 'a')
        for itm in test_loss:
            f.write('{},{}\n'.format(i, itm))
        f.close()

if __name__ == '__main__':

    train()