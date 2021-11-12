import torch
import numpy as np
import os
import sys
import torch.nn as nn
import math
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel


class SentimentModel(torch.nn.Module):

    def __init__(self,
                 embed_size=200,
                 name='Test',
                 device='cpu',
                 embedding=None,
                 sentences_length=500):
        super(SentimentModel, self).__init__()
        self.name = name
        self.embed_size = embed_size
        self.device = device
        self.hidden_size = 2000
        self.sentences_length = sentences_length
        self.nb_heads = 4  # Number of heads for multi head self attention

        # Store the embedding layer: the model have to be given in parameters
        self.embed_layer = embedding

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

    def forward(self, x, mask):
        # Get shapess
        N = x.size()[0]  # Batch size
        lng = x.size()[1]  # length of sequences
        # Get the embedding of inputs
        # embed_x = self.embed_layer.forward(x, get_embed=True)
        with torch.no_grad():
            embed_x = self.embed_layer(x, mask)

        print(len(embed_x))
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

    pass





if __name__ == '__main__':

    train()