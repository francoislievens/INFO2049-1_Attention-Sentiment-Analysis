import torch
from config import *


class SentimentModel(torch.nn.Module):

    def __init__(self,
                 input_size,
                 embed_size=300,
                 name='Test',
                 device='cpu',
                 rnn_type='LSTM'):
        super(SentimentModel, self).__init__()
        self.name = name
        self.embed_size = embed_size
        self.device = device
        self.hidden_size = HIDDEN_SIZE

        # The embedding layer
        self.embed_layer = torch.nn.Embedding(input_size, embed_size)

        # Bi directional LSTM layer
        self.rnn_1 = None
        if rnn_type == 'LSTM':
            self.rnn_1 = torch.nn.LSTM(embed_size,
                                       self.hidden_size,
                                       num_layers=1,
                                       bidirectional=True,
                                       batch_first=True)
        else:
            self.rnn_1 = torch.nn.GRU(embed_size,
                                      self.hidden_size,
                                      num_layers=1,
                                      bidirectional=True,
                                      batch_first=True)

        # Energy FC layer
        self.energy_1 = torch.nn.Linear(in_features=self.hidden_size * 4 + self.embed_size,
                                        out_features=self.hidden_size)
        self.energy_relu = torch.nn.ReLU()
        self.energy_2 = torch.nn.Linear(in_features=self.hidden_size,
                                        out_features=1)
        self.energy_sm = torch.nn.Softmax(dim=0)

        # output layer
        self.output_fc = torch.nn.Linear(in_features=self.hidden_size * 2,
                                         out_features=1)
        self.output_sm = torch.nn.Sigmoid()

        # Weights init

        for name, param in self.rnn_1.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias':
                param.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.energy_1.weight)
        torch.nn.init.xavier_uniform_(self.energy_2.weight)
        torch.nn.init.xavier_uniform_(self.output_fc.weight)

    def forward(self, x, return_att=False):

        # Get the embedding of inputs
        # embed_x = self.embed_layer.forward(x, get_embed=True)
        with torch.no_grad():
            embed_x = self.embed_layer(x)
        if EVAL:
            for i in range(0, embed_x[0].shape[0]):
                print(embed_x[0, i, 0:10])
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
        #final_state = torch.cat((final_h_states[0:1, :, :], final_h_states[1:2, :, :]),
        #                        dim=2)
        final_state = torch.cat((hid_states[:, -1:, self.hidden_size:],
                                 hid_states[:, 0:1, 0:self.hidden_size]),
                                dim=2)
        final_state = final_state.reshape(N, -1, self.hidden_size * 2)
        # Repeat this sequence to math the sequence length
        final_state = final_state.repeat(1, lng, 1)
        # concatenate with all hidden states
        att_states = torch.cat((hid_states, final_state, embed_x), dim=2)

        # Apply the attention fc layers
        att = self.energy_1(att_states.reshape(N * lng, self.hidden_size * 4 + self.embed_size))
        att = self.energy_relu(att)
        att = self.energy_2(att)
        # Apply the softmax to get attention weights vector
        att = self.energy_sm(att)
        # Reshape the outputs per sequences length
        att = att.reshape(N, lng, 1)

        # Apply attention weights on hidden states
        context = torch.einsum('nsk,nsl->nkl', att, hid_states).reshape(N, self.hidden_size * 2)

        # Apply the prediction layer
        output = self.output_fc(context)
        output = self.output_sm(output)

        if return_att:
            return output, att

        return output