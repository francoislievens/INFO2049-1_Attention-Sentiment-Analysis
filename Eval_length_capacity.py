import torch
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from DatasetPreparator import prepare_csv
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from Utils import evaluator, load_model, save_model, save_logs, save_length_logs
from CreateVocab import prepare_vocab, load_vocab
from config import *
from SentimentModel import SentimentModel

import sys


def eval_length_capacity(params):

    p = params
    # If csv file not already prepared:
    if not os.path.exists('data/train.csv'):
        prepare_csv()

    # Prepare fields
    text_field = Field(
        tokenize='basic_english',
        lower=True
    )
    label_field = Field(sequential=False, use_vocab=False)
    length_field = Field(sequential=False, use_vocab=False)
    ft_fields = {'text': ('t', text_field), 'sentiments': ('s', label_field), 'length': ('l', length_field)}

    data = TabularDataset(
        path='data/LargeMovieV1/by_length.csv',
        format='csv',
        fields=ft_fields
    )
    # Load vocabulary file or create it from the train set if needed
    if not os.path.exists('vocab/{}_vocab.pt'.format(p['embedding'])):
        print('Not ready vocab detected. Prepare a new one...')
        prepare_vocab(method=p['embedding'])
        print('... Done')
    # Load the existing embedding vocabulary
    text_field.vocab = load_vocab(p['embedding'])

    # Get iterators
    iterator = BucketIterator(
        data, sort=False, batch_size=1, device=DEVICE
    )

    # Get parameters for the model
    input_size = len(text_field.vocab)

    # Instanciate the model
    model = SentimentModel(input_size=input_size,
                           embed_size=300,
                           name=p['name'],
                           device=DEVICE,
                           rnn_type=p['rnn_type'],
                           use_attention=p['use_attention']).to(DEVICE)
    # Load pre-trained embedding parameters
    model.embed_layer.weight.data.copy_(text_field.vocab.vectors).to(DEVICE)

    # The loss function
    loss_fn = torch.nn.MSELoss()

    # Load exiting model
    start_epoch = load_model(model=model,
                             optimizer=None,
                             model_path=MODEL_PATH,
                             model_name=p['name'])

    # Testing loop
    loop = tqdm(iterator)
    model.eval()

    save_length_logs([], [], [], 'model', p['name'])

    tot_lngts = []
    tot_loss = []
    tot_acc = []
    for step, batch in enumerate(loop):
        with torch.no_grad():
            # Get data
            text = batch.t.permute(1, 0).to(DEVICE)
            sentiment = batch.s.to(DEVICE)
            lngts = batch.l.to(DEVICE)

            # Makes prediction
            pred = model(text)

            # Compute loss and accuracy
            loss = loss_fn(pred.flatten(), sentiment.float())
            acc = accuracy(pred.flatten(), sentiment)
            avg_lng = torch.mean(lngts.type(torch.FloatTensor))
            tot_loss.append(loss.cpu().detach().item())
            tot_lngts.append(avg_lng.cpu().detach().item())
            tot_acc.append(acc.cpu().detach().item())


    save_length_logs(tot_lngts, tot_loss, tot_acc, 'model', p['name'])



def accuracy(pred, target):

    rounded = torch.round(pred)
    # Get correct predictions
    accu = (rounded == target).sum() / pred.shape[0]

    return accu

if __name__ == '__main__':

    elem = {'name': 'LSTM_glove_a',
           'embedding': 'glove',
           'epoch': 2,
           'rnn_type': 'LSTM',
           'use_attention': True}

    eval_length_capacity(elem)