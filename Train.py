import torch
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from DatasetPreparator import prepare_csv
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from Utils import evaluator, load_model, save_model, save_logs
from CreateVocab import prepare_vocab, load_vocab
from config import *
from SentimentModel import SentimentModel
import sys


def train(parameters):
    p = parameters
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
        train='tweet_train.csv',
        test='tweet_test.csv',
        format='csv',
        fields=ft_fields,
    )

    # Use pre-trained vocabulary embedding
    # Check if embedding exists:
    if not os.path.exists('vocab/{}_vocab.pt'.format(p['embedding'])):
        print('Not ready vocab detected. Prepare a new one...')
        prepare_vocab(method=p['embedding'])
        print('... Done')
    # Load the existing embedding vocabulary
    text_field.vocab = load_vocab(p['embedding'])

    # Get iterators
    train_iterator, test_iterator = BucketIterator.splits(
        (train, test), sort=False, batch_size=BATCH_SIZE, device=DEVICE
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

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    # Tensorboard
    tb = SummaryWriter()

    # Load exiting model
    start_epoch = load_model(model=model,
                             optimizer=optimizer,
                             model_path=MODEL_PATH,
                             model_name=p['name'])

    # Epoch loop
    for i in range(start_epoch, p['epoch']):

        print('Training epoch {} / {}'.format(i, p['epoch']))
        loop = tqdm(train_iterator, leave=True)

        train_loss = []
        train_accuracy = []
        for step, batch in enumerate(loop):
            # Get data
            text = batch.t.permute(1, 0).to(DEVICE)
            sentiment = batch.s.to(DEVICE)

            # Makes predictions
            attention = None
            if EVAL:
                pred, attention = model.forward(text, return_att=True)
            else:
                pred = model(text)
            # Compute loss
            loss = loss_fn(pred.flatten(), sentiment.float())

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            # Compute accuracy:
            acc = accuracy(pred.flatten(), sentiment)

            if EVAL:
                evaluator(text, text_field.vocab, pred, attention, sentiment)

            # Logs
            tb.add_scalar('{}_Train_loss'.format(p['name']), loss.item(), i)
            tb.add_scalar('{}_Train_accuracy'.format(p['name']), acc.item(), i)
            train_loss.append(loss.item())
            train_accuracy.append(acc.item())
            loop.set_postfix(loss=loss.item())

            if step % 500 == 0:
                save_model(model, optimizer, MODEL_PATH, p['name'])
                save_logs(i, train_loss, train_accuracy, MODEL_PATH, p['name'], 'train')
                train_loss = []
                train_accuracy = []

        save_model(model, optimizer, MODEL_PATH, p['name'])
        save_logs(i, train_loss, train_accuracy, MODEL_PATH, p['name'], 'train')

        # Test loop

        print('Testing epoch {} / {}'.format(i, p['epoch']))
        loop = tqdm(test_iterator, leave=True)
        test_loss = []
        test_accuracy = []
        for step, batch in enumerate(loop):

            with torch.no_grad():
                # Get data
                text = batch.t.permute(1, 0).to(DEVICE)
                sentiment = batch.s.to(DEVICE)

                # Makes predictions
                pred = model(text)

                # Compute loss
                loss = loss_fn(pred.flatten(), sentiment)

                # Compute the accuracy
                acc = accuracy(pred.flatten(), sentiment)

            # Logs
            tb.add_scalar('{}_Test_loss'.format(p['name']), loss.item(), i)
            tb.add_scalar('{}_Test_accuracy'.format(p['name']), acc.item(), i)
            test_loss.append(loss.item())
            test_accuracy.append(acc.item())
            loop.set_postfix(loss=loss.item())

            if step % 500 == 0:
                save_logs(i, test_loss, test_accuracy, MODEL_PATH, p['name'], 'test')
                test_loss = []
                test_accuracy = []
        save_logs(i, test_loss, test_accuracy, MODEL_PATH, p['name'], 'test')


def accuracy(pred, target):
    rounded = torch.round(pred)
    # Get correct predictions
    accu = (rounded == target).sum() / pred.shape[0]

    return accu


if __name__ == '__main__':

    parameters = [
        {
            'name': 'LSTM_w2v_a',
            'embedding': 'word2vec',
            'epoch': 3,
            'rnn_type': 'LSTM',
            'use_attention': True
        }, {
            'name': 'GRU_w2v_a',
            'embedding': 'word2vec',
            'epoch': 3,
            'rnn_type': 'GRU',
            'use_attention': True
        }, {
            'name': 'LSTM_glove_a',
            'embedding': 'glove',
            'epoch': 3,
            'rnn_type': 'LSTM',
            'use_attention': True
        }, {
            'name': 'LSTM_fasttext_a',
            'embedding': 'fasttext',
            'epoch': 3,
            'rnn_type': 'LSTM',
            'use_attention': True
        }, {
            'name': 'GRU_glove_a',
            'embedding': 'glove',
            'epoch': 3,
            'rnn_type': 'GRU',
            'use_attention': True
        }, {
            'name': 'GRU_fasttext_a',
            'embedding': 'fasttext',
            'epoch': 3,
            'rnn_type': 'GRU',
            'use_attention': True
        }, {
            'name': 'LSTM_glove_na',
            'embedding': 'glove',
            'epoch': 3,
            'rnn_type': 'LSTM',
            'use_attention': False
        }, {
            'name': 'GRU_glove_na',
            'embedding': 'glove',
            'epoch': 3,
            'rnn_type': 'GRU',
            'use_attention': False
        }
    ]
    for prm in parameters:
        print('* --------------------------------------- *')
        print('*      Starting training {}'.format(prm['name']))
        print('* --------------------------------------- *')
        train(prm)
