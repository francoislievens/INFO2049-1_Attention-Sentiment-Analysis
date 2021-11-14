import torch
import pickle
import numpy as np
import pandas as pd
import os
import sys

def evaluator(token_idx, vocab, pred, attention, target):

    # For each elements in the batch
    for i in range(0, token_idx.shape[0]):
        tok_idx = token_idx[i]
        # Get tokens
        tok = []
        for idx in tok_idx:
            tok.append(vocab.itos[idx])

        # Print results:
        print(' ========================== ')
        print('Final positivity: {}'.format(pred[i].item()))
        print('Attention sum: {}'.format(torch.sum(attention[i]).detach().cpu().numpy()))
        print('Target: {}'.format(target[i].item()))
        for j in range(0, len(tok_idx)):
            print('{} - {} - {}'.format(tok_idx[j],
                                        tok[j],
                                        attention[i, j].item()))


def load_model(model, optimizer, model_path, model_name, device='cuda'):
    epoch_idx = 0
    if not os.path.exists('{}/{}'.format(model_path, model_name)):
        os.mkdir('{}/{}'.format(model_path, model_name))
        file = open('{}/{}/train_logs.csv'.format(model_path, model_name), 'w')
        file.write('Epoch,loss,accuracy\n')
        file.close()
        file = open('{}/{}/test_logs.csv'.format(model_path, model_name), 'w')
        file.write('Epoch,loss,accuracy\n')
        file.close()
    else:
        # Try to load model's weights
        try:
            model.load_state_dict(
                (torch.load('{}/{}/model_weights.pt'.format(model_path, model_name), map_location=device)))
            print('Previous model loaded')
        except:
            print('Impossible to load existing model: Maybe an existing empty folder exists')
            sys.exit(1)
        # Try to load optimizer weights
        try:
            if optimizer is not None:
                optimizer.load_state_dict(
                    torch.load('{}/{}/optimizer_weights.pt'.format(model_path, model_name), map_location=device))
                print('Optimizer weights loaded')
        except:
            print('Fail to load optimizer weights')
        # Load epoch index if the model exists
        train_logs = pd.read_csv('{}/{}/train_logs.csv'.format(model_path, model_name), sep=',')
        epoch_idx = int(train_logs.iloc[-1]['Epoch']) + 1
    if optimizer is not None:
        return epoch_idx


def save_model(model, optimizer, model_path, model_name):

    torch.save(model.state_dict(), '{}/{}/model_weights.pt'.format(model_path, model_name))
    torch.save(optimizer.state_dict(), '{}/{}/optimizer_weights.pt'.format(model_path, model_name))

def save_logs(epoch, loss, accuracy, model_path, model_name, type='train'):

    f = open('{}/{}/{}_logs.csv'.format(model_path, model_name, type), 'a')
    for i in range(0, len(loss)):
        f.write('{},{},{}\n'.format(epoch, loss[i], accuracy[i]))
    f.close()

def save_length_logs(lgts, loss, accuracy, model_path, model_name):

    if not os.path.exists('{}/{}/lngts_logs.csv'.format(model_path, model_name)):
        f = open('{}/{}/lngts_logs.csv'.format(model_path, model_name), 'w')
        f.write('length,loss,accuracy\n')
        f.close()
    f = open('{}/{}/lngts_logs.csv'.format(model_path, model_name), 'a')
    for i in range(0, len(lgts)):
        f.write('{},{},{}\n'.format(lgts[i],
                                    loss[i],
                                    accuracy[i]))
        f.close()

