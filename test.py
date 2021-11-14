import torch
from CreateVocab import *
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from Train import train
import matplotlib.pyplot as plt

if __name__ == '__main__':

    prepare_vocab(method='word2vec')

