import torch
from CreateVocab import *
from torchtext.legacy.data import Field, TabularDataset, BucketIterator


if __name__ == '__main__':

    prepare_vocab(method='glove')
