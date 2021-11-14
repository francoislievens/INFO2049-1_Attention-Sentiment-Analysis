from torchtext.legacy.data import Field, TabularDataset
import pickle
import sys
import gensim
from tqdm import tqdm
import torch

def prepare_vocab(method='fasttext'):
    """
    This function save a vector dictionary for each
    pre-trained embedding in order to be used by the
    sentiment analysis model. To build this dictionary,
    the 100 000 most used words are taken in the dataset
    :param method: "fasttext", "glove" or "word2vec"
    """

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

    # Build Vocabulary
    if method == 'fasttext':
        text_field.build_vocab(train, max_size=100000, min_freq=3, vectors='fasttext.en.300d')
        save_vocab(text_field.vocab, 'vocab/fasttext_vocab.pt')
    elif method == 'glove':
        text_field.build_vocab(train, max_size=100000, min_freq=3, vectors='glove.6B.300d')
        save_vocab(text_field.vocab, 'vocab/glove_vocab.pt')

    elif method == 'word2vec':
        print('Start word2vec...')
        # SOURCE: https://github.com/alexandres/lexvec#pre-trained-vectors
        # Start by using fasttext vocabulary to get tokens
        ft_vocab = load_vocab(method='fasttext')
        # Load word2vec vectors using gensim
        #wtv_gensim = gensim.models.word2vec.Word2Vec.load('.vector_cache/lexvec.commoncrawl.300d.W.pos.vectors.gz')
        wtv_gensim = gensim.models.KeyedVectors.load_word2vec_format('.vector_cache/lexvec.commoncrawl.300d.W.pos.neg3.vectors', binary=False)
        # Set vectors manually for each wanted tokens
        wtv_vectors = []
        for token, idx in tqdm(ft_vocab.stoi.items()):
            if token in wtv_gensim.key_to_index.keys():
                wtv_vectors.append(torch.FloatTensor(wtv_gensim[token]))
            else:
                wtv_vectors.append(torch.zeros(300))
        ft_vocab.set_vectors(ft_vocab.stoi, wtv_vectors, 300)
        save_vocab(ft_vocab, 'vocab/word2vec_vocab.pt')
        print('Done')

    else:
        print('Embeding \" {} \" not already implemented in this project.')
        sys.exit(1)


def save_vocab(vocab, path):

    f = open(path, 'wb')
    pickle.dump(vocab, f)
    f.close()

def load_vocab(method='fasttext'):

    f = open('vocab/{}_vocab.pt'.format(method), 'rb')
    vocab = pickle.load(f)
    f.close()
    return vocab