"""
The code in this file can be used to evaluate how our
trained algorithm work on example sentences. Input text
have to be written in Input/inputs.csv. For all sequences
in this file, the sentiment prediction and the attention
tensor are outputs. Also, plots with attention weights are
builds according the the class variables selection
"""
import torch
from torchtext.legacy.data import Field, TabularDataset
import os
from SentimentModel import SentimentModel
from Utils import load_model
from CreateVocab import load_vocab, prepare_vocab
from Ploter import plot_words_attention

# Put here the target model specification
MODEL_NAME = 'LSTM_glove_a'
DEVICE = 'cuda'
EMBEDDING = 'glove'
MODEL_PATH = 'model'
USE_ATTENTION = True
FT_INPUT_SIZE = 100002
EMBED_SIZE = 300
PLOT = True
SHOW_PLOT = False
SAVE_PLOT = True

if __name__ == '__main__':

    # =============================== #
    #       Data loading              #
    # =============================== #

    # Prepare fields
    text_field = Field(
        tokenize='basic_english',
        lower=True
    )
    label_field = Field(sequential=False, use_vocab=False)
    ft_fields = {'text': ('t', text_field), 'sentiments': ('s', label_field)}

    # Get datasets objects
    text = TabularDataset(
        path='Inputs/inputs.csv',
        format='csv',
        fields=ft_fields,
    )

    # Load vocabulary file or create it from the train set if needed
    if not os.path.exists('vocab/{}_vocab.pt'.format(EMBEDDING)):
        print('Not ready vocab detected. Prepare a new one...')
        prepare_vocab(method=EMBEDDING)
        print('... Done')
    # Load the existing embedding vocabulary
    text_field.vocab = load_vocab(EMBEDDING)

    # Get parameters for the model
    input_size = len(text_field.vocab)
    print('Input size:  {} '.format(input_size))

    # =============================== #
    #       Model loading             #
    # =============================== #
    inpt_size = FT_INPUT_SIZE
    if EMBEDDING == 'fasttext':
        # Change the input size for fasttext
        pass
    model = SentimentModel(input_size=inpt_size,
                           embed_size=300,
                           name=MODEL_NAME,
                           device=DEVICE,
                           use_attention=USE_ATTENTION).to(DEVICE)
    # Load existing weights
    load_model(model,
               optimizer=None,
               model_path=MODEL_PATH,
               model_name=MODEL_NAME,
               device=DEVICE)

    # Reset the embeding dict with actual one
    model.embed_layer = torch.nn.Embedding(input_size, EMBED_SIZE).to(DEVICE)
    # Load weights
    model.embed_layer.weight.data.copy_(text_field.vocab.vectors).to(DEVICE)

    # =============================== #
    #       Reading Loop              #
    # =============================== #

    # To see what is in the vocab
    #print(text_field.vocab.stoi.items())
    #print(text_field.vocab.stoi['it'])

    idx = 0
    for itm in text.examples:
        print(' ======================================================================= ')

        txt = ['<unk>']
        txt = txt + list(itm.t)
        for k in range(5):
            txt.append('<pad>')
        # Get indexes using the vocab
        txt_idx = []
        for tok in txt:
            txt_idx.append(text_field.vocab.stoi[tok])

        txt_idx_tensor = torch.tensor(txt_idx).reshape(1, -1)
        with torch.no_grad():
            pred, att = model.forward(txt_idx_tensor.to(DEVICE), return_att=True)

        pred.cpu().numpy()
        att = att.flatten().cpu().numpy()
        print('prediction:  {}'.format(pred.item()))
        for i in range(0, len(att)):
            print('{} - {} - {} - {} '.format(i, txt[i], txt_idx[i], att[i]))

        plot_words_attention(txt, att, save_fig=SAVE_PLOT, show_fig=SHOW_PLOT, idx=idx)
        idx += 1







