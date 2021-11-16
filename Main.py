from Train import train
from Eval_length_capacity import eval_length_capacity
from Ploter import plot_model
from Ploter import compare_all_models
from Ploter import generate_performances_array



if __name__ == '__main__':

    # The dictionary who contain the description of all models that we want to create, train and evaluate
    tmp = {
            'name': 'LSTM_w2v_a',
            'embedding': 'word2vec',
            'epoch': 3,
            'rnn_type': 'LSTM',
            'use_attention': True
        }, {
            'name': 'LSTM_glove_na',
            'embedding': 'glove',
            'epoch': 3,
            'rnn_type': 'LSTM',
            'use_attention': False
        }
    parameters = [
        {
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
            'name': 'GRU_glove_na',
            'embedding': 'glove',
            'epoch': 3,
            'rnn_type': 'GRU',
            'use_attention': False
        }
    ]

    # Training loop
    for prm in parameters:
        print('* --------------------------------------- *')
        print('*      Starting training {}'.format(prm['name']))
        print('* --------------------------------------- *')
        train(prm)

    parameters = [
        {
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
            'name': 'GRU_glove_na',
            'embedding': 'glove',
            'epoch': 3,
            'rnn_type': 'GRU',
            'use_attention': False
        }, {
            'name': 'LSTM_w2v_a',
            'embedding': 'word2vec',
            'epoch': 3,
            'rnn_type': 'LSTM',
            'use_attention': True
        }, {
            'name': 'LSTM_glove_na',
            'embedding': 'glove',
            'epoch': 3,
            'rnn_type': 'LSTM',
            'use_attention': False
        }
    ]

    # Evaluate on long sequences
    for prm in parameters:
        print('* --------------------------------------- *')
        print('*      Starting evaluating {}'.format(prm['name']))
        print('* --------------------------------------- *')
        eval_length_capacity(prm)

    # A new model list to plot results
    model_list = [
        ['LSTM_w2v_a', 'LSTM with Word2Vec and Attention', 'LSTM\nW2V\nAttention'],
        ['LSTM_glove_a', 'LSTM with Glove and Attention', 'LSTM\nGlove\nAttention'],
        ['LSTM_glove_na', 'LSTM with Glove and no Attention', 'LSTM\nGlove\nNo Att'],
        ['GRU_glove_a', 'GRU with Glove and Attention', 'GRU\nGlove\nAttention'],
        ['GRU_glove_na', 'GRU with Glove and no Attention', 'GRU\nGlove\nNo Att'],
        ['GRU_fasttext_a', 'GRU with FastText and Attention', 'GRU\nFT\nAttention'],
        ['GRU_w2v_a', 'GRU with Word2Vec and Attention', 'GRU\nGlove\nAttention'],
        ['LSTM_fasttext_a', 'LSTM with FastText and Attention', 'LSTM\nFT\nAttention']
    ]

    # Plot all models results
    for md in model_list:
        plot_model(model_name=md[0], show_name=md[1])

    # Plots to compare all models
    compare_all_models(model_list)

    # Generate the array with performances data
    #generate_performances_array(model_list)