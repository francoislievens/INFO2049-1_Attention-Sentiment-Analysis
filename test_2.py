from Train import train


if __name__ == '__main__':

    train({'name': 'test_2',
           'embedding': 'glove',
           'epoch': 1,
           'rnn_type': 'LSTM'})
