import numpy as np


#######################################################################
#                              Load Data                              #
#######################################################################
def load_data(mode, type=0, snr=0):
    if mode == 'train':
        print('<Training dataset>')
        print('Loading the data...', end='')
        input = np.load('train.npy')
        X = input[:, 0, :].reshape(-1, 16000)
        y = input[:, 1, :].reshape(-1, 16000)
        print('\rComplete to load the data')
    elif mode == 'valid':
        print('<Validation dataset>')
        print('Loading the data...', end='')
        input = np.load('valid.npy')
        X = input[:, 0, :].reshape(-1, 16000)
        y = input[:, 1, :].reshape(-1, 16000)
        print('\rComplete to load the data')
    elif mode == 'test':
        print('<Test dataset>')
        print('Loading the data...', end='')
        input = np.load('test.npy')
        X = input[:, 0, :].reshape(-1, 16000)
        y = input[:, 1, :].reshape(-1, 16000)
        print('\rComplete to load the data')
    return X, y
