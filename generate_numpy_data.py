import numpy as np
import time
import os
from scipy.io import wavfile
from scipy import signal
import glob
from sklearn.model_selection import train_test_split

fs = 16000
data_length = 48000


class Bar():
    def __init__(self, data, label=None):
        self.data = data
        self.iterator = iter(data)
        self._idx = 1
        self._time = None
        self._DISPLAY_LENGTH = 50

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self._time is None:
            self._time = time.time()

        try:
            data = next(self.iterator)
            self._display()
        except StopIteration:
            raise StopIteration()

        self._idx += 1
        if self._idx > len(self.data):
            self._reset()

        return data

    def _display(self):
        if self._time is not None:
            t = (time.time() - self._time) / self._idx
            eta = t * (len(self.data) - self._idx)
        else:
            eta = 0

        rate = self._idx / len(self.data)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        bar = ('=' * len_bar + ('>' if rate < 1 else '')).ljust(self._DISPLAY_LENGTH, '.')

        tmpl = '\r{}/{}: [{}] - ETA {:>5.1f}s'.format(
            self._idx,
            len(self.data),
            bar,
            eta
        )

        print(tmpl, end='')
        if self._idx == len(self.data):
            print()

    def _reset(self):
        self._idx = 1
        self._time = []


def load_wav_file(file):
    rate, wav = wavfile.read(file)
    if rate != fs:
        wav = signal.resample(wav, fs)
    return wav


def preprocess_wav_data(wav):
    wav = wav[:len(wav) - len(wav) % data_length].reshape(-1, 1, data_length)
    wav = wav.astype(np.float32) / np.max(np.abs(wav))
    return wav


def load_wav_data(path):
    i = 0
    clean_dir = os.path.join(path, 'clean')
    noisy_dir = os.path.join(path, 'noisy')

    clean_files = os.listdir(clean_dir)
    datalist = list()
    for clean_file in Bar(clean_files):
        clean_file_wav = load_wav_file(os.path.join(clean_dir, clean_file))
        clean_file_wav = preprocess_wav_data(clean_file_wav)
        clean_file_name = os.path.splitext(clean_file)[0]
        clean_file_pattern = os.path.join(noisy_dir, clean_file_name + '*' + '.wav')

        for noisy_file in glob.glob(clean_file_pattern):
            noisy_file_wav = load_wav_file(os.path.join(noisy_file))
            noisy_file_wav = preprocess_wav_data(noisy_file_wav)
            data = np.concatenate((noisy_file_wav, clean_file_wav), axis=1)
            datalist.extend(data)

    return np.array(datalist, dtype=np.float32)


data = load_wav_data('train')
train, test = train_test_split(data, test_size=500)

print('data shape: ' + str(data.shape))
print('train shape: ' + str(train.shape))
print('test shape: ' + str(test.shape))

print('Saving Numpy Data...', end='')
np.save('train', train)
np.save('valid', test)
print('\rSaved Numpy Data    ')
