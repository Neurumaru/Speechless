import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *

import config as cfg
from tools_for_model import STFT, ISTFT, CCAttention, CSAttention

#######################################################################
#                                models                               #
#######################################################################
filters = [16, 16, 32, 32, 64, 64, 128, 128]
conv2d0_list = list()
conv2d1_list = list()

inputs = Input((16000,))
minmax = tf.keras.constraints.MinMaxNorm(-1, 1)(inputs)
stft = STFT(frame_length=cfg.frame_length, frame_step=cfg.frame_step, fft_length=cfg.fft_len)(minmax)
minmax = tf.keras.constraints.MinMaxNorm(-1, 1)(stft)
split = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(minmax)
conv2d0 = Conv2D(filters[0], (3, 3), strides=(1, 1), padding='same', activation=None)(split[0])
conv2d1 = Conv2D(filters[0], (3, 3), strides=(1, 1), padding='same', activation=None)(split[1])
batch_normalization0 = BatchNormalization()(conv2d0)
batch_normalization1 = BatchNormalization()(conv2d1)
activation0 = PReLU()(batch_normalization0)
activation1 = PReLU()(batch_normalization1)
conv2d0_list.append(activation0)
conv2d1_list.append(activation1)
for f in filters[1:]:
    conv2d0 = Conv2D(f, (3, 3), strides=(1, 2), padding='same', activation=None)(activation0)
    conv2d1 = Conv2D(f, (3, 3), strides=(1, 2), padding='same', activation=None)(activation1)
    batch_normalization0 = BatchNormalization()(conv2d0)
    batch_normalization1 = BatchNormalization()(conv2d1)
    activation0 = PReLU()(batch_normalization0)
    activation1 = PReLU()(batch_normalization1)
    conv2d0_list.append(conv2d0)
    conv2d1_list.append(conv2d1)
reshape0 = Reshape((163, -1))(conv2d0)
reshape1 = Reshape((163, -1))(conv2d1)
lstm0 = LSTM(cfg.n_freqs, return_sequences=True)(reshape0)
lstm1 = LSTM(cfg.n_freqs, return_sequences=True)(reshape1)
lstm0 = LSTM(cfg.n_freqs, return_sequences=True)(lstm0)
lstm1 = LSTM(cfg.n_freqs, return_sequences=True)(lstm1)
dense0 = Dense(cfg.n_freqs, activation=None)(lstm0)
dense1 = Dense(cfg.n_freqs, activation=None)(lstm1)
reshape0 = Reshape((163, -1, filters[-1]))(dense0)
reshape1 = Reshape((163, -1, filters[-1]))(dense1)
batch_normalization0 = BatchNormalization()(reshape0)
batch_normalization1 = BatchNormalization()(reshape1)
activation0 = PReLU()(batch_normalization0)
activation1 = PReLU()(batch_normalization1)

attention0, attention1 = CCAttention(r=2.0, input_shape=conv2d0_list[-1].shape)(conv2d0_list[-1], conv2d1_list[-1])
attention0, attention1 = CSAttention(r=2.0, input_shape=attention0.shape)(attention0, attention1)

concat0 = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([activation0, attention0])
concat1 = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([activation1, attention1])
conv2d_transpose0 = Conv2DTranspose(filters[-2], (3, 3), strides=(1, 2), padding='same', activation=None)(concat0)
conv2d_transpose1 = Conv2DTranspose(filters[-2], (3, 3), strides=(1, 2), padding='same', activation=None)(concat1)
batch_normalization0 = BatchNormalization()(conv2d_transpose0)
batch_normalization1 = BatchNormalization()(conv2d_transpose1)
activation0 = PReLU()(batch_normalization0)
activation1 = PReLU()(batch_normalization1)
for idx, f in enumerate(filters[-3::-1]):
    attention0, attention1 = CCAttention(r=2.0, input_shape=conv2d0_list[-(idx + 2)].shape)(conv2d0_list[-(idx + 2)], conv2d1_list[-(idx + 2)])
    attention0, attention1 = CSAttention(r=2.0, input_shape=attention0.shape)(attention0, attention1)

    concat0 = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([activation0, attention0])
    concat1 = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([activation1, attention1])
    conv2d_transpose0 = Conv2DTranspose(f, (3, 3), strides=(1, 2), padding='same', activation=None)(concat0)
    conv2d_transpose1 = Conv2DTranspose(f, (3, 3), strides=(1, 2), padding='same', activation=None)(concat1)
    batch_normalization0 = BatchNormalization()(conv2d_transpose0)
    batch_normalization1 = BatchNormalization()(conv2d_transpose1)
    activation0 = PReLU()(batch_normalization0)
    activation1 = PReLU()(batch_normalization1)

concat0 = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([activation0, conv2d0_list[-(idx + 3)]])
concat1 = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([activation1, conv2d1_list[-(idx + 3)]])
conv2d_transpose0 = Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same', activation=None)(concat0)
conv2d_transpose1 = Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same', activation=None)(concat1)
batch_normalization0 = BatchNormalization()(conv2d_transpose0)
batch_normalization1 = BatchNormalization()(conv2d_transpose1)
activation0 = PReLU()(batch_normalization0)
activation1 = PReLU()(batch_normalization1)
concat = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([activation0, activation1])

attention0, attention1 = CSAttention(r=2.0, input_shape=split[0].shape)(split[0], split[1])
concat_stft = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([attention0, attention1])

join = Multiply()([concat_stft, concat])
istft = ISTFT(frame_length=cfg.frame_length, frame_step=cfg.frame_step, fft_length=cfg.fft_len)(join)
minmax = tf.keras.constraints.MinMaxNorm(-1, 1)(istft)

model = keras.models.Model(inputs, minmax)
