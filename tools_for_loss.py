import tensorflow.python.ops.math_ops as M
import tensorflow.keras.backend as K

import keras.backend as K
import tensorflow as tf
import functools
from OBM import *
import pmsqe as pmsqe

window_fn = functools.partial(tf.signal.hann_window, periodic=True)


############################################################################
#               for model structure & loss function                        #
############################################################################
# https://github.com/tensorflow/tensorflow/issues/1666
def log10(x):
    numerator = M.log(x)
    denominator = M.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def l2_norm(s1, s2):
    norm = tf.reduce_sum(s1 * s2, -1, keepdims=True)
    return norm


def sdr_linear(s1, s2, eps=1e-20):
    sn = l2_norm(s1, s1)
    sn_m_shn = l2_norm(s1 - s2, s1 - s2)
    sdr_loss = sn ** 2 / (sn_m_shn ** 2 + eps)
    return -tf.reduce_mean(sdr_loss)


def sdr(s1, s2, eps=1e-20):
    sn = l2_norm(s1, s1)
    sn_m_shn = l2_norm(s1 - s2, s1 - s2)
    sdr_loss = 10 * log10(sn ** 2 / (sn_m_shn ** 2 + eps) + eps)
    return -tf.reduce_mean(sdr_loss)


def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_noise = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)
    snr = 10 * log10(target_norm / (noise_norm + eps) + eps)
    return -tf.reduce_mean(snr)


def si_sdr(reference, estimation, eps=1e-20):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        eps:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    """
    reference_energy = tf.reduce_sum(reference ** 2, axis=-1, keepdims=True)
    optimal_scaling = tf.reduce_sum(reference * estimation, axis=-1, keepdims=True) / (reference_energy + eps)
    projection = optimal_scaling * reference
    noise = estimation - projection
    ratio = tf.reduce_sum(projection ** 2, axis=-1) / (tf.reduce_sum(noise ** 2, axis=-1) + eps)
    ratio = tf.reduce_mean(ratio)
    return -(10 * log10(ratio + eps))


def si_sdr_loss(y_true, y_pred):
    # print("######## SI-SDR LOSS ########")
    # print("y_true shape:      ", K.int_shape(y_true))
    # print("y_pred shape:      ", K.int_shape(y_pred))

    x = y_true
    y = y_pred

    smallVal = 0.0000000001  # To avoid divide by zero

    a = K.sum(y * x, axis=-1, keepdims=True) / (K.sum(x * x, axis=-1, keepdims=True) + smallVal)
    # print("a shape:      ", K.int_shape(a))
    # print("a:            ", K.eval(a))

    xa = a * x
    # print("xa shape:      ", K.int_shape(xa))
    # print("xa:            ", K.eval(xa))

    xay = xa - y
    # print("xay shape:      ", K.int_shape(xay))
    # print("xay:            ", K.eval(xay))

    d = K.sum(xa * xa, axis=-1, keepdims=True) / (K.sum(xay * xay, axis=-1, keepdims=True) + smallVal)
    # print("d shape:      ", K.int_shape(d))
    # print("d:            ", K.eval(d))

    d = -K.mean(10 * log10(d))

    # print("d shape:      ", K.int_shape(d))
    # print("d:            ", K.eval(d))

    # print("Compiling SI-SDR LOSS Done!")
    return d


def estoi_loss(I=8, nbf=200):
    def estoi_loss_inner(y_true, y_pred):
        # print("######## ESTOI LOSS ########")
        # print("y_true shape:      ", K.int_shape(y_true))
        # print("y_pred shape:      ", K.int_shape(y_pred))
        y_pred_shape = K.shape(y_pred)

        stft_true = tf.signal.stft(y_true, 256, 128, 512, window_fn, pad_end=False)
        stft_pred = tf.signal.stft(y_pred, 256, 128, 512, window_fn, pad_end=False)
        # print("stft_true shape:   ", K.int_shape(stft_true))
        # print("stft_pred shape:   ", K.int_shape(stft_pred))

        OBM1 = tf.convert_to_tensor(OBM)
        OBM1 = K.tile(OBM1, [y_pred_shape[0], 1, ])
        OBM1 = K.reshape(OBM1, [y_pred_shape[0], 15, 257, ])
        # print("OBM1 shape:        ", K.int_shape(OBM1))

        OCT_pred = K.sqrt(tf.matmul(OBM1, K.square(K.abs(tf.transpose(stft_pred, perm=[0, 2, 1])))))
        OCT_true = K.sqrt(tf.matmul(OBM1, K.square(K.abs(tf.transpose(stft_true, perm=[0, 2, 1])))))

        # print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[0]))
        # print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[1]))
        # print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[2]))
        N = 30  # length of temporal envelope vectors
        J = 15  # Number of one-third octave bands (cannot be varied)
        M = int(nbf - (N - 1))  # number of temporal envelope vectors
        smallVal = 0.0000000001  # To avoid divide by zero

        d = K.variable(0.0, 'float')
        for i in range(0, I):
            for m in range(0, M):
                x = K.squeeze(tf.slice(OCT_true, [i, 0, m], [1, J, N]), axis=0)
                y = K.squeeze(tf.slice(OCT_pred, [i, 0, m], [1, J, N]), axis=0)
                # print("x shape:   ", K.int_shape(x))
                # print("y shape:   ", K.int_shape(y))
                # print("x shape:   ", K.eval(x))
                # print("y shape:   ", K.eval(y))

                xn = x - K.mean(x, axis=-1, keepdims=True)
                # print("xn shape:   ", K.eval(xn))
                yn = y - K.mean(y, axis=-1, keepdims=True)
                # print("yn shape:   ", K.eval(yn))

                xn = xn / (K.sqrt(K.sum(xn * xn, axis=-1, keepdims=True)) + smallVal)
                # print("xn shape:   ", K.eval(xn))

                yn = yn / (K.sqrt(K.sum(yn * yn, axis=-1, keepdims=True)) + smallVal)
                # print("yn shape:   ", K.eval(yn))

                xn = xn - K.tile(K.mean(xn, axis=-2, keepdims=True), [J, 1, ])
                # print("xn shape:   ", K.eval(xn))

                yn = yn - K.tile(K.mean(yn, axis=-2, keepdims=True), [J, 1, ])
                # print("yn shape:   ", K.eval(yn))

                xn = xn / (K.sqrt(K.sum(xn * xn, axis=-2, keepdims=True)) + smallVal)
                # print("xn shape:   ", K.eval(xn))

                yn = yn / (K.sqrt(K.sum(yn * yn, axis=-2, keepdims=True)) + smallVal)
                # print("yn shape:   ", K.eval(yn))

                di = K.sum(xn * yn, axis=-1, keepdims=True)
                # print("di shape:   ", K.eval(di))
                di = 1 / N * K.sum(di, axis=0, keepdims=False)
                # print("di shape:   ", K.eval(di))
                d = d + di
                # print("d shape:   ", K.eval(d))

        # print("Compiling ESTOI LOSS Done!")
        return 1 - (d / K.cast(I * M, dtype='float'))

    return estoi_loss_inner


def stoi_loss(I=8, nbf=200):
    def stoi_loss_inner(y_true, y_pred):
        # print("######## STOI LOSS ########")
        # print("y_true shape:      ", K.int_shape(y_true))
        # print("y_pred shape:      ", K.int_shape(y_pred))

        y_true = K.squeeze(y_true, axis=-1)
        y_pred = K.squeeze(y_pred, axis=-1)
        y_pred_shape = K.shape(y_pred)

        stft_true = tf.signal.stft(y_true, 256, 128, 512, window_fn, pad_end=False)
        stft_pred = tf.signal.stft(y_pred, 256, 128, 512, window_fn, pad_end=False)
        # print("stft_true shape:   ", K.int_shape(stft_true))
        # print("stft_pred shape:   ", K.int_shape(stft_pred))

        OBM1 = tf.convert_to_tensor(OBM)
        OBM1 = K.tile(OBM1, [y_pred_shape[0], 1, ])
        OBM1 = K.reshape(OBM1, [y_pred_shape[0], 15, 257, ])
        # print("OBM1 shape:        ", K.int_shape(OBM1))

        OCT_pred = K.sqrt(tf.matmul(OBM1, K.square(K.abs(tf.transpose(stft_pred, perm=[0, 2, 1])))))
        OCT_true = K.sqrt(tf.matmul(OBM1, K.square(K.abs(tf.transpose(stft_true, perm=[0, 2, 1])))))

        # print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[0]))
        # print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[1]))
        # print("OCT_pred_shape:    ", K.eval(OCT_pred_shape[2]))

        N = 30  # length of temporal envelope vectors
        J = 15  # Number of one-third octave bands (cannot be varied)
        M = int(nbf - (N - 1))  # number of temporal envelope vectors
        smallVal = 0.0000000001  # To avoid divide by zero
        doNorm = True

        # if doNorm:
        # print("Apply Normalization and Clipping")

        c = K.constant(5.62341325, 'float')  # 10^(-Beta/20) with Beta = -15
        d = K.variable(0.0, 'float')
        for i in range(0, I):  # Run over mini-batches
            for m in range(0, M):  # Run over temporal envelope vectors
                x = K.squeeze(tf.slice(OCT_true, [i, 0, m], [1, J, N]), axis=0)
                y = K.squeeze(tf.slice(OCT_pred, [i, 0, m], [1, J, N]), axis=0)
                # print("x shape:   ", K.int_shape(x))
                # print("y shape:   ", K.int_shape(y))
                # print("x shape:   ", K.eval(x))
                # print("y shape:   ", K.eval(y))

                if doNorm:
                    alpha = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True) / (
                        K.sum(K.square(y), axis=-1, keepdims=True)) + smallVal)
                    # print("alpha shape:   ", K.int_shape(alpha))
                    # print("alpha shape:   ", K.eval(alpha))

                    alpha = K.tile(alpha, [1, N, ])
                    # print("alpha shape:   ", K.int_shape(alpha))
                    # print("alpha shape:   ", K.eval(alpha))

                    ay = y * alpha
                    # print("aY shape:   ", K.int_shape(ay))
                    # print("aY shape:   ", K.eval(ay))

                    y = K.minimum(ay, x + x * c)
                    # print("aY shape:   ", K.int_shape(ay))
                    # print("aY shape:   ", K.eval(ay))

                xn = x - K.mean(x, axis=-1, keepdims=True)
                # print("xn shape:   ", K.eval(xn))
                xn = xn / (K.sqrt(K.sum(xn * xn, axis=-1, keepdims=True)) + smallVal)
                # print("xn shape:   ", K.eval(xn))
                yn = y - K.mean(y, axis=-1, keepdims=True)
                # print("yn shape:   ", K.eval(yn))
                yn = yn / (K.sqrt(K.sum(yn * yn, axis=-1, keepdims=True)) + smallVal)
                # print("yn shape:   ", K.eval(yn))
                di = K.sum(xn * yn, axis=-1, keepdims=True)
                # print("di shape:   ", K.eval(di))
                d = d + K.sum(di, axis=0, keepdims=False)
                # print("d shape:   ", K.eval(K.sum( di ,axis=0,keepdims=False)))

        # print("Compiling STOI LOSS Done!")
        return 1 - (d / K.cast(I * J * M, dtype='float'))

    return stoi_loss_inner


def stsa_mse(y_true, y_pred):
    stft_true = tf.math.abs(tf.signal.stft(y_true, 256, 128, 256, window_fn, pad_end=False))
    stft_pred = tf.math.abs(tf.signal.stft(y_pred, 256, 128, 256, window_fn, pad_end=False))

    d = tf.reduce_mean(tf.math.square(stft_true - stft_pred))

    return d


def pmsqe_loss(I=8):
    def pmsqe_loss_inner(y_true, y_pred):
        stft_true = tf.math.square(tf.math.abs(tf.signal.stft(y_true, 256, 128, 512, window_fn, pad_end=True)))
        stft_pred = tf.math.square(tf.math.abs(tf.signal.stft(y_pred, 256, 128, 512, window_fn, pad_end=True)))

        d = tf.Variable(0.0, 'float')
        for i in range(0, I):
            x = tf.squeeze(tf.slice(stft_true, [i, 0, 0], [1, -1, -1]), axis=0)
            y = tf.squeeze(tf.slice(stft_pred, [i, 0, 0], [1, -1, -1]), axis=0)
            d = d + tf.reduce_mean(pmsqe.per_frame_PMSQE(x, y))
        return d / tf.cast(I, dtype='float')

    return pmsqe_loss_inner


def pmsqe_log_mse_loss(I=8):
    def pmsqe_log_mse_loss_inner(y_true, y_pred):
        stft_true = tf.math.square(tf.math.abs(tf.signal.stft(y_true, 256, 128, 512, window_fn, pad_end=True)))
        stft_pred = tf.math.square(tf.math.abs(tf.signal.stft(y_pred, 256, 128, 512, window_fn, pad_end=True)))

        d = tf.Variable(0.0, 'float')
        for i in range(0, I):
            x = tf.squeeze(tf.slice(stft_true, [i, 0, 0], [1, -1, -1]), axis=0)
            y = tf.squeeze(tf.slice(stft_pred, [i, 0, 0], [1, -1, -1]), axis=0)

            x_log = tf.math.log(x + K.epsilon())
            y_log = tf.math.log(y + K.epsilon())
            logmse = tf.math.square(x_log - y_log)
            logmse = tf.divide(logmse, I)
            logmse = tf.reduce_mean(logmse, axis=-1, keepdims=False)

            d = d + tf.reduce_mean(logmse + pmsqe.per_frame_PMSQE(x, y))
        return d / tf.cast(I, dtype='float')

    return pmsqe_log_mse_loss_inner


def pmsqe_si_snr_loss(I=8):
    def pmsqe_si_snr_loss_inner(y_true, y_pred):
        stft_true = tf.math.square(tf.math.abs(tf.signal.stft(y_true, 256, 128, 512, window_fn, pad_end=True)))
        stft_pred = tf.math.square(tf.math.abs(tf.signal.stft(y_pred, 256, 128, 512, window_fn, pad_end=True)))

        d = tf.Variable(0.0, 'float')
        for i in range(0, I):
            x = tf.squeeze(tf.slice(stft_true, [i, 0, 0], [1, -1, -1]), axis=0)
            y = tf.squeeze(tf.slice(stft_pred, [i, 0, 0], [1, -1, -1]), axis=0)
            d = d + tf.reduce_mean(pmsqe.per_frame_PMSQE(x, y))
        return d / tf.cast(I, dtype='float') + si_snr(y_true, y_pred) / 8

    return pmsqe_si_snr_loss_inner
