import tensorflow as tf


###############################################################################
#                             STFT & InverseSTFT                              #
###############################################################################
class STFT(tf.keras.layers.Layer):
    def __init__(self, frame_length, frame_step, fft_length):
        super(STFT, self).__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

    def call(self, inputs):
        pad = tf.pad(inputs, tf.constant([[0, 0], [300, 300]]))
        comp = tf.signal.stft(pad, self.frame_length, self.frame_step, self.fft_length)
        return tf.stack([tf.math.real(comp), tf.math.imag(comp)], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'frame_length': self.frame_length,
            'frame_step': self.frame_step,
            'fft_length': self.fft_length,
        })
        return config


class ISTFT(tf.keras.layers.Layer):
    def __init__(self, frame_length, frame_step, fft_length):
        super(ISTFT, self).__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

    def call(self, inputs):
        real = inputs[:, :, :, 0]
        imag = inputs[:, :, :, 1]
        comp = tf.complex(real, imag)
        return tf.signal.inverse_stft(comp, self.frame_length, self.frame_step, self.fft_length,
                                      window_fn=tf.signal.inverse_stft_window_fn(self.frame_step))[:, 300:-300]

    def get_config(self):
        config = super().get_config()
        config.update({
            'frame_length': self.frame_length,
            'frame_step': self.frame_step,
            'fft_length': self.fft_length,
        })
        return config


###############################################################################
#                                 Attention                                   #
###############################################################################
class CCAttention(tf.keras.layers.Layer):
    def __init__(self, r, input_shape):
        self.r = r

        self.concat = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))
        self.maxpooling2D = tf.keras.layers.MaxPooling2D(pool_size=(input_shape[1], input_shape[2]))
        self.avgpooling2D = tf.keras.layers.AvgPool2D(pool_size=(input_shape[1], input_shape[2]))

        if input_shape[3] * 2 // self.r == 0:
            self.maxpoolingConv2D0 = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), activation=None)
            self.avgpoolingConv2D0 = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), activation=None)
        else:
            self.maxpoolingConv2D0 = tf.keras.layers.Conv2D(input_shape[3] * 2 // self.r, kernel_size=(1, 1),
                                                            activation=None)
            self.avgpoolingConv2D0 = tf.keras.layers.Conv2D(input_shape[3] * 2 // self.r, kernel_size=(1, 1),
                                                            activation=None)

        self.maxpoolingConv2D1 = tf.keras.layers.Conv2D(input_shape[3] * 2, kernel_size=(1, 1), activation=None)
        self.avgpoolingConv2D1 = tf.keras.layers.Conv2D(input_shape[3] * 2, kernel_size=(1, 1), activation=None)

        self.split = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))

        super(CCAttention, self).__init__()

    def build(self, input_shape):
        self.max_kernel1 = self.add_weight(
            name='max_kernel',
            shape=[1, 1, int(input_shape[-1]) * 2]
        )
        self.max_kernel2 = self.add_weight(
            name='max_kernel',
            shape=[1, 1, int(input_shape[-1]) * 2]
        )
        self.avg_kernel1 = self.add_weight(
            name='avg_kernel',
            shape=[1, 1, int(input_shape[-1]) * 2]
        )
        self.avg_kernel2 = self.add_weight(
            name='avg_kernel',
            shape=[1, 1, int(input_shape[-1]) * 2]
        )
        super(CCAttention, self).build(input_shape)

    def call(self, real, imag):
        concat = self.concat([real, imag])

        maxpooling = self.maxpooling2D(concat)
        avgpooling = self.avgpooling2D(concat)

        maxpooling = self.maxpoolingConv2D0(maxpooling)
        avgpooling = self.avgpoolingConv2D0(avgpooling)

        maxpooling = self.maxpoolingConv2D1(maxpooling)
        avgpooling = self.avgpoolingConv2D1(avgpooling)

        maxpooling = tf.multiply(self.max_kernel1, maxpooling)
        avgpooling = tf.multiply(self.avg_kernel1, avgpooling)

        maxpooling = tf.keras.activations.relu(maxpooling)
        avgpooling = tf.keras.activations.relu(avgpooling)

        maxpooling = tf.multiply(self.max_kernel2, maxpooling)
        avgpooling = tf.multiply(self.avg_kernel2, avgpooling)

        maxpooling = tf.keras.activations.sigmoid(maxpooling)
        avgpooling = tf.keras.activations.sigmoid(avgpooling)

        Gc = tf.add(maxpooling, avgpooling)

        real_Gc, imag_Gc = self.split(Gc)

        real = tf.multiply(real, real_Gc)
        imag = tf.multiply(imag, imag_Gc)

        return real, imag

    def get_config(self):
        config = super().get_config()
        config.update({'r': self.r})
        return config


class CSAttention(tf.keras.layers.Layer):
    def __init__(self, r, input_shape):
        self.r = r

        self.permute_encoder = tf.keras.layers.Permute((3, 1, 2))
        self.reshape_encoder = tf.keras.layers.Reshape((input_shape[3], input_shape[1] * input_shape[2]))
        self.real_maxpooling1D = tf.keras.layers.MaxPooling1D(pool_size=input_shape[3])
        self.imag_maxpooling1D = tf.keras.layers.MaxPooling1D(pool_size=input_shape[3])
        self.real_avgpooling1D = tf.keras.layers.AvgPool1D(pool_size=input_shape[3])
        self.imag_avgpooling1D = tf.keras.layers.AvgPool1D(pool_size=input_shape[3])
        self.concat0 = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))
        self.concat1 = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))
        self.reshape_decoder = tf.keras.layers.Reshape((4, input_shape[1], input_shape[2]))
        self.permute_decoder = tf.keras.layers.Permute((2, 3, 1))
        self.conv2D = tf.keras.layers.Conv2D(2, kernel_size=(7, 7), padding='same', activation=None)
        self.split = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))

        super(CSAttention, self).__init__()

    def build(self, input_shape):
        super(CSAttention, self).build(input_shape)

    def call(self, real, imag):
        real_transpose = self.permute_encoder(real)
        imag_transpose = self.permute_encoder(imag)

        real_reshape = self.reshape_encoder(real_transpose)
        imag_reshape = self.reshape_encoder(imag_transpose)

        real_maxpooling = self.real_maxpooling1D(real_reshape)
        imag_maxpooling = self.imag_maxpooling1D(imag_reshape)
        real_avgpooling = self.real_avgpooling1D(real_reshape)
        imag_avgpooling = self.imag_avgpooling1D(imag_reshape)

        maxpooling = self.concat0([real_maxpooling, imag_maxpooling])
        avgpooling = self.concat0([real_avgpooling, imag_avgpooling])
        concat = self.concat1([maxpooling, avgpooling])

        concat = self.reshape_decoder(concat)
        concat = self.permute_decoder(concat)

        concat = self.conv2D(concat)
        Gc = tf.keras.activations.sigmoid(concat)

        real_Gc, imag_Gc = self.split(Gc)

        real = tf.multiply(real, real_Gc)
        imag = tf.multiply(imag, imag_Gc)

        return real, imag

    def get_config(self):
        config = super().get_config()
        config.update({'r': self.r})
        return config
