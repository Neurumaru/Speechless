import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

from config import valid_metrics, fs
from tools_for_estimator import pesq, stoi


###############################################################################
#                             Custom TensorBoard                              #
###############################################################################
class AudioTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self,
                 validation_data,
                 log_dir='logs',
                 batch_size=5,
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,
                 write_steps_per_second=False,
                 update_freq='epoch',
                 profile_batch=2,
                 embeddings_freq=0,
                 embeddings_metadata=None,
                 **kwargs):
        super().__init__(log_dir,
                         histogram_freq,
                         write_graph,
                         write_images,
                         write_steps_per_second,
                         update_freq,
                         profile_batch,
                         embeddings_freq,
                         embeddings_metadata,
                         **kwargs)
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.batch_size = batch_size

    def set_model(self, model):
        super().set_model(model)
        self._writer = tf.summary.create_file_writer(self.log_dir + '/validation')

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)
        batch = self.create_datasets().batch(self.batch_size)
        progbar = tf.keras.utils.Progbar(len(batch), stateful_metrics=['loss'], unit_name='step')
        pesq_scores = []
        stoi_scores = []
        for b in batch:
            y_valid, y_pred = b[:, 0, :], b[:, 1, :]
            values = []
            # need to connect every valid_metrics_list
            if 'pesq' in valid_metrics:
                pesq_score = pesq(y_valid, y_pred)
                pesq_scores.append(pesq_score)
                values.append(('val_pesq', np.nanmean(pesq_scores)))
            if 'stoi' in valid_metrics:
                stoi_score = stoi(y_valid, y_pred)
                stoi_scores.append(stoi_score)
                values.append(('val_stoi', np.nanmean(stoi_scores)))
            progbar.add(1, values=values)

        X_valid, y_valid = self.validation_data
        y_pred = self.model.predict(X_valid[0:3])
        X_valid = tf.cast(tf.reshape(X_valid[0:3], (1, fs * 3, 1)), dtype=tf.float32)
        y_valid = tf.cast(tf.reshape(y_valid[0:3], (1, fs * 3, 1)), dtype=tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, (1, fs * 3, 1)), dtype=tf.float32)

        with summary_ops_v2.record_if(True):
            with self._val_writer.as_default():
                if 'pesq' in valid_metrics:
                    summary_ops_v2.scalar('epoch_pesq', np.nanmean(pesq_scores), step=epoch)
                if 'stoi' in valid_metrics:
                    summary_ops_v2.scalar('epoch_stoi', np.nanmean(stoi_scores), step=epoch)
                if epoch % 5 == 0:
                    summary_ops_v2.audio('clean_target_wav', y_valid, 16000, max_outputs=1, step=epoch)
                    summary_ops_v2.audio('estimated_wav', y_pred, 16000, max_outputs=1, step=epoch)
                    summary_ops_v2.audio('mixed_wav', X_valid, 16000, max_outputs=1, step=epoch)

    def create_datasets(self):
        X_valid, y_valid = self.validation_data
        y_pred = self.model.predict(X_valid)
        y_valid, y_pred = np.expand_dims(y_valid, axis=1), np.expand_dims(y_pred, axis=1)
        data = np.concatenate((y_valid, y_pred), 1)
        return tf.data.Dataset.from_tensor_slices(data)
