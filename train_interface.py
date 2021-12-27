###############################################################################
#                           Warning Message Ignore                            #
###############################################################################
import os
import warnings
import absl.logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
absl.logging.set_verbosity(absl.logging.ERROR)

###############################################################################
#                               Import modules                                #
###############################################################################
import config as cfg
from dataloader import load_data
from models import model
from tools_for_loss import sdr, si_snr, si_sdr, pmsqe_loss, pmsqe_log_mse_loss, stsa_mse, pmsqe_si_snr_loss
from tools_for_estimator import pesq, stoi, cal_pesq
from tools_for_tensorboard import AudioTensorBoard

import time
from contextlib import redirect_stdout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np

###############################################################################
#                                Load Datasets                                #
###############################################################################
X_train, y_train = load_data('train')
X_valid, y_valid = load_data('valid')

###############################################################################
#         Parameter Initialization and Setting for model training             #
###############################################################################
if cfg.loss == 'sdr':
    cfg.loss = sdr
elif cfg.loss == 'si_snr':
    cfg.loss = si_snr
elif cfg.loss == 'si_sdr':
    cfg.loss = si_sdr
elif cfg.loss == 'pmsqe_loss':
    cfg.loss = pmsqe_loss(cfg.batch)
elif cfg.loss == 'pmsqe_log_mse_loss':
    cfg.loss = pmsqe_log_mse_loss(cfg.batch)
elif cfg.loss == 'stsa_mse':
    cfg.loss = stsa_mse
elif cfg.loss == 'pmsqe_si_snr_loss':
    cfg.loss = pmsqe_si_snr_loss(cfg.batch)

if cfg.optimizer == 'adam':
    cfg.optimizer = Adam(clipnorm=1.)

for i in range(len(cfg.metrics)):
    if cfg.metrics[i] == 'pesq':
        cfg.metrics[i] = pesq
    elif cfg.metrics[i] == 'stoi':
        cfg.metrics[i] = stoi

if cfg.dir_add_time:
    t = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    cfg.job_dir = cfg.job_dir + t
    cfg.logs_dir = cfg.logs_dir + t

atb = AudioTensorBoard(validation_data=(X_valid, y_valid),
                       log_dir=cfg.logs_dir,
                       batch_size=cfg.batch,
                       profile_batch=0)
cp = ModelCheckpoint(filepath=cfg.job_dir + cfg.job_file,
                     save_weights_only=True)

callbacks = []

if cfg.use_callbacks_metrics:
    callbacks.append(atb)
if cfg.model_checkpoint:
    callbacks.append(cp)

###############################################################################
#                                   Summary                                   #
###############################################################################

print()
model.summary()
print()
print('Models Save: ' + cfg.job_dir)
print('Logs Save: ' + cfg.logs_dir)
print()

if not os.path.exists(cfg.job_dir):
    os.makedirs(cfg.job_dir)
with open(cfg.job_dir + '/' + 'logs.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

###############################################################################
#                                   Compile                                   #
###############################################################################
model.compile(loss=cfg.loss,
              optimizer=cfg.optimizer,
              metrics=cfg.metrics,
              run_eagerly=True)

###############################################################################
#                                    Train                                    #
###############################################################################
model.load_weights('checkpoint-011.ckpt')

model.fit(X_train, y_train,
          epochs=cfg.max_epochs,
          initial_epoch=cfg.start_epochs,
          validation_data=(X_valid, y_valid),
          batch_size=cfg.batch,
          callbacks=callbacks,
          workers=-1,
          use_multiprocessing=True)

model.save_weights(cfg.job_dir + '/ckpt')