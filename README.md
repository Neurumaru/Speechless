# Speechless
2021-2 Speech Enhancement Project   
   
Seon-Jun Kim, Hee-Young Ahn, Ji-Hoon Choi,     
Supervised and helped by C-J Lee, S-R Hwang, and H-U Yoon

## Requirements
> This repository is tested on Window 10, and
- tensorflow 2.7.0
- CUDA 11.2
- cuDNN 8.1.1
- tensorboard 2.7.0
- numpy 1.20.3
- pesq 0.0.3
- pystoi 0.3.3

Getting Started
1. Install the necessary libraries
2. Make a dataset for train and validation
```sh
# The shape of the dataset
[data_num, 2 (inputs and targets), sampling_frequency * data_length]
```
3. Set [dataloader.py](https://github.com/Neurumaru/Speechless/blob/main/dataloader.py)
```sh
input = np.load('DATASET_FILE(NUMPY)')

# Example
input = np.load('train.npy')
```

4. Set [config.py](https://github.com/Neurumaru/Speechless/blob/main/config.py)
```sh
#######################################################################
#                           current setting                           #
#######################################################################
# Select loss function, optimatzer, metrics and valid_matrics
loss = loss_list[7]
optimizer = optimizer_list[4]
metrics = [metrics_list[0], metrics_list[2]]
valid_metrics = [valid_metrics_list[0], valid_metrics_list[1]]
use_callbacks_metrics = True
model_checkpoint = True

# hyper-parameters
start_epochs = 0
max_epochs = 100
batch = 10


#######################################################################
#                         model information                           #
#######################################################################
fs = 16000
frame_length = 400
frame_step = 100
fft_len = 1023
n_freqs = fft_len // 2 + 1
```

5. Make model at [models.py](https://github.com/Neurumaru/Speechless/blob/main/models.py)
```sh
# If you need to another layer, add at tools_for_model.py
```

6. Run [train_interface.py](https://github.com/Neurumaru/Speechless/blob/main/train_interface.py)
```sh
###############################################################################
#                                    Train                                    #
###############################################################################
# if you want to load weight, edit this code
# model.load_weights('checkpoint-011.ckpt')

model.fit(X_train, y_train,
          epochs=cfg.max_epochs,
          initial_epoch=cfg.start_epochs,
          validation_data=(X_valid, y_valid),
          batch_size=cfg.batch,
          callbacks=callbacks,
          workers=-1,
          use_multiprocessing=True)

model.save_weights(cfg.job_dir + '/ckpt')
```
7. Edit weights file and run [test_interface.py](https://github.com/Neurumaru/Speechless/blob/main/test_interface.py) for testing
```sh
###############################################################################
#                                     Test                                    #
###############################################################################
model.load_weights('weights.ckpt')
y_pred = model.predict(X_test)
pesq_score = cal_pesq(y_pred, y_test)
print(f'pesq mean:{np.nanmean(pesq_score)}')
```

## Run with colab
'[train_interface.ipynb](https://github.com/Neurumaru/Speechless/blob/main/train_interface.ipynb)' was made for running at colab.

## Model
CNN + LSTM + Attention Model
(Not full attention connect)

## Layer
- STFT
- Inverse STFT
- Complex Channel Attention
- Complex Spatial Attention

## Loss Function
- mse
- sdr
- si-snr
- si-sdr
- pmsqe
- pmsqe + log-mse
- pmsqe + si-snr
- stsa

## Reference
**MONAURAL SPEECH ENHANCEMENT WITH COMPLEX CONVOLUTIONAL BLOCK ATTENTION MODULE AND JOINT TIME FREQUENCY LOSSES**   
Shengkui Zhao, Trung Hieu Nguyen, Bin Ma   
[[arXiv]](https://arxiv.org/pdf/2102.01993v1.pdf)   
**Other**   
https://github.com/seorim0/DNN-based-Speech-Enhancement-in-the-frequency-domain
https://git.its.aau.dk/ZQ62WN/Speech_Enhancement_Loss
