# Speechless
2021-2 Yonsei Speech Enhancement Challenge   
   
Member: Seon-Jun Kim, Hee-Young Ahn, Ji-Hoon Choi, Su-Hwan Myeong    
   
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

## Dataset
Clean Dataset: 1680    
Noise Dataset: 298   
![image](https://user-images.githubusercontent.com/20028521/147527612-f898c91a-1088-4a1d-a285-a2e11b76470b.png)   
![image](https://user-images.githubusercontent.com/20028521/147527619-c3cf6d68-604f-4719-8d49-119f7d6d076d.png)   
![image](https://user-images.githubusercontent.com/20028521/147527625-596fe3d1-7561-42a5-be8e-106dc6b6d3a2.png)


## Getting Started
1. Install the necessary libraries   
[Anaconda3](https://www.anaconda.com/products/individual)     
[CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)     
[cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)   
- Set environment
```sh
# Run on anaconda prompt after install anaconda
# If anaconda wasn't installed, download on https://www.anaconda.com/products/individual
conda create –n speech_enhancement python=3.7
conda activate speech_enhancement
pip install tensorflow-gpu=2.7.0 tensorboard=2.7.0 pesq pystoi
```
- Install CUDA 11.2
![image](https://user-images.githubusercontent.com/20028521/147528419-ddce05ce-0889-4fd5-b4a6-4881c78752ea.png)     

- Copy cuDNN to CUDA 11.2 Directory   
![image](https://user-images.githubusercontent.com/20028521/147528464-f1234f4b-758f-4993-8091-93dfe2f6d0d7.png)   
![image](https://user-images.githubusercontent.com/20028521/147528469-8e187bb1-dbca-434d-80d9-f97c6da709d3.png)   


2. Make a dataset for train and validation
```sh
# Run
generate_noisy_data.py [Mode] [SNR] [fs]

# Example)
generate_noisy_data.py train 5,10,15 16000
```
![image](https://user-images.githubusercontent.com/20028521/147527755-8486dfef-fca6-4c7d-bf5a-32179cd14902.png)   
![image](https://user-images.githubusercontent.com/20028521/147527796-65db40e6-cfa0-4313-b02e-d912d29b41b7.png)

```sh
# Run
generate_numpy_data.py
```
![image](https://user-images.githubusercontent.com/20028521/147527759-ba5ed57f-89fd-4b74-a605-4f1e2c792a80.png)   

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
![image](https://user-images.githubusercontent.com/20028521/147527809-40d57f12-d6da-4370-86a5-3d3800e41538.png)   
![image](https://user-images.githubusercontent.com/20028521/147527810-b3ac479e-439a-4c6e-af12-1526a8ab4625.png)


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
![image](https://user-images.githubusercontent.com/20028521/147527821-59769513-0728-4169-89bc-04f5e12baae5.png)


5. Make model at [models.py](https://github.com/Neurumaru/Speechless/blob/main/models.py)
```sh
# If you need to another layer, add at tools_for_model.py
```
![image](https://user-images.githubusercontent.com/20028521/147527858-3ef270fd-6d4d-4ef6-84df-6554460e9143.png)

6. Run [train_interface.py](https://github.com/Neurumaru/Speechless/blob/main/train_interface.py) and check on tensorboard
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
![image](https://user-images.githubusercontent.com/20028521/147527827-7db4d306-efaa-4281-8dd2-8a9ce433fab9.png)
![image](https://user-images.githubusercontent.com/20028521/147527893-3238a7df-f452-4961-a97a-e9e1317db0d5.png)
![image](https://user-images.githubusercontent.com/20028521/147527894-06c10f81-fcf3-44c4-bf53-a523b41ac7f0.png)

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
