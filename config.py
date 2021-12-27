#######################################################################
#                                 path                                #
#######################################################################
job_dir = 'models/'
job_file = '/checkpoint-{epoch:03d}.ckpt'
logs_dir = 'logs/'
dir_add_time = True


#######################################################################
#                         possible setting                            #
#######################################################################
loss_list = ['mse', 'sdr', 'si_snr', 'si_sdr', 'pmsqe_loss', 'pmsqe_log_mse_loss', 'stsa_mse', 'pmsqe_si_snr_loss']
optimizer_list = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
metrics_list = ['mae', 'mape', 'mse', 'msle', 'cosine_similarity', 'logcosh', 'pesq', 'stoi']
valid_metrics_list = ['pesq', 'stoi']


#######################################################################
#                           current setting                           #
#######################################################################
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