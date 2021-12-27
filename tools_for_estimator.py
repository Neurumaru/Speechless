import numpy as np
from pesq import pesq as py_pesq
from pystoi import stoi as py_stoi


############################################################################
#                                   PESQ                                   #
############################################################################
# https://github.com/ludlows/python-pesq
def cal_pesq(dirty_wavs, clean_wavs):
    scores = []
    for i in range(len(dirty_wavs)):
        pesq_score = py_pesq(16000, dirty_wavs[i], clean_wavs[i], 'wb', 1)
        if pesq_score > 0:
            scores.append(pesq_score)
    return scores


def pesq(y_true, y_pred):
    pesq_score = cal_pesq(y_pred.numpy(), y_true.numpy())
    if len(pesq_score) == 0:
        return 0
    return np.nanmean(pesq_score)


###############################################################################
#                                     STOI                                    #
###############################################################################
def cal_stoi(estimated_speechs, clean_speechs):
    stoi_scores = []
    for i in range(len(estimated_speechs)):
        stoi_score = py_stoi(clean_speechs[i], estimated_speechs[i], 16000, extended=False)
        stoi_scores.append(stoi_score)
    return stoi_scores


def stoi(y_true, y_pred):
    stoi_score = cal_stoi(y_pred.numpy(), y_true.numpy())
    if len(stoi_score) == 0:
        return 0
    return np.nanmean(stoi_score)
