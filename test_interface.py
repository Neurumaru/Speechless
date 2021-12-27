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
from dataloader import load_data
from models import model
from tools_for_estimator import cal_pesq

import numpy as np

###############################################################################
#                                Load Datasets                                #
###############################################################################
X_test, y_test = load_data('test')

###############################################################################
#                                     Test                                    #
###############################################################################
model.load_weights('checkpoint-011.ckpt')
y_pred = model.predict(X_test)
pesq_score = cal_pesq(y_pred, y_test)
print(f'pesq mean:{np.nanmean(pesq_score)}')

