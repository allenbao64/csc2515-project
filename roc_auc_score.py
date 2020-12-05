import numpy as np
from sklearn.metrics import roc_auc_score


Y_val = np.array([1,1,1,0,0,0,0,0,0,0])   # ground truth label in validation set;  3 ones, 7 zeros
pred  = np.array([0,1,1,0,0,0,1,0,0,0])   # prediction:  1 false-negative at position[0];  1 false-positive at position[6]
classifier_auc_score = roc_auc_score(Y_val, pred)

print("classifier_auc_score: {}".format(classifier_auc_score))  # 0.7619

## below is tensorflow code if you use TF
#tf.keras.metrics.AUC(name='roc_auc')
