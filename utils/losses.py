import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def get_loss_ce(self, y_pred, ycrf,num_classes):
    '''
    y_pred and y_crf are batches
    '''
    denom = 0.0
    numer = 0.0
    n_classes_arr=torch.from_numpy(np.arange(num_classes+1))
    s_class = ( ycrf[:,:,:,None] ==n_classes_arr) & ( yret[:,:,:,None] ==n_classes_arr )
    denom=torch.sum(s_class)
    num=torch.sum(torch.log(y_pred[s_class]))
    return -num/denom 

