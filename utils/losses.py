import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def get_loss_ce(y_pred, ycrf,yret,num_classes):
    '''
    y_pred and y_crf are batches
    '''
    n_classes_arr=torch.from_numpy(np.arange(num_classes+1))
    s_class = ( ycrf[:,:,:,None] ==n_classes_arr) & ( yret[:,:,:,None] ==n_classes_arr )
    denom=torch.sum(s_class)
    num=torch.sum(torch.log(y_pred[s_class]))
    return -num/denom 

def get_loss_wce(y_pred,ycrf,yret,feature_map,classifier_weight,num_classes,gamma):

    cos=torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    n_classes_arr=torch.from_numpy(np.arange(num_classes))
    # Correlation Map (CHECK BATCH)
    correlation_map = torch.zeros((num_classes+1,feature_map.shape[1], feature_map.shape[2])).cuda()  # No change
    correlation_map[:num_classes] = 1 + cos(feature_map[None,...], classifier_weight[:num_classes])   # 3d 

    # Confidence Map
    correlation_map_cstar = torch.zeros((feature_map.shape[1], feature_map.shape[2])).cuda()
    idx = (ycrf[:,:,None] == n_classes_arr)
    idx= torch.permute(idx,(2,0,1))
    correlation_map_cstar[idx[i] for i in range(num_classes)] = correlation_map[idx] # check
    confidence_map = (correlation_map_cstar / torch.max(correlation_map, dim=0).values) ** gamma  # 2d
        
    # Final Loss
    not_s_class = torch.logical_not(( ycrf[:,:,None] ==n_classes_arr) & (yret[:,:,None] ==n_classes_arr))
    not_s_class = torch.permute(not_s_class,(2,0,1))
    t = confidence_map[not_s_class[i] for i in range(num_classes) ]
    denom = torch.sum(t)
    numer = torch.sum( t * torch.log(y_pred[not_s_class]))
    return -numer/denom