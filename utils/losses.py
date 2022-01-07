import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

def get_loss_ce(y_pred, ycrf,yret,num_classes):
    '''
    y_pred and y_crf are batches
    '''
    n_classes_arr=torch.from_numpy(np.arange(num_classes)).to('cuda')
    s_class = ( ycrf[:,:,:,None] ==n_classes_arr) & ( yret[:,:,:,None] ==n_classes_arr )
    s_class= torch.permute(s_class,(0,3,1,2))  
    denom=torch.sum(s_class)
    num=torch.sum(torch.log(y_pred[s_class]))
    return -num/denom 

def get_loss_wce(y_pred,ycrf,yret,feature_map,classifier_weight,num_classes,gamma):
     
    cos=torch.nn.CosineSimilarity(dim=2, eps=1e-6)  # 1024
    n_classes_arr=torch.from_numpy(np.arange(num_classes)).to('cuda')
    classifier_weight= classifier_weight.unsqueeze(0)  # shape  = (1,num_classes,1024,...,...)
    feature_map= feature_map.unsqueeze(1)              # shape = (batchsize,1,1024,...,...)
    print("f-",feature_map.shape)
    print("c-",classifier_weight.shape)
    correlation_map = torch.zeros((feature_map.shape[0],num_classes,feature_map.shape[2], feature_map.shape[3])).cuda() 
    correlation_map[:,:num_classes,...] = 1 + cos(feature_map,classifier_weight)  # shape- batchsize,num_classes,...,...

    # Confidence Map
    correlation_map_cstar = torch.zeros((feature_map.shape[0],feature_map.shape[2], feature_map.shape[3])).cuda()  
    idx = (ycrf[:,:,:,None] == n_classes_arr)    # will be of shape batchsize,...,...,num_classes
    idx= torch.permute(idx,(0,3,1,2))  # we want batchsize, num_classes,...,...
    for i in range(num_classes):
      correlation_map_cstar[idx[:,i,:,:]] = correlation_map[idx] 
    confidence_map = (correlation_map_cstar / torch.max(correlation_map, dim=1).values) ** gamma 
        
    # Final Loss
    not_s_class = torch.logical_not(( ycrf[:,:,:,None] ==n_classes_arr) & (yret[:,:,:,None] ==n_classes_arr))  # size- batchsize,...,...,num_classes
    not_s_class = torch.permute(not_s_class,(0,3,1,2))  # batchsize,num_classes,...,... 
    t = confidence_map[not_s_class[:,n_classes_arr,:,:]]
    denom = torch.sum(t)
    numer = torch.sum( t * torch.log(y_pred[not_s_class]))
    return -numer/denom