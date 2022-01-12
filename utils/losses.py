import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

# def cosinesimilarity(x1,x2,dim=0,eps=1e-6):
#   # x1_norm=torch.linalg.norm(x1,dim=dim,ord=2)
#   # x2_norm=torch.linalg.norm(x2,dim=dim,ord=2)
#   F.normalize(x1,p=2.0,dim=dim,eps=eps,out=None)
#   F.normalize(x2,p=2.0,dim=dim,eps=eps,out=None)
#   # denom=max(x1_norm*x2_norm,eps)
#   # return (x1*x2).sum(dim=dim)/denom
#   return (x1*x2).sum(dim=dim)

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
     
    cos=torch.nn.CosineSimilarity(dim=1, eps=1e-6)  # 1024
    n_classes_arr=torch.from_numpy(np.arange(num_classes)).to('cuda')
    feature_map=F.interpolate(feature_map,(321,321),mode='bilinear',align_corners=False)
   
    correlation_map = torch.zeros((feature_map.shape[0],num_classes,321,321)).cuda() 
    # Confidence Map
    correlation_map_cstar = torch.zeros((feature_map.shape[0],321,321)).cuda()
    
    for i in range(num_classes):
      correlation_map[:,i,...] = 1 +cos(feature_map,classifier_weight[i,...].unsqueeze(0))

    idx = (ycrf[:,:,:,None] == n_classes_arr)    # will be of shape batchsize,...,...,num_classes
    idx= torch.permute(idx,(0,3,1,2))  # we want batchsize, num_classes,...,...
  
    for i in range(num_classes):
      t=idx[:,i,:,:]
      correlation_map_cstar[t] = correlation_map[:,i,:,:][t] 
    confidence_map = (correlation_map_cstar / torch.max(correlation_map, dim=1).values) ** gamma 

    # Final Loss
    denom=0
    numer=0
    not_s_class = torch.logical_not(( ycrf[:,:,:,None] ==n_classes_arr) & (yret[:,:,:,None] ==n_classes_arr))  # size- batchsize,...,...,num_classes
    not_s_class = torch.permute(not_s_class,(0,3,1,2))  # batchsize,num_classes,...,...

    # No need of for loop if the following line works
    # t = confidence_map[:,None,:,:].expand(-1,21,-1,-1)[not_s_class[:,n_classes_arr,:,:]]
    for i in range(num_classes):
      t= not_s_class[:,i,:,:]
      denom += torch.sum( confidence_map[t] )
      numer += torch.sum( confidence_map[t] * torch.log(y_pred[:,i,:,:][t]))
    # denom = torch.sum(t)
    # numer = torch.sum( t * torch.log(y_pred[not_s_class]))
    return -numer/denom