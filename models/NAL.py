import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class NoiseAwareLoss(nn.Module):

    def __init__(self, num_classes, gamma, lambda_wgt):
        super(NoiseAwareLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.lambda_wgt = lambda_wgt
        self.soft_max = nn.Softmax(dim=1)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.n_classes_arr=torch.arange(self.num_classes).cuda()

    def forward(self, y_pred, ycrf, yret, feature_map, classifier_weight):
        y_pred = self.soft_max(y_pred)
        loss_ce = self.get_loss_ce(y_pred, ycrf, yret)
        loss_wce = self.get_loss_wce(y_pred, ycrf, yret, feature_map, classifier_weight)
        total_loss = loss_ce + self.lambda_wgt * loss_wce
        return total_loss

    def get_s_class(self,ycrf,yret):
        s_class = (ycrf[:,:,:,None] == self.n_classes_arr) & (yret[:,:,:,None] == self.n_classes_arr)
        s_class = torch.permute(s_class, (0, 3, 1, 2))
        return s_class
    
    def get_correlation_map(self,feature_map,classifier_weight):
        correlation_map = torch.zeros((feature_map.shape[0], self.num_classes, 41, 41)).cuda()
        for i in range(self.num_classes):
            correlation_map[:,i,...] = 1 + self.cosine_similarity(feature_map[:,...], classifier_weight[i])
        correlation_map = F.interpolate(correlation_map, (321,321), mode='bilinear', align_corners=False)
        return correlation_map
    
    def get_confidence_map(self,ycrf,feature_map, classifier_weight):
        correlation_map = self.get_correlation_map(feature_map,classifier_weight)
        correlation_map_cstar = torch.zeros((feature_map.shape[0], 321, 321)).cuda()
        idx = (ycrf[:,:,:,None] == self.n_classes_arr)    
        idx = torch.permute(idx, (0, 3, 1, 2))  
        for i in range(self.num_classes):
            t = idx[:,i,:,:]
            correlation_map_cstar[t] = correlation_map[:,i,:,:][t] 
        return (correlation_map_cstar / torch.max(correlation_map, dim=1).values) ** self.gamma

    def get_loss_ce(self, y_pred, ycrf, yret):
        s_class = self.get_s_class(ycrf,yret)
        denom = torch.sum(s_class)
        num = torch.sum(torch.log(y_pred[s_class]))
        return -num/denom 

    def get_loss_wce(self, ypred, ycrf, yret, feature_map, classifier_weight):
        confidence_map = self.get_confidence_map(self,ycrf,feature_map, classifier_weight)
        denom=numer=0
        not_s_class = torch.logical_not(self.get_s_class(ycrf,yret))  
        for i in range(self.num_classes):
            t = not_s_class[:,i,:,:]
            denom += torch.sum(confidence_map[t])
            numer += torch.sum(confidence_map[t] * torch.log(ypred[:,i,:,:][t]))
        return -numer/denom