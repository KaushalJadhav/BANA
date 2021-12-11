import cv2
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
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, y_pred, ycrf, yret, feature_map, classifier_weight):
        loss_ce = self.get_loss_ce(y_pred, ycrf, yret)
        loss_wce = self.get_loss_wce(y_pred, ycrf, yret, feature_map, classifier_weight)
        return loss_ce + self.lambda_wgt * loss_wce, loss_ce, loss_wce

    def get_loss_ce(self, y_pred, ycrf, yret):
        denom = 0.0
        numer = 0.0
        for i in range(self.num_classes):
          s_class = ( ycrf == i ) & ( yret == i )
          denom += torch.sum(s_class)
          numer += torch.sum(torch.log(y_pred[i][s_class]))
        return -numer/denom

    def get_loss_wce(self, y_pred, ycrf, yret, feature_map, classifier_weight):

        # Correlation Map
        correlation_map = torch.zeros((self.num_classes+1, feature_map.shape[1], feature_map.shape[2])).cuda()
        for i in range(self.num_classes):
          correlation_map[i] = 1 + self.cos(feature_map, classifier_weight[i])

        # Confidence Map
        correlation_map_cstar = torch.zeros((feature_map.shape[1], feature_map.shape[2])).cuda()
        for i in range(self.num_classes):
          idx = ( ycrf == i )
          correlation_map_cstar[idx] = correlation_map[i][idx]
        confidence_map = ( correlation_map_cstar / torch.max(correlation_map, dim=0).values ) ** self.gamma
        
        # Final Loss
        denom = 0.0
        numer = 0.0
        for i in range(self.num_classes):
          not_s_class = torch.logical_not( ( ycrf == i ) & ( yret == i ) )
          denom += torch.sum( confidence_map[not_s_class] )
          numer += torch.sum( confidence_map[not_s_class] * torch.log(y_pred[i][not_s_class]) )
        return -numer/denom