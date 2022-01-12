import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Layers import VGG16, RES101, ASPP
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from utils.losses import *


class DeepLab_LargeFOV(nn.Module):
    def __init__(self, num_classes, is_CS=False):
        super().__init__()
        self.backbone = VGG16(dilation=12)
        self.is_CS = is_CS
        if self.is_CS:
            self.temperature = 20 # Scale parameter for the CS based loss
        self.classifier = nn.Conv2d(1024, num_classes, 1, bias=False)
        self.from_scratch_layers = [self.classifier]
        
    def forward(self, x, img_size):
        return self.forward_classifier(self.get_features(x), img_size)
    
    def get_features(self, x):
        return self.backbone(x)
    
    def forward_classifier(self, x, img_size):
        if self.is_CS:
            normed_x = F.normalize(x)
            normed_w = F.normalize(self.classifier.weight)
            logits = F.conv2d(normed_x, normed_w)
            logits = F.interpolate(logits, img_size, mode='bilinear', align_corners=False)
            return self.temperature * logits
        else:
            logit = F.interpolate(self.classifier(x), img_size, mode='bilinear', align_corners=False)
            if self.training:
                return logit, x
            else:
                return logit
    
    def get_params(self):
        # pret_weight, pret_bias, scratch_weight, scratch_bias
        params = ([], [], [], [])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m in self.from_scratch_layers:
                    nn.init.normal_(m.weight, std=0.01)
                    params[2].append(m.weight)
                else:
                    params[0].append(m.weight)
                if m.bias is not None:
                    if m in self.from_scratch_layers:
                        nn.init.constant_(m.bias, 0)
                        params[3].append(m.bias)
                    else:
                        params[1].append(m.bias)
        return params


                            
class DeepLab_ASPP(nn.Module):
    def __init__(self, num_classes, output_stride, sync_bn, is_CS=True):
        super().__init__()
        self.backbone = RES101(sync_bn)
        self.is_CS = is_CS
        if self.is_CS:
            self.temperature = 20
        ndim = 256
        self.rates = [6, 12, 18, 24]
        bias = False 
        self.c1 = nn.Conv2d(2048, ndim, 3, 1, bias=bias, padding=self.rates[0], dilation=self.rates[0])
        self.c2 = nn.Conv2d(2048, ndim, 3, 1, bias=bias, padding=self.rates[1], dilation=self.rates[1])
        self.c3 = nn.Conv2d(2048, ndim, 3, 1, bias=bias, padding=self.rates[2], dilation=self.rates[2])
        self.c4 = nn.Conv2d(2048, ndim, 3, 1, bias=bias, padding=self.rates[3], dilation=self.rates[3])
        self.classifier = nn.Conv2d(ndim, num_classes, 1, bias=False)
        for m in [self.c1, self.c2, self.c3, self.c4, self.classifier]:
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
                
    def forward(self, x, img_size):
        return self.forward_classifier(self.get_features(x), img_size)
    
    def get_features(self, x):
        x = self.backbone(x)
        return F.relu(self.c1(x) + self.c2(x) + self.c3(x) + self.c4(x))
    
    def forward_classifier(self, x, img_size):
        if self.is_CS:
            normed_x = F.normalize(x)
            normed_w = F.normalize(self.classifier.weight)
            logits = F.conv2d(normed_x, normed_w)
            logits = F.interpolate(logits, img_size, mode='bilinear', align_corners=False)
            return self.temperature * logits
        else:
            return F.interpolate(self.classifier(x), img_size, mode='bilinear', align_corners=False)
        
    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) or isinstance(m[1], nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.c1, self.c2, self.c3, self.c4, self.classifier]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) or isinstance(m[1], nn.BatchNorm2d)):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

# Can keep this after testing 
class NoiseAwareLoss(nn.Module):

    def __init__(self, num_classes, gamma, lambda_wgt,batchsize=None):
        super(NoiseAwareLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.lambda_wgt = lambda_wgt
        self.batchsize=batchsize

    def forward(self, y_pred, ycrf, yret,img,model):
        feature_map = model.get_features(img.cuda())
        classifier_weight = torch.clone(model.classifier.weight.data)
        soft_max = nn.Softmax(dim=1)
        y_pred = soft_max(y_pred)
        loss_ce = get_loss_ce(y_pred, ycrf,yret,self.num_classes)
        loss_wce = get_loss_wce(y_pred,ycrf,yret,feature_map,classifier_weight,self.num_classes,self.gamma)
        total_loss=loss_ce + self.lambda_wgt * loss_wce
        return total_loss,loss_ce, loss_wce