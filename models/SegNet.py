import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Layers import VGG16, RES101, ASPP
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


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

# Shift to loss.py 
class NoiseAwareLoss(nn.Module):

    def __init__(self, num_classes, gamma, lambda_wgt,tau):
        super(NoiseAwareLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.lambda_wgt = lambda_wgt
        self.tau=tau
        self.n_classes_arr=torch.from_numpy(np.arange(num_classes)).to('cuda')
        self.cos=nn.CosineSimilarity(dim=1, eps=1e-6)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self,ycrf, yret,img,model):
        img = img.to('cuda')
        ycrf = ycrf.to('cuda').long()
        yret = yret.to('cuda').long()
        
        feature_map = model.get_features(img)
        classifier_weight = torch.clone(model.classifier.weight.data)
        feature_map=F.interpolate(feature_map,(ycrf.shape[1],ycrf.shape[2]),mode='bilinear',align_corners=False)
        H = self.get_H(feature_map,classifier_weight)
        loss_ce = self.ce_loss(H,ycrf,yret)
        loss_wce = self.wce_loss(feature_map,classifier_weight,H,ycrf,yret)
        total_loss=loss_ce + self.lambda_wgt * loss_wce

        ycrf=ycrf.detach().cpu()
        yret=yret.detach().cpu()

        return total_loss,loss_ce, loss_wce
    
    def get_cosine_similarity(self,feature_map,classifier_weight):
        cos_sim = torch.zeros((feature_map.shape[0],self.num_classes,feature_map.shape[2],feature_map.shape[3])).cuda()
        for i in range(self.num_classes):
            cos_sim[:,i,...] = self.cos(feature_map,classifier_weight[i,...].unsqueeze(0)) # 1024
        return cos_sim
    
    def get_H(self,feature_map,classifier_weight):
        return self.soft_max(self.tau*self.get_cosine_similarity(feature_map,classifier_weight))

    def get_idx(self,ycrf,yret=None,neg=False):
        if yret is None:
            idx = (ycrf[:,:,:,None] == self.n_classes_arr)    # will be of shape batchsize,...,...,num_classes
        else:
            idx = ( ycrf[:,:,:,None] ==self.n_classes_arr) & (yret[:,:,:,None] ==self.n_classes_arr)
            if neg:
                idx= torch.logical_not(idx)
        idx= torch.permute(idx,(0,3,1,2)) # we want batchsize, num_classes,...,...
        return idx 

    def ce_loss(self,H,ycrf,yret):
        s_class = self.get_idx(ycrf,yret) 
        denom=torch.sum(s_class)
        num=torch.sum(torch.log(H[s_class]))
        return -num/denom 
    
    def wce_loss(self,feature_map,classifier_weight,H,ycrf,yret):
        numer=denom=0
        not_s_class =self.get_idx(ycrf,yret,neg=True) 
        confidence_map = self.get_confidence_map(feature_map,classifier_weight,ycrf)
        # No need of for loop if the following line works
        # t = confidence_map[:,None,:,:].expand(-1,21,-1,-1)[not_s_class[:,n_classes_arr,:,:]]

        for i in range(self.num_classes):
           t= not_s_class[:,i,:,:]
           denom += torch.sum( confidence_map[t])
           numer += torch.sum( confidence_map[t] * torch.log(H[:,i,:,:][t]))
        return -numer/denom
    
    def get_correlation_map(self,feature_map,classifier_weight):
        return 1+self.get_cosine_similarity(feature_map,classifier_weight)
    
    def get_confidence_map(self,feature_map,classifier_weight,ycrf):
        correlation_map_cstar = torch.zeros((feature_map.shape[0],feature_map.shape[2],feature_map.shape[3])).cuda()
        correlation_map = self.get_correlation_map(feature_map,classifier_weight)
        idx = self.get_idx(ycrf,yret=None,neg=False)
        for i in range(self.num_classes):
            t=idx[:,i,:,:]
            correlation_map_cstar[t] = correlation_map[:,i,:,:][t] 
        return (correlation_map_cstar / torch.max(correlation_map, dim=1).values) ** self.gamma

class CrossEntropyLoss(nn.Module):
    def __init__(self,ignore_index=255):
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self,y,img,model):
        # Forward pass
        img = img.to('cuda')
        img_size = img.size()
        logit = model(img, (img_size[2], img_size[3]))
        y= y.to('cuda').long()
        return self.criterion(logit,y)