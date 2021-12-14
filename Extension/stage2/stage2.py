import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import data.transforms_bbox as Tr
from data.voc import VOC_box
from models.ClsNet import Labeler, pad_for_grid
from utils.densecrf import DENSE_CRF
from PIL import Image
from tqdm import tqdm

class VOCDataLoader():
    def __init__(self,cfg):
        self.transforms= Tr.Normalize_Caffe()
        self.cfg=cfg 
        self.dataset=VOC_box(self.cfg,self.transforms)
    @ property
    def num_classes(self) -> int:
        return self.cfg.DATA.NUM_CLASSES 

    def get_dataloader(self,batch_size=1):
        return DataLoader(self.dataset,batch_size=batch_size)

class generate_PSEUDOLABELS():
    def __init__(self,cfg):
        self.model=Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE).cuda()
        self.cfg=cfg
        self.load_weights(f"{cfg.MODEL.WEIGHTS}")  #load pre-trained weights
        self.classifier_weights=torch.clone(self.model.classifier.weight.data)
    
    def DENSE_CRF(self):
        bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std = self.cfg.MODEL.DCRF
        return DENSE_CRF(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)
    
    def CHECK_SAVE_PSEUDO_LABLES(self):
        save_paths = []
        if self.cfg.SAVE_PSEUDO_LABLES:
         folder_name = os.path.join(self.cfg.DATA.ROOT,self.cfg.NAME)
         if not os.path.isdir(folder_name):
             os.mkdir(folder_name)
         save_paths = []
         for txt in ("Y_crf", "Y_ret", "Y_crf_u0"):
             sub_folder = folder_name + f"/{txt}"
             if not os.path.isdir(sub_folder):
                 os.mkdir(sub_folder)
             save_paths += [os.path.join(sub_folder, "{}.png")]
        return save_paths  
    
    def get_features(self,img):
        self.features = self.model.get_features(img.cuda()) # Output from the model backbone
        self.features = F.interpolate(self.features, img.shape[-2:], mode='bilinear', align_corners=True)
        self.padded_features = pad_for_grid(self.features, self.cfg.MODEL.GRID_SIZE)
        self.normed_f = F.normalize(self.features)
    
    def normed_bg_p(self,bg_mask):
        padded_bg_mask = pad_for_grid(bg_mask.cuda(),self.cfg.MODEL.GRID_SIZE)
        grid_bg, valid_gridIDs = self.model.get_grid_bg_and_IDs(padded_bg_mask,self.cfg.MODEL.GRID_SIZE)
        bg_protos = self.model.get_bg_prototypes(self.padded_features, padded_bg_mask, grid_bg,self.cfg.MODEL.GRID_SIZE)
        bg_protos = bg_protos[0,valid_gridIDs] # (1,GS**2,dims,1,1) --> (len(valid_gridIDs),dims,1,1)
        normed_bg_p = F.normalize(bg_protos)
        return normed_bg_p

    def get_bg_attn(self,normed_bg_p):
        bg_attns = F.relu(torch.sum(normed_bg_p*self.normed_f, dim=1))
        bg_attn = torch.mean(bg_attns, dim=0, keepdim=True) # (len(valid_gridIDs),H,W) --> (1,H,W)
        bg_attn[bg_attn <self.cfg.MODEL.BG_THRESHOLD * bg_attn.max()] = 0
        Bg_unary = torch.clone(bg_mask[0]) # (1,H,W)
        region_inside_bboxes = Bg_unary[0]==0 # (H,W)
        Bg_unary[:,region_inside_bboxes] = bg_attn[:,region_inside_bboxes].detach().cpu()
        return region_inside_bboxes,Bg_unary

    def get_Fg_CAMS(self,gt_labels,bboxes):
        Fg_unary = []
        for uni_cls in gt_labels:
            w_c =self.classifier_weights[uni_cls][None]
            raw_cam = F.relu(torch.sum(w_c*self.features, dim=1)) # (1,H,W)
            normed_cam = torch.zeros_like(raw_cam)
            for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                denom = raw_cam[:,hmin:hmax,wmin:wmax].max() + 1e-12
                normed_cam[:,hmin:hmax,wmin:wmax] = raw_cam[:,hmin:hmax,wmin:wmax] / denom
            Fg_unary += [normed_cam]
        Fg_unary = torch.cat(Fg_unary, dim=0).detach().cpu()
        return Fg_unary

    def get_BG_CAMS(self,gt_labels,bboxes):
        w_c_bg =self.classifier_weights[0][None]
        raw_cam_bg = F.relu(torch.sum(w_c_bg*self.features,dim=1)) # (1,H,W)
        normed_cam_bg = raw_cam_bg.clone().detach()
        for uni_cls in gt_labels:
            for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                normed_cam_bg[:,hmin:hmax,wmin:wmax] = 0
        normed_cam_bg = (normed_cam_bg / normed_cam_bg.max()).detach().cpu()
        return normed_cam_bg    
    
    def get_unary(self,Bg_unary, Fg_unary,region_inside_bboxes,rgb_img):
        unary = torch.cat((Bg_unary, Fg_unary), dim=0)
        unary[:,region_inside_bboxes] = torch.softmax(unary[:,region_inside_bboxes], dim=0)
        refined_unary = dCRF.inference(rgb_img, unary.numpy())
        return refined_unary
    
    def get_Y_crf(self,refined_unary,refined_unary_u0,gt_labels):
        tmp_mask = refined_unary.argmax(0)
        tmp_mask_u0 = refined_unary_u0.argmax(0)
        Y_crf = np.zeros_like(tmp_mask, dtype=np.uint8)
        Y_crf_u0 = np.zeros_like(tmp_mask_u0, dtype=np.uint8)
        for idx_cls, uni_cls in enumerate(gt_labels,1):
            Y_crf[tmp_mask==idx_cls] = uni_cls
            Y_crf_u0[tmp_mask_u0==idx_cls] = uni_cls
        Y_crf[tmp_mask==0] = 0
        Y_crf_u0[tmp_mask_u0==0] = 0
        return Y_crf,Y_crf_u0
    
    def get_corr_maps(self,gt_labels,Y_crf,region_inside_bboxes,bboxes):
        tmp_Y_crf = torch.from_numpy(Y_crf) # (H,W)
        gt_labels_with_Bg = [0] + gt_labels.tolist()
        corr_maps = []
        for uni_cls in gt_labels_with_Bg:
            indices = tmp_Y_crf==uni_cls
            if indices.sum():
                normed_p = F.normalize(self.features[...,indices].mean(dim=-1))   # (1,dims)
                corr = F.relu((self.normed_f*normed_p[...,None,None]).sum(dim=1)) # (1,H,W)
            else:
                normed_w = F.normalize(self.classifier_weights[uni_cls][None])
                corr = F.relu((self.normed_f*normed_w).sum(dim=1)) # (1,H,W)
            corr_maps.append(corr)
        corr_maps = torch.cat(corr_maps) # (1+len(gt_labels),H,W)
            
        # (Out of bboxes) reset Fg correlations to zero
        for idx_cls, uni_cls in enumerate(gt_labels_with_Bg):
            if uni_cls == 0:
                corr_maps[idx_cls, ~region_inside_bboxes] = 1
            else:
                mask = torch.zeros(img_H,img_W).type_as(corr_maps)
                for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                    mask[hmin:hmax,wmin:wmax] = 1
                corr_maps[idx_cls] *= mask
        return corr_maps

    def get_Y_ret(self,corr_maps,gt_labels):
        tmp_mask = corr_maps.argmax(0).detach().cpu().numpy()
        Y_ret = np.zeros_like(tmp_mask, dtype=np.uint8)
        for idx_cls, uni_cls in enumerate(gt_labels,1):
            Y_ret[tmp_mask==idx_cls] = uni_cls
        Y_ret[tmp_mask==0] = 0
        return Y_ret

    def forward(self,dataloader):
        self.model.eval()
        dCRF = self.DENSE_CRF()
        trainset=dataloader.dataset
        with torch.no_grad():
            for it, (img, bboxes, bg_mask) in enumerate(tqdm(dataloader.get_dataloader())):
                '''
                img     : (1,3,H,W) float32
                bboxes  : (1,K,5)   float32
                bg_mask : (1,H,W)   float32
                '''

                fn = trainset.filenames[it]
                rgb_img = np.array(Image.open(trainset.img_path.format(fn))) # RGB input image
                bboxes = bboxes[0] # (1,K,5) --> (K,5) bounding boxes
                bg_mask = bg_mask[None] # (1,H,W) --> (1,1,H,W) background mask
                img_H,img_W = img.shape[-2:]
                norm_H, norm_W = (img_H-1)/2, (img_W-1)/2
                bboxes[:,[0,2]] = bboxes[:,[0,2]]*norm_W + norm_W
                bboxes[:,[1,3]] = bboxes[:,[1,3]]*norm_H + norm_H
                bboxes = bboxes.long()
                gt_labels = bboxes[:,4].unique()
                
                self.get_features(img)
                normed_bg_p=self.normed_bg_p(bg_mask)

                # Background attention maps (u0)
                region_inside_bboxes,Bg_unary=self.get_bg_attn(normed_bg_p)
            
                # CAMS for foreground classes (uc)
                Fg_unary=self.get_Fg_CAMS(gt_labels,bboxes)

                # CAMS for background classes (ub)
                normed_cam_bg=self.get_BG_CAMS(gt_labels,bboxes)

                # Final unary by concatinating foreground and background unaries
                refined_unary=self.get_unary(Bg_unary, Fg_unary,region_inside_bboxes,rgb_img)

                # Unary witout background attn
                refined_unary_u0=self.get_unary(normed_cam_bg, Fg_unary,region_inside_bboxes,rgb_img)
            
                # (Out of bboxes) reset Fg scores to zero
                for idx_cls, uni_cls in enumerate(gt_labels,1):
                    mask = np.zeros((img_H,img_W))
                    for wmin,hmin,wmax,hmax,_ in bboxes[bboxes[:,4]==uni_cls]:
                        mask[hmin:hmax,wmin:wmax] = 1
                    refined_unary[idx_cls] *= mask
                    refined_unary_u0[idx_cls] *= mask

                # Y_crf and Y_crf_u0
                Y_crf,Y_crf_u0=self.get_Y_crf(refined_unary,refined_unary_u0,gt_labels)

                # Y_ret
                corr_maps=self.get_corr_maps(gt_labels,Y_crf,region_inside_bboxes,bboxes)
                Y_ret=self.get_Y_ret(self,corr_maps,gt_labels)

                paths=self.CHECK_SAVE_PSEUDO_LABLES()
                if paths:
                    for pseudo, save_path in zip([Y_crf, Y_ret, Y_crf_u0],path):
                        Image.fromarray(pseudo).save(save_path.format(fn)) 
    
    def load_weights(self,path):
        self.model.load_state_dict(torch.load(path), strict=False)


    








    
    