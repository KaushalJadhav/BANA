import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import data.transforms_bbox as Tr
from data.voc import VOC_box
from models.ClsNet import Labeler
from utils.BgMaskfromBoxes import VOC_BgMaskfromBoxes

def custom_collate(batch):
    '''
    This is to assign a batch-wise index for each box.
    Args:
         batch (torch.Tensor): batch of Dataloader
    Returns:  
             sample (Dict): Dictionary #CHECK 
    '''
    sample = {}
    img = []
    bboxes = []
    bg_mask = []
    batchID_of_box = []
    for batch_id, item in enumerate(batch):
        img.append(item[0])
        bboxes.append(item[1]) 
        bg_mask.append(item[2])
        for _ in range(len(item[1])):
            batchID_of_box += [batch_id]
    sample["img"] = torch.stack(img, dim=0)
    sample["bboxes"] = torch.cat(bboxes, dim=0)
    sample["bg_mask"] = torch.stack(bg_mask, dim=0)[:,None]
    sample["batchID_of_box"] = torch.tensor(batchID_of_box, dtype=torch.long)
    return sample


class VOCDataModule(pl.LightningDataModule):
    '''
    VOC DataModule
    Generates train and validation dataloaders for VOC dataset
    Args:
         cfg: namespace of config file variables  
    '''
    def __init__(self,cfg):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # Check whether to save hparams 
        # self.save_hyperparameters(logger=False)      
        # data transformations
        self.transforms = Tr.Compose([
            Tr.RandomScale(0.5, 1.5),
            Tr.ResizeRandomCrop(cfg.DATA.CROP_SIZE), 
            Tr.RandomHFlip(0.5), 
            Tr.ColorJitter(0.5,0.5,0.5,0),
            Tr.Normalize_Caffe(),
        ])
        self.cfg = cfg 

    @ property
    def num_classes(self) -> int:
        '''
        Returns:
                Number of classes
        '''
        return self.cfg.DATA.NUM_CLASSES 

    def setup(self,stage=None):
        '''
        setup data for loading it into dataloaders 
        '''
        print("Generating Background masks")
        VOC_BgMaskfromBoxes(self.cfg.DATA.ROOT)
        self.train_dataset = VOC_box(self.cfg, self.transforms, True)
        self.val_dataset = VOC_box(self.cfg, self.transforms, False)

    def train_dataloader(self):
        ''' train dataloader'''
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.DATA.BATCH_SIZE,
            collate_fn=custom_collate,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True 
        )

    def val_dataloader(self):
        ''' validation dataloader'''
        return DataLoader(
            self.val_dataset, 
            batch_size=self.cfg.DATA.BATCH_SIZE,
            collate_fn=custom_collate,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True 
        )


class LabelerLitModel(pl.LightningModule):
    '''
    Lightning extension of the Pytorch based model.
    Args:
         cfg: namespace of config file variables 
    '''
    def __init__(self,cfg):
        super().__init__()
        self.model = Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE)
        self.cfg = cfg
        self.params = self.model.get_params()
        self.criterion = nn.CrossEntropyLoss()  # CE loss used
        self.backbone = self.model.backbone
        self.save_hyperparameters()           # to automatically log hyperparameters to W&B
        self.load_weights(f"./weights/{cfg.MODEL.WEIGHTS}")  # loading pre-trained weights

    def training_step(self, batch, batch_idx):
        '''
        Train step.
        Args:
             batch (torch.Tensor): batch of Dataloader
             batch_idx (int): batch index
        Returns:
                loss_dict (Dict): Train loss
        '''                      
        loss = self.step(batch)
        loss_dict={"train_loss":loss}
        self.log_dict(
            loss_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss_dict

    def validation_step(self, batch, batch_idx):
        '''
        Validation step.
        Args:
             batch (torch.Tensor): batch of Dataloader
             batch_idx (int): batch index
        Returns:
                loss_dict (Dict): Validation loss
        '''                    
        loss = self.step(batch)
        loss_dict={"val_loss":loss}
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss_dict

    def step(self,sample):
        '''
        Common step used to calculate train loss and val loss.
        Args:
                sample (Dict): Dictionary #CHECK
        Returns:
                loss
        '''
        img = sample["img"]
        bboxes = sample["bboxes"]
        bg_mask = sample["bg_mask"]
        batchID_of_box = sample["batchID_of_box"]
        ind_valid_bg_mask = bg_mask.mean(dim=(1,2,3)) > 0.125 # This is because VGG16 has output stride of 8.
        logits = self.model(img, bboxes, batchID_of_box, bg_mask, ind_valid_bg_mask)
        logits = logits[...,0,0]
        fg_t = bboxes[:,-1][:,None].expand(bboxes.shape[0], np.prod(self.cfg.MODEL.ROI_SIZE))
        fg_t = fg_t.flatten().long()
        target = torch.zeros(logits.shape[0], dtype=torch.long,device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        target[:fg_t.shape[0]] = fg_t
        loss = self.criterion(logits, target)
        return loss  

    def configure_optimizers(self):
        '''
        Defines learning rate scheduler and optimizer
        Returns: 
                 Dict: {"optimizer": optimizer,"lr_scheduler":lr_scheduler} 
        '''
        lr = self.cfg.SOLVER.LR
        wd = self.cfg.SOLVER.WEIGHT_DECAY

        optimizer = optim.SGD(
            [
                {"params":self.params[0], "lr":lr,    "weight_decay":wd},
                {"params":self.params[1], "lr":2*lr,  "weight_decay":0 },
                {"params":self.params[2], "lr":10*lr, "weight_decay":wd},
                {"params":self.params[3], "lr":20*lr, "weight_decay":0 }
            ], 
            momentum=self.cfg.SOLVER.MOMENTUM
        )

        lr_scheduler = {
            "scheduler": optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=self.cfg.SOLVER.MILESTONES, 
                gamma=0.1
                ),
            # 'epoch' updates the scheduler on epoch end whereas 'step' updates it after a optimizer update.
            "interval": "step",
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "train_loss", 
            "strict": True,
            "name": None,
        }

        return { 
            "optimizer": optimizer,
            "lr_scheduler":lr_scheduler 
        }
    
    def load_weights(self,path):
        '''
        loads state_dict from given path
        Args:
             path: path of file containing pre-trained weight.
             (PyCaffe and VGG-16 ImageNet pretrained weights can be downloaded from-
             [vgg16_20M.caffemodel] (http://liangchiehchen.com/projects/Init%20Models.html))
        '''
        self.backbone.load_state_dict(torch.load(path), strict=False)