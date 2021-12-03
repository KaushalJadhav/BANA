import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import data.transforms_bbox as Tr
from data.voc import VOC_box
from models.ClsNet import Labeler


def my_collate(batch):
    '''
    This is to assign a batch-wise index for each box.
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

    def __init__(self, cfg):
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
        self.dataset = None
        self.cfg = cfg 

    @ property
    def num_classes(self) -> int:
        return self.cfg.DATA.NUM_CLASSES

    def setup(self):
        self.train_dataset = VOC_box(self.cfg, self.transforms, is_train=True)
        self.val_dataset = VOC_box(self.cfg, self.transforms, is_train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.DATA.BATCH_SIZE,
            collate_fn=my_collate,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True 
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.cfg.DATA.BATCH_SIZE,
            collate_fn=my_collate,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True 
        )


class LabelerLitModel(pl.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self.model=Labeler(cfg.DATA.NUM_CLASSES, cfg.MODEL.ROI_SIZE, cfg.MODEL.GRID_SIZE)
        self.cfg=cfg
        self.params = self.model.get_params()
        self.criterion = nn.CrossEntropyLoss()
        self.interval_verbose = self.cfg.SOLVER.MAX_ITER // 40
        self.backbone=self.model.backbone
        self.val_losses = []                               # can replace by floats
        self.avgval_losses = []
        self.train_losses = []
        self.avgtrain_losses = []
        self.save_hyperparameters()           # to automatically log hyperparameters to W&B
        # load checkpoint 
        # need to see whether the function also works on .pt files
        self.load_weights(f"./weights/{cfg.MODEL.WEIGHTS}")  # Just loading pre-trained weights

    def training_step(self, batch, batch_idx):
        if self.cfg.DATA.MODE == "train":
         sample=train_batch                      # Need to check whether validation and training to be done at the same time
         loss = common_step(sample)
         self.train_losses.append(loss.item())
         result=pl.TrainResult(loss)
         return result

    def validation_step(self, batch, batch_idx):
        if self.cfg.DATA.MODE == "val":
             #to do... right now kept same as train
             sample=val_batch                      # Need to check whether validation and training to be done at the same time
             loss = common_step(sample)
             self.val_losses.append(loss.item())
             result=pl.EvalResult(loss)
             return result

    def common_step(sample):
        img = sample["img"]
        bboxes = sample["bboxes"]
        bg_mask = sample["bg_mask"]
        batchID_of_box = sample["batchID_of_box"]
        ind_valid_bg_mask = bg_mask.mean(dim=(1,2,3)) > 0.125 # This is because VGG16 has output stride of 8.
        logits = self.model(img, bboxes, batchID_of_box, bg_mask, ind_valid_bg_mask)
        logits = logits[...,0,0]
        fg_t = bboxes[:,-1][:,None].expand(bboxes.shape[0], np.prod(self.cfg.MODEL.ROI_SIZE))
        fg_t = fg_t.flatten().long()
        target = torch.zeros(logits.shape[0], dtype=torch.long)
        target[:fg_t.shape[0]] = fg_t
        loss = self.criterion(logits, target)
        return loss 

    def validation_epoch_end(self, outputs):
        self.avgval_losses.append(sum(self.val_losses) / len(self.val_losses))
        self.val_losses=[]
        if self.current_epoch()+1 % self.interval_verbose ==0:
            # log
            self.log("val-average-loss",sum(self.avgval_losses) / len(self.avgval_losses))
            self.avgval_losses=[]

    def training_epoch_end(self, outputs):
        self.avgtrain_losses.append(sum(self.train_losses) / len(self.train_losses))
        self.train_losses=[]
        if self.current_epoch()+1 % self.interval_verbose ==0:
            # log
            self.log("train-average-loss",sum(self.avgtrain_losses) / len(self.avgtrain_losses))
            self.avgtrain_losses=[] 
    
    def test_step(self, batch, batch_idx):
        if self.cfg.DATA.MODE == "val":
             #to do... right now kept same as validation
             sample=val_batch                      # Need to check this
             loss = common_step(sample)
             result=pl.EvalResult(loss)
             # ADD LOGGER-
             return result
    
    def test_epoch_end(self, outputs):
        # to do
        pass

    def configure_optimizers(self):
        lr = self.cfg.SOLVER.LR
        wd = self.cfg.SOLVER.WEIGHT_DECAY
        optimizer = optim.SGD(
        [{"params":self.params[0], "lr":lr,    "weight_decay":wd},
         {"params":self.params[1], "lr":2*lr,  "weight_decay":0 },
         {"params":self.params[2], "lr":10*lr, "weight_decay":wd},
         {"params":self.params[3], "lr":20*lr, "weight_decay":0 }], 
        momentum=self.cfg.SOLVER.MOMENTUM
        )
        lr_scheduler = {
        "scheduler": optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.SOLVER.MILESTONES, gamma=0.1),
        # 'epoch' updates the scheduler on epoch end whereas 'step' updates it after a optimizer update.
        "interval": "step",
        "frequency": 1,
        # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        "monitor": "train_loss" if self.cfg.DATA.MODE == "train" else "val_loss",
        "strict": True,
        "name": None,
        }
        return { "optimizer": optimizer,"lr_scheduler":lr_scheduler }
    
    def load_weights(self,path):
        self.backbone.load_from_checkpoint(path)