import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import data.transforms_seg as Trs
from data.voc import VOC_seg
from models.SegNet import DeepLab_LargeFOV, DeepLab_ASPP
from models.NAL import NoiseAwareLoss
from models.PolyScheduler import PolynomialLR
from utils.logging import log_eval
from utils.densecrf import DENSE_CRF
from utils.metric import Evaluator,scores

class VOCDataModule(pl.LightningDataModule):
    '''
    VOC DataModule
    Generates train and validation dataloaders for VOC dataset
    Args:
         cfg: namespace of config file variables  
    '''
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg 
        self.mode = self.cfg.DATA.MODE
        if self.mode == "val":
            self.BATCH_SIZE = 1
        else:
            self.BATCH_SIZE = self.cfg.DATA.BATCH_SIZE
        self.root_dir = self.cfg.DATA.ROOT
        self.num_workers = 4

    @ property
    def num_classes(self) -> int:
        '''
        Returns:
                Number of classes
        '''
        return self.cfg.DATA.NUM_CLASSES 

    def train_dataloader(self):
        if self.mode == "train_weak":
            ''' train dataloader'''
            tr_transforms = self.get_transforms()
            f_path = os.path.join(self.root_dir, "ImageSets/SegmentationAug","train_aug.txt")
            annot_folders = ["SegmentationClassAug",self.cfg.DATA.PSEUDO_LABEL_FOLDER]
            dataset = VOC_seg(self.root_dir,f_path,annot_folders,transforms=tr_transforms)
            return self.get_dataloader(dataset)
        return None 


    def val_dataloader(self):
        ''' validation dataloader'''
        if self.mode == "val":
            tr_transforms = self.get_transforms()
            f_path = os.path.join(self.root_dir, "ImageSets/Segmentation","val.txt")
            annot_folders = ["SegmentationClassAug"]
            dataset = VOC_seg(self.root_dir,f_path,annot_folders,transforms=tr_transforms)
            return self.get_dataloader(dataset)
        return None

    def test_dataloader(self):
        ''' test dataloader'''
        if self.mode == "test":
            tr_transforms = self.get_transforms()
            f_path = os.path.join(self.root_dir, "ImageSets/Segmentation","test.txt")
            annot_folders = None 
            dataset = VOC_seg(self.root_dir,f_path,annot_folders,transforms=tr_transforms)
            return self.get_dataloader(dataset)
        return None 
    
    def get_dataloader(self,dataset):
        '''
        dataset (torch.utils.data.Dataset) : dataset to be loaded in dataloader
        '''
        return DataLoader(
            dataset, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True 
        )
    
    def get_transforms(self):
        if self.mode == "train_weak":
            return Trs.Compose([
            Trs.RandomScale(0.5, 1.5),
            Trs.ResizeRandomCrop(cfg.DATA.CROP_SIZE), 
            Trs.RandomHFlip(0.5), 
            Trs.ColorJitter(0.5,0.5,0.5,0),
            Trs.Normalize_Caffe()
            ])
        if self.mode == "val":
            return Trs.Compose([Trs.Normalize_Caffe(),])
        return None 


class SegLitModel(pl.LightningModule):
    '''
    Lightning extension of the Pytorch based model.
    Args:
         cfg: namespace of config file variables 
    '''
    def __init__(self,cfg):
        self.cfg=cfg 
        self.num_classes = self.cfg.DATA.NUM_CLASSES
        self.name = cfg.NAME
        if self.name == "SegNet_VGG":
            self.model = DeepLab_LargeFOV(self.num_classes, is_CS=True)
        elif self.name == "SegNet_ASPP":
            self.model = DeepLab_ASPP(self.num_classes,output_stride=None,sync_bn=False,is_CS=True)
        
        self.mode = self.cfg.DATA.MODE
        
        # Load pre-trained backbone weights
        self.load_weights(f"./weights/{self.cfg.MODEL.WEIGHTS}")
        
        self.loss = self.cfg.MODEL.LOSS
        if self.loss == "NAL":
            self.criterion = NoiseAwareLoss(self.num_classes,self.cfg.MODEL.DAMP,self.cfg.MODEL.LAMBDA)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        if self.name == "SegNet_VGG":
            self.params = self.model.get_params()
        elif self.name == "SegNet_ASPP":
            self.params = [self.model.get_1x_lr_params(),self.model.get_10x_lr_params()]

        self.save_hyperparameters()           # to automatically log hyperparameters to W&B
        self.eval_interval = self.cfg.MODEL.EVAL_INTERVAL
        # Evaluator
        self.evaluator = Evaluator(self.num_classes)
        self.evaluator.reset()

        # Initialising Dense CRF object
        bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std = self.cfg.MODEL.DCRF
        self.dCRF = DENSE_CRF(bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std)

        if self.mode == "val":
            self.label_trues = []
            self.label_preds = []
        
    def training_step(self, batch, batch_idx):
        '''
        Train step.
        Args:
             batch (torch.Tensor): batch of Dataloader
             batch_idx (int): batch index
        Returns:
                loss_dict (Dict): Train loss
        ''' 
        img, masks = batch # VOC_seg dataloader returns image and the corresponing (pseudo) label 
        ygt, ycrf, yret = masks
        img_size = img.size()
        logit, feature_map = self.model(img,(img_size[2], img_size[3]))
        # Loss calculation
        if self.loss == "NAL":
            ycrf = ycrf.cuda().long()
            yret = yret.cuda().long()
            classifier_weight = torch.clone(self.model.classifier.weight.data)
            loss = self.criterion(logit, 
                             ycrf, 
                             yret, 
                             feature_map, 
                             classifier_weight)
            
        elif self.loss == "CE_CRF":
            ycrf = ycrf.long()
            loss = self.criterion(logit, ycrf)
        elif self.loss == "CE_RET":
            yret = yret.long()
            loss = self.criterion(logit, yret)

        loss_dict={"train_loss":loss}
        if self.cfg.LOGGER.LOGGING:
            self.log_dict(
                loss_dict,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        return loss_dict
    
    def on_train_epoch_end(self):
        '''
        Performs evaluation at the end of training epoch after each interval specified by self.cfg.MODEL.EVAL_INTERVAL
        '''
        if self.eval_interval is not None and self.eval_interval>0:
            if self.current_epoch+1 % self.eval_interval==0 :
                self.evaluator.reset()
                data_loader = VOCDataModule(self.cfg)
                with torch.no_grad():
                    self.model.eval()
                    for batch in data_loader.train_dataloader():
                        self.evaluate(batch,dcrf=False)
                    # Calculate final metrics
                    self.train_accuracy = self.evaluator.MACU()
                    self.train_iou = self.evaluator.MIOU()
                if self.cfg.LOGGER.LOGGING:
                    self.log_dict(
                        {
                            "Mean IoU": self.train_iou,
                            "Mean Accuracy": self.train_accuracy
                        },
                        on_step=True,
                        on_epoch=True,
                        prog_bar=True,
                        )
                print("Train Mean IoU at epoch={epoch} is {iou:.2f} ".format(epoch=self.current_epoch,iou=self.train_iou))
                print("Train Mean Accuracy at epoch={epoch} is {acc:.2f} ".format(epoch=self.current_epoch,acc=self.train_accuracy))

    def on_save_checkpoint(self,checkpoint):
        if self.eval_interval is not None and self.eval_interval>0:
            if self.current_epoch+1 % self.eval_interval !=0:
                self.evaluator.reset()
                data_loader = VOCDataModule(self.cfg)
                with torch.no_grad():
                    self.model.eval()
                    for batch in data_loader.train_dataloader():
                        self.evaluate(batch,dcrf=False)
                    # Calculate final metrics
                    self.train_accuracy = self.evaluator.MACU()
                    self.train_iou = self.evaluator.MIOU()
                self.model.train()
            checkpoint["Train_iou"] = self.train_iou
            checkpoint["Train Accuracy"] = self.train_accuracy

    def validation_step(self,batch,batch_idx):
        '''
        Validation step.
        Args:
             batch (torch.Tensor): batch of Dataloader
             batch_idx (int): batch index
        Returns:
                loss_dict (Dict): Validation loss
        '''  
        self.evaluate(batch,dcrf=True)                  
    
    def on_val_epoch_start(self):
        self.evaluator.reset()

    def on_val_epoch_end(self):
        accuracy = self.evaluator.MACU()
        iou = self.evaluator.MIOU()
        print("Validation Mean Accuracy ", accuracy)
        print("Validation Mean IoU ", iou)
        log_eval(key="Validation Mean Accuracy",value=accuracy)
        log_eval(key="Validation Mean IoU",value=iou)

        # Evaluating the validation dataloader after CRF post-processing
        results = scores(self.label_trues,self.label_preds,self.num_classes)
        crf_accuracy = results["Mean Accuracy"]
        crf_iou = results["Mean IoU"]
        print("CRF Validation Mean Accuracy ", crf_accuracy)
        print("CRF Validation Mean IoU ", crf_iou)
        log_eval(key="CRF Validation Mean Accuracy",value=crf_accuracy)
        log_eval(key="CRF Validation Mean IoU",value=crf_iou)

    def evaluate(self,batch,dcrf=False):
        img, masks = batch
        ygt = masks[0]
        # Forward pass
        img_size = img.size()
        logit, feature_map = self.model(img,(img_size[2], img_size[3]))
        pred = torch.argmax(logit, dim=1)
        pred = pred.cpu().detach().numpy()
        ygt = ygt.cpu().detach().numpy()
        self.evaluator.add_batch(ygt, pred)
        
        if dcrf:
            prob = F.softmax(logit, dim=1)[0].cpu().detach().numpy()
            img = img[0].cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)
            ygt = ygt[0].cpu().detach().numpy()
            # Apply DenseCRF
            prob = self.dCRF.inference(img, prob)
            label = np.argmax(prob, axis=0)
            # Append labels for evaluation
            self.label_preds.append(label)
            self.label_trues.append(ygt)

    def configure_optimizers(self):
        '''
        Defines learning rate scheduler and optimizer
        Returns: 
                 Dict: {"optimizer": optimizer,"lr_scheduler":lr_scheduler} 
        '''
        lr = self.cfg.SOLVER.LR
        wd = self.cfg.SOLVER.WEIGHT_DECAY
        momentum = self.cfg.SOLVER.MOMENTUM

        if self.name == "SegNet_VGG":
            optimizer = optim.SGD(
                [{"params":self.params[0], "lr":lr, "weight_decay":wd},
                {"params":self.params[1], "lr":lr, "weight_decay":0.0 },
                {"params":self.params[2], "lr":10*lr, "weight_decay":wd},
                {"params":self.params[3], "lr":10*lr, "weight_decay":0.0 }], 
                lr=lr,
                weight_decay= wd,
                momentum= momentum
        )
        elif cfg.NAME == "SegNet_ASPP":
            optimizer = optim.SGD(
                params=[
                    {
                        "params":self.params[0],
                        "lr":lr,
                        "weight_decay":wd
                    },
                    {
                        "params":self.params[1],
                        "lr":10*lr,
                        "weight_decay":wd
                    }
                    ],
        lr=lr,
        weight_decay=wd,
        momentum=momentum
        )
        
        scheduler = PolynomialLR(optimizer,
                                 step_size=self.cfg.SOLVER.STEP_SIZE, 
                                 iter_max=self.cfg.SOLVER.MAX_ITER, 
                                 power=self.cfg.SOLVER.GAMMA)
        lr_scheduler = {
            "scheduler": scheduler,
            # 'epoch' updates the scheduler on epoch end whereas 'step' updates it after a optimizer update.
            "interval": "step",
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "train_loss", 
            "strict": True,
            "name": "PolynomialLR_scheduler",
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
        '''
        state_dict = torch.load(path)
        # Manually matching the sate dicts if model and pretrained weights (only for res101)
        if self.name == "SegNet_ASPP":
            for key in list(state_dict.keys()):
                state_dict[key.replace('base.', '')] = state_dict.pop(key)
        self.model.backbone.load_state_dict(state_dict, strict=False)
