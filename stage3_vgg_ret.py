import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import data.transforms_seg as Trs
from data.voc import VOC_seg
from configs.defaults import _C

from models.SegNet import DeepLab_LargeFOV
from models.loss import NoiseAwareLoss

from tqdm import tqdm
import wandb
from utils.wandb import wandb_log_seg, init_wandb

def main(cfg):    
    if cfg.SEED:
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)

    tr_transforms = Trs.Compose([
        Trs.RandomScale(0.5, 1.5),
        Trs.ResizeRandomCrop(cfg.DATA.CROP_SIZE), 
        Trs.RandomHFlip(0.5), 
        Trs.ColorJitter(0.5,0.5,0.5,0),
        Trs.Normalize_Caffe(),
    ])

    trainset = VOC_seg(cfg, tr_transforms)
    train_loader = DataLoader(trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    model = DeepLab_LargeFOV(cfg.DATA.NUM_CLASSES, is_CS=True).cuda()
    criterion = nn.CrossEntropyLoss()
    
    params = model.get_params()
    lr = cfg.SOLVER.LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    optimizer = optim.SGD(
        [{"params":params[0], "lr":lr,    "weight_decay":wd},
         {"params":params[1], "lr":2*lr,  "weight_decay":0 },
         {"params":params[2], "lr":10*lr, "weight_decay":wd},
         {"params":params[3], "lr":20*lr, "weight_decay":0 }], 
        lr=lr,
        weight_decay=wd,
        momentum=cfg.SOLVER.MOMENTUM
    ) # learning rate and weight decay is kept same for all the layers 
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES, gamma=0.1)

    # Initializing W&B
    init_wandb(model, cfg)

    # Load pretrained model from wandb if present

    model.train()
    iterator = iter(train_loader)

    for it in tqdm(range(1, cfg.SOLVER.MAX_ITER+1)):
        try:
            sample = next(iterator)
        except:
            iterator = iter(train_loader)
            sample = next(iterator)
        img, masks = sample # VOC_seg dataloader returns image and the corresponing (pseudo) label
        ycrf, yret = masks

        img = img.to('cuda')
        yret = yret.to('cuda').long()
        
        img_size = img.size()
        logit = model(img, (img_size[2], img_size[3]))
        loss = criterion(logit, yret)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging Loss and LR on wandb
        wandb_log_seg(loss.item(), optimizer.param_groups[0]["lr"], it)

        # Save model locally and then on wandb
        save_dir = './ckpts/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), save_dir + 'checkpoint.pth')
        wandb.save(save_dir + 'checkpoint.pth')
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    main(cfg)