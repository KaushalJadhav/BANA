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

from models.SegNet import DeepLab_LargeFOV, DeepLab_ASPP
from models.loss import NoiseAwareLoss
from models.lr_scheduler import PolynomialLR

from tqdm import tqdm
import wandb
from utils.wandb import wandb_log_seg, init_wandb
from utils.metric import Evaluator

def evaluate(cfg, train_loader, model):
    with torch.no_grad():
        model.eval()
        evaluator = Evaluator(cfg.DATA.NUM_CLASSES)
        evaluator.reset()
        for batch in tqdm(train_loader):
            img, masks = batch
            ygt, ycrf, yret = masks
            # Forward pass
            img = img.to('cuda')
            img_size = img.size()
            logit = model(img, (img_size[2], img_size[3]))
            pred = torch.argmax(logit, dim=1)
            pred = pred.cpu().detach().numpy()
            ygt = ygt.cpu().detach().numpy()
            evaluator.add_batch(ygt, pred)
        # Calculate final metrics
        accuracy = evaluator.MACU()
        iou = evaluator.MIOU()
        if cfg.WANDB.MODE: 
            # Log on WandB
            wandb.log({
                "Mean IoU": iou,
                "Mean Accuracy": accuracy
                })

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
    
    if cfg.WANDB.MODE: 
        init_wandb(cfg)

    trainset = VOC_seg(cfg, tr_transforms)
    train_loader = DataLoader(trainset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    if cfg.NAME == "SegNet_VGG":
        model = DeepLab_LargeFOV(cfg.DATA.NUM_CLASSES, is_CS=True).cuda()
    elif cfg.NAME == "SegNet_ASPP":
        model = DeepLab_ASPP(cfg.DATA.NUM_CLASSES, output_stride=None, sync_bn=False, is_CS=True).cuda()

    if cfg.MODEL.LOSS == "NAL":
        criterion = NoiseAwareLoss(cfg.DATA.NUM_CLASSES, cfg.MODEL.DAMP, cfg.MODEL.LAMBDA)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        
    model.backbone.load_state_dict(torch.load(f"./weights/{cfg.MODEL.WEIGHTS}"), strict=False)
    
    lr = cfg.SOLVER.LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    
    # Creating optimizer for the both models
    if cfg.NAME == "SegNet_VGG":
        params = model.get_params()
        optimizer = optim.SGD(
            [{"params":params[0], "lr":lr,    "weight_decay":wd},
             {"params":params[1], "lr":lr,  "weight_decay":0.0 },
             {"params":params[2], "lr":10*lr, "weight_decay":wd},
             {"params":params[3], "lr":10*lr, "weight_decay":0.0 }], 
            lr=lr,
            weight_decay=wd,
            momentum=cfg.SOLVER.MOMENTUM
        )
    elif cfg.NAME == "SegNet_ASPP":
        optimizer = optim.SGD(
        params=[
            {
                "params":model.get_1x_lr_params(),
                "lr":lr,
                "weight_decay":wd
            },
            {
                "params":model.get_10x_lr_params(),
                "lr":10*lr,
                "weight_decay":wd
            }
        ],
        lr=lr,
        weight_decay=wd,
        momentum=cfg.SOLVER.MOMENTUM
        )

    # Poly learning rate scheduler according to the paper 
    scheduler = PolynomialLR(optimizer, step_size=10, iter_max=cfg.SOLVER.MAX_ITER, power=cfg.SOLVER.GAMMA)
    curr_it = 0

    # Save model locally and then on wandb
    save_dir = './ckpts/'
    # Load pretrained model from wandb if present
    try:
        wandb_checkpoint = wandb.restore('ckpts/checkpoint.pth')    
        checkpoint = torch.load(wandb_checkpoint.name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        scheduler.load_state_dict(checkpoint['sched_state_dict'])
        curr_it = checkpoint['iter']
        print("WandB checkpoint Loaded with iteration: ", curr_it)
    except:
        print("WandB checkpoint not Loaded")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    iterator = iter(train_loader)

    for it in tqdm(range(curr_it+1, cfg.SOLVER.MAX_ITER+1)):
        try:
            sample = next(iterator)
        except:
            iterator = iter(train_loader)
            sample = next(iterator)
        img, masks = sample # VOC_seg dataloader returns image and the corresponing (pseudo) label
        ygt, ycrf, yret = masks
        
        model.train()

        # Forward pass
        img = img.to('cuda')
        ycrf = ycrf.to('cuda').long()
        yret = yret.to('cuda').long()
        ygt = ygt.to('cuda').long()
        img_size = img.size()
        logit = model(img, (img_size[2], img_size[3]))

        # Loss calculation
        if cfg.MODEL.LOSS == "NAL":
            feature_map = model.get_features(img.cuda())
            classifier_weight = torch.clone(model.classifier.weight.data)
            loss, loss_ce, loss_wce = criterion(logit[0], ycrf[0], yret[0], feature_map[0], classifier_weight)
        elif cfg.MODEL.LOSS == "CE_CRF":
            loss = criterion(logit, ycrf)
        elif cfg.MODEL.LOSS == "CE_RET":
            loss = criterion(logit, yret)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Update the learning rate using poly scheduler
        scheduler.step()

        train_loss = loss.item()
        # Logging Loss and LR on wandb
        wandb_log_seg(train_loss, optimizer.param_groups[0]["lr"], it)

        if it%1000 == 0 or it == cfg.SOLVER.MAX_ITER:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'sched_state_dict': scheduler.state_dict(),
                'iter': it,
            }
            torch.save(checkpoint, save_dir + 'checkpoint.pth')
            if cfg.WANDB.MODE: 
                wandb.save(save_dir + 'checkpoint.pth')
            # Evaluate the train_loader at this checkpoint
            evaluate(cfg, train_loader, model)
            

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