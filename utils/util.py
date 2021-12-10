import numpy as np
import random
import torch
from configs.defaults import _C
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os
from pytorch_lightning.utilities.seed import seed_everything

def seed(cfg):
    '''
    sets seed for pseudo-random number generators in: pytorch, numpy, python.random. 
    In addition,sets the following environment variables:
    1. PL_GLOBAL_SEED: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    2. PL_SEED_WORKERS: (optional) is set to 1 if workers=True.
    3. PYTHONHASHSEED: set equal to seed
    More information- https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html

    Args:
         cfg : namespace of config file arguments
    '''
    if cfg.SEED:
        os.environ["PYTHONHASHSEED"] = str(cfg.SEED)
        seed_everything(cfg.SEED)

def process_cfg(config_file):
    '''
    generates cfg (namespace of config file arguments) from given config_file 
    Args:
            config_file
    Returns:
            cfg: Namespace of config file arguments
    '''
    cfg = _C.clone()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg 

def get_args():
    ''' 
    get command line arguments
    Returns:
            Namespace of arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",type=str,help="path of config file ")
    parser.add_argument("--stage", type=str, default="1", help="select stage")
    parser.add_argument("--gpu-id", type=str, default="0", help="select a GPU index")
    parser.add_argument("--resume", type=str, default="None", help="filename of the checkpoint")
    parser.add_argument("--epoch", type=int, default=100, help="Number of maximum epochs.When set equal to -1, disables it.")
    parser.add_argument("--step", type=int, default=8000, help="Number of maximum steps.When set equal to -1, disables it.")
    parser.add_argument("--save_after_n_epochs", type=int, default=0, 
    help="Number of epochs after which model is to be saved.When set equal to 0 saving after epochs disabled.")
    parser.add_argument("--save_after_n_steps", type=int, default=0, 
    help="Number of steps after which model is to be saved.When set equal to 0 saving after steps disabled.")
    return parser.parse_args()

def checkpoint_callback_stage1(cfg,n_epoch,n_step):
    '''
    Custom callback for saving checkpoint
    Args:
         cfg: namespace of config file variables
         n_epoch: number of epochs between checkpoints. If equal to zero, saving after epochs disabled.
         n_step: number of training steps between checkpoints. If equal to zero, saving after training steps disabled.
    Returns:
         checkpoint_callback
    '''
    if not cfg.MODEL.SAVING:   # Disable saving
        return None
    if epoch==0 and step==0:
        return None 
    return ModelCheckpoint(
        dirpath=f"{cfg.NAME}",                        # directory path to save checkpoints
        filename='{epoch}-{step}-{train_loss:.2f}',   
        save_last=True,                               # save last checkpoint
        save_top_k =1,                                # save the best checkpoint
        monitor='train/loss'                          # check train loss
        mode='min',                     # criteria for best checkpoint is minimum train loss
        every_n_epochs=n_epoch,              # if n_epoch is not None after epochs=n_epoch checkpoint saved.
        save_on_train_epoch_end=True if epoch not None,   
        #  if True checkpoint saved after train_epoch_end
        every_n_train_steps=n_step      # if n_step is not None after steps=n_step checkpoint saved.  
        )    