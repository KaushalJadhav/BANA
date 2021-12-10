import os
import sys
from utils.util import seed,process_cfg
from Extension.stage2.stage2 import VOCDataModule,LightningModel

def stage2(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg=process_cfg(args.config_file)
    seed(cfg)
    datamodule=VOCDataModule(cfg)
    # Note- datamodule.get_dataloader() will return Dataloader
    model=generate_PSEUDOLABELS(cfg)
    # Need to update this
    


