import os
import sys
from utils.util import seed,process_cfg
from extension.stage2.stage2 import VOCDataLoader,generate_PSEUDOLABELS

def stage2(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg=process_cfg(args.config_file)
    seed(cfg)
    dataloader=VOCDataLoader(cfg)
    model=generate_PSEUDOLABELS(cfg)
    model.forward(dataloader)
    


