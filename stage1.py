from pytorch_lightning import Trainer
import os
import sys
from utils.util import seed,process_cfg,checkpoint_callback_stage1
from Extension.stage1.Lightningextension import VOCDataModule,LabelerLitModel
from utils.logging import get_logger
try:
    import wandb
except ModuleNotFoundError:
    pass

def stage1(args):
    '''
    Basic function to train models in stage-1
    Args:
         args: namespace of command line arguments
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # populate cfg namespace with config file variables
    cfg=process_cfg(args.config_file)
    # seed random number generators
    seed(cfg)
    if cfg.LOGGER.LOGGING and cfg.LOGGER.TYPE.lower()=='wandb':
         wandb.login()
    # load dataset
    if cfg.DATA.DATASET.lower()=="voc":
      datamodule=VOCDataModule(cfg)
    # create model
    model=LabelerLitModel(cfg)

    checkpoint_callback=checkpoint_callback_stage1(cfg,args.save_after_n_epochs,args.save_after_n_steps)
    if args.resume is not "None":
        resume=f"{cfg.MODEL.SAVE_DIR}/{args.resume}"
    else:
        resume=None 

    # train/eval the model

    trainer = Trainer(
    logger=get_logger(cfg,model=model),
    # if cfg.LOGGER.LOGGING=True logging will be enabled, else disabled
    deterministic=True, 
    #sets whether PyTorch operations must use deterministic algorithms.
    max_steps=min(cfg.SOLVER.MAX_ITER,args.step),
    max_epochs=args.epoch,
    enable_checkpointing=cfg.MODEL.SAVING 
    # if cfg.MODEL.SAVING=True checkpointing will be enabled, else disabled
    callbacks=checkpoint_callback,
    # log data every training step
    log_every_n_steps=1,
    gpus=[int(args.gpu_id)],
    resume_from_checkpoint=resume
    )

    trainer.fit(model,datamodule=datamodule)
    if cfg.LOGGER.LOGGING and cfg.LOGGER.TYPE.lower()=='wandb':
        wandb.finish()
