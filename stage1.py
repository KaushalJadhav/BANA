from pytorch_lightning import Trainer
import os
import sys
from utils.util import seed,process_cfg,checkpoint_callback_stage1
from extension.stage1.Lightningextension import VOCDataModule,LabelerLitModel
from utils.logging import get_logger,log_lr
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

    checkpoint_callback=checkpoint_callback_stage1(cfg)
    lr_monitor = log_lr()

    resume=None
    if args.resume is not "None":
        resume=f"{cfg.MODEL.SAVE_DIR}/{args.resume}"
    elif cfg.LOGGER.LOGGING and cfg.LOGGER.TYPE.lower()=='wandb':
            if cfg.LOGGER.CHECKPOINT is not None:
                f_path = wandb.restore(os.path.join(cfg.MODEL.SAVE_DIR,cfg.LOGGER.CHECKPOINT))
                resume = f_path.name 

    # train/eval the model

    trainer = Trainer(
    logger=get_logger(cfg,model=model),
    # if cfg.LOGGER.LOGGING=True logging will be enabled, else disabled
    deterministic=True, 
    #sets whether PyTorch operations must use deterministic algorithms.
    max_steps=cfg.SOLVER.MAX_ITER,
    max_epochs=cfg.SOLVER.MAX_EPOCH,
    enable_checkpointing=cfg.MODEL.SAVING, 
    # if cfg.MODEL.SAVING=True checkpointing will be enabled, else disabled
    callbacks=[checkpoint_callback,lr_monitor],
    # log data every training step
    log_every_n_steps=1,
    gpus=[int(args.gpu_id)],
    resume_from_checkpoint=resume
    )

    trainer.fit(model,datamodule=datamodule)
    if cfg.LOGGER.LOGGING and cfg.LOGGER.TYPE.lower()=='wandb':
        wandb.finish()
