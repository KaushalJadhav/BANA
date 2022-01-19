from pytorch_lightning.loggers import WandbLogger
try:
    import wandb
except ModuleNotFoundError:
    pass

def wandblogging(cfg,model):
    '''
    logs data to wandb.
    Args:
         cfg   : namespace of config file variables
         model (LightningModule): model to train/eval
    Returns:
         wandb_logger (WandbLogger): logger that logs data to Weights and Biases
    '''
    if cfg.LOGGER.RESUME:
        resume='allow'
    else:
        resume=None
        
    wandb_logger = WandbLogger(
        name=cfg.LOGGER.NAME,
        project=cfg.LOGGER.PROJECT, # group runs in the specified project
        log_model='all', # log all new checkpoints during training
        id=cfg.LOGGER.ID,  # run id, necessary for resuming
        resume=resume # if cfg.LOGGER.RESUME=True,resumes run else overwrites previous run if exists
    )    
    wandb_logger.watch(model,log='all')  # logs histogram of gradients and parameters
    return wandb_logger 

def get_logger(cfg,model=None):
    '''
    Args:
         cfg   : namespace of config file variables
         model (LightningModule): model to train/eval. 
                                  Required for wandblogging. 
                                  Default:None
    Returns:
         logger #CHECK
    ''' 
    if not cfg.LOGGER.LOGGING:
        return cfg.LOGGER.LOGGING
    else:
        if cfg.LOGGER.TYPE.lower()=='wandb':
            return wandblogging(cfg,model)
        # Default logger is TensorBoardLogger
        return cfg.LOGGER.LOGGING
