from utils.metric import Evaluator

import torch
import numpy as np
from tqdm import tqdm

def evaluate(cfg, data_loader, model):
    with torch.no_grad():
        model.eval()
        evaluator = Evaluator(cfg.DATA.NUM_CLASSES)
        evaluator.reset()
        for batch in tqdm(data_loader):
            img, masks = batch
            ygt = masks[0]
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
        return accuracy, iou