import os
import collections
import numpy as np
import torch
import logging
from torch.utils.data import Dataset,DataLoader
import torchvision
from PIL import Image
from pycocotools.coco import COCO

class COCO_box(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        
        cat_ids = self.coco.getCatIds()
        img_ids = []
        for cat in cat_ids:
            img_ids.extend(self.coco.getImgIds(catIds=cat))   
        img_ids = list(set(img_ids))

        #REMOVE GRAY IMAGE IDs
        # gray_ids = set([431848,7888,205289,141671,274219,209222,24021,353180,61418,130465])
        # img_ids = list(set(img_ids) - gray_ids)
        print("Number of Images : ",len(img_ids))

        self.ids = [20992, 10583]

        cat_id_map = [-1]*91
        for i in range(0,80):
          cat_id_map[cat_ids[i]] = i+1
        self.cat_id_map = cat_id_map

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        
        # open the input image, convert to np array for transforms
        img = np.array(Image.open(os.path.join(self.root, path)),dtype=np.float32)

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        bboxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            cat_id = coco_annotation[i]['category_id']
            cls_num = self.cat_id_map[cat_id]
            bboxes.append([xmin, ymin, xmax, ymax,cls_num])
       
        #converted to np array for transforms
        bboxes = np.array(bboxes).astype('float32')
        
        #ADD MASK PATH
        #mask_path = '/kaggle/input/coco-train-bgmaskfromboxes/BgMaskfromBoxes_coco_train/kaggle/working/BgMaskfromBoxes'
        #mask_path = '/kaggle/input/coco-val-bgmaskfromboxes/kaggle/working/BgMaskfromBoxes'
        bg_mask = np.array(Image.open(os.path.join(mask_path,path[:-4]+'.png')), dtype=np.int32)
        
        if self.transforms is not None:
            img, bboxes, bg_mask = self.transforms(img, bboxes, bg_mask)

        return img, bboxes, bg_mask

    def filename(self,index):
      img_id = self.ids[index]
      path = self.coco.loadImgs(img_id)[0]['file_name']
      return path[:-4],os.path.join(self.root, path)

    def __len__(self):
        return len(self.ids)

    '''
    Output format as required for Transforms
    img    : (H, W, 3) numpy float32
    bboxes : (wmin, hmin, wmax, hmax, cls) N x 5 numpy float32
    bg_mask : (H, W) numpy int32
    '''