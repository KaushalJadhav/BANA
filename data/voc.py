import pytorch_lightning as pl
import os
import collections
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset,DataLoader

# stage-1
class VOC_box(Dataset):
    '''
    loads VOC dataset 
    Args:
         cfg  : parameters loaded from config file 
         transforms (List): transforms to be applied to images
         is_train (bool): train dataset loaded if True else val dataset loaded. Default: True
    ''' 
    def __init__(self, cfg, transforms=None, is_train=True):
        print("loading dataset from"+cfg.DATA.ROOT)
        if is_train:
            if cfg.DATA.AUG:
                txt_name = "train_aug.txt"
            else:
                txt_name = "train.txt" 
        else:
            txt_name = "val.txt"

        if cfg.DATA.AUG:
            f_path = os.path.join(cfg.DATA.ROOT,"ImageSets/SegmentationAug", txt_name)
        else:
            f_path = os.path.join(cfg.DATA.ROOT, "ImageSets/Segmentation", txt_name)

        self.filenames  = [x.split('\n')[0] for x in open(f_path)]
        self.transforms = transforms
        self.img_path  = os.path.join(cfg.DATA.ROOT, 'JPEGImages/{}.jpg')
        self.xml_path  = os.path.join(cfg.DATA.ROOT, 'Annotations/{}.xml')
        self.mask_path = os.path.join(cfg.DATA.ROOT, 'BgMaskfromBoxes/{}.png')
        self.CLASSES=self.get_classes()
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        fn  = self.filenames[index]
        img = np.array(Image.open(self.img_path.format(fn)), dtype=np.float32) 
        bboxes  = self.load_bboxes(self.xml_path.format(fn))
        bg_mask = np.array(Image.open(self.mask_path.format(fn)), dtype=np.int64)
        if self.transforms is not None:
            img, bboxes, bg_mask = self.transforms(img, bboxes, bg_mask)
        return img, bboxes, bg_mask, self.filenames

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    def get_classes(self):
        '''
        Returns:
                 CLASSES (tuple): list of class names
        '''
        CLASSES = (
            "background", 
            "aeroplane", 
            "bicycle", 
            "bird", 
            "boat", 
            "bottle", 
            "bus", 
            "car", 
            "cat", 
            "chair", 
            "cow", 
            "diningtable", 
            "dog", 
            "horse", 
            "motorbike",
            "person",
            "pottedplant", 
            "sheep",
            "sofa", 
            "train",
            "tvmonitor"
            ) 
        return CLASSES

    def load_bboxes(self,xml_path):
        '''
        Load bounding boxes for VOC dataset
        Args:
             xml_path (str): path of xml file
        Returns:
             bounding boxes (float32) : array of coordinates (wmin,wmax,hmin,hmax,class_number)
        '''
        XML = self.parse_voc_xml(ET.parse(xml_path).getroot())['annotation']['object']
        if not isinstance(XML, list):
            XML = [XML]
        bboxes = []
        for xml in XML:
            bb_wmin = float(xml['bndbox']['xmin'])
            bb_wmax = float(xml['bndbox']['xmax'])
            bb_hmin = float(xml['bndbox']['ymin'])
            bb_hmax = float(xml['bndbox']['ymax'])
            cls_num = self.CLASSES.index(xml['name'])
            bboxes.append([bb_wmin, bb_hmin, bb_wmax, bb_hmax, cls_num])
        return np.array(bboxes).astype('float32')

# stage-3
class VOC_seg(Dataset):
    def __init__(self, cfg, transforms=None):
        self.train = False
        if cfg.DATA.MODE == "train_weak":
            txt_name = "train_aug.txt"
            self.train = True
        if cfg.DATA.MODE == "val":
            txt_name = "val.txt"
        if cfg.DATA.MODE == "test":
            txt_name = "test.txt"
            
        f_path = os.path.join(cfg.DATA.ROOT, "ImageSets/Segmentation", txt_name)
        self.filenames = [x.split('\n')[0] for x in open(f_path)]
        self.transforms = transforms
        
        self.annot_folders = ["SegmentationClassAug"]
        if cfg.DATA.PSEUDO_LABEL_PATH:
            self.annot_folders = cfg.DATA.PSEUDO_LABEL_PATH
        if cfg.DATA.MODE == "test":
            self.annot_folders = None
        
        self.img_path  = os.path.join(cfg.DATA.ROOT, "JPEGImages", "{}.jpg")
        if self.annot_folder is not None:
            self.mask_paths = [os.path.join(cfg.DATA.ROOT, folder, "{}.png") for folder in self.annot_folders]
        self.len = len(self.filenames)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        fn  = self.filenames[index]
        img = Image.open(self.img_path.format(fn))
        if self.annot_folder is not None:
            masks = [Image.open(mp.format(fn)) for mp in self.mask_paths]
        else:
            masks = None
            
        if self.transforms != None:
            img, masks = self.transforms(img, masks)
        
        return img, masks