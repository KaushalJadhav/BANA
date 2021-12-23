import os
import numpy as np
from PIL import Image
try:
    from pycocotools.coco import COCO
    import imagesize
except ModuleNotFoundError:
    pass

root_dir = os.getcwd()
directory = os.path.join(root_dir,"val2017")
ann_path = os.path.join(root_dir,'annotations/instances_val2017.json')
annotations = COCO(ann_path)
cat_ids = annotations.getCatIds()
img_ids = []
for cat in cat_ids:
    img_ids.extend(annotations.getImgIds(catIds=cat))      
img_ids = list(set(img_ids))

j=0        
for img_id in img_ids:
        
    ann_ids = annotations.getAnnIds(imgIds=img_id)
    coco_annotation = annotations.loadAnns(ann_ids)
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
        bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
       
    path = annotations.loadImgs(img_id)[0]['file_name']
    img_path =os.path.join(directory,path)
    # img = np.array(Image.open(img_path))
    #h,w,dim = img.shape
    img_size =imagesize.get(img_path)  # This will make it a bit faster...Check whether to keep
    mask = np.ones((img_size[0],img_size[1]),dtype=np.uint8)
    for bbox in bboxes:
        (xmin,ymin,xmax,ymax) = bbox
        mask[ymin:ymax,xmin:xmax] = 0
    
    im = Image.fromarray(mask)
    output_name = path[:-4] + ".png"
    output_dir = os.path.join(root_dir,"BgMaskfromBoxes")
    output_path = os.path.join(output_dir,output_name)
    j=j+1
    #print(output_name, j)
    im.save(output_path)