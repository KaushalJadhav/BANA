# [RE] Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation

This repository is the PyTorch and PyTorch Lightning implementation of the paper ["Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation"](https://arxiv.org/pdf/2104.00905.pdf). It is well documented version of the original repository with the code flow available [here](). The paper address the problem of weakly-supervised semantic segmentation (WSSS) using bounding box annotations by proposing two novel methods:
- **Background-aware pooling (BAP)**, to extract high-quality pseudo segmentation labels
- **Noise-aware Loss (NAL)**, to make the networks less susceptible to incorrect labels

<p align="center">
<a><img src="https://i.ibb.co/rcn1F2D/error.png" alt="error" border="0"><br>Visual comparison of pseudo ground-truth labels</a>
</p>

For more information, please checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/BANA/)].

## Requirements

The code is developed and tested using `Python >= 3.6`. To install the requirements:

```bash
pip install -r requirements.txt
```

To setup the dataset:

```bash
bash data/setup_voc.bash /path-to-data-directory
```

To generate background masks:
```bash
cd /path-to-data-directory
python3 voc_bbox_setup.py
```
Once finished, the folder `data` should be like this:

```
    data   
    └── VOCdevkit
        └── VOC2012
            ├── JPEGImages
            ├── SegmentationClassAug
            ├── Annotations
            ├── ImageSets
            ├── BgMaskfromBoxes
            └── Generation
                ├── Y_crf
                └── Y_ret
```

## Training

The training procedure is divided into 3 stages and example commands for each have been given below. Hyperparameters can be adjusted accordingly in the corresponding configuration files.

### **Stage 1:** Training the classification network

Change the `MODEL.GAP` parameter in config file to train the model: 

1. `True`: With the Global Average Pooling method 
2. `False`: With the proposed Background Aware Pooling method

By default, all the models are trained using Augmented PASCAL VOC containing 10,582 images and can be trained using Non-Augmented dataset by changing the `DATA.AUG` parameter to `False`. 

```bash
python3 stage1.py --config-file configs/stage1.yml --gpu-id 0
```

### **Stage 2:** Generating pseudo labels

```bash
python3 stage2.py --config-file configs/stage2.yml --gpu-id 0
```

### **Stage 3:** Training a CNN using the pseudo labels

DeepLab Large FOV (VGG Backbone):

```bash
python3 stage3.py --config-file configs/stage3_vgg.yml --gpu-id 0
```

DeepLab ASPP (Resnet Backbone): 

```bash
python3 stage3.py --config-file configs/stage3_res.yml --gpu-id 0
```

Change the `MODEL.LOSS` parameter in the corresponding config file to train the model: 

1. `NAL` : With the proposed Noise Aware Loss using Ycrf and Yret
2. `CE_CRF`: With the Cross Entropy Loss using Ycrf
3. `CE_RET`: With the Cross Entropy Loss using Yret
## Evaluation

To evaluate the model on the validation set of Pascal VOC 2012 dataset before and after Dense CRF processing change the `DATA.MODE` parameter to `val` in the corresponding config file:

```bash
python3 stage3.py --config-file configs/stage3_vgg.yml --gpu-id 0
```

DeepLab ASPP (Resnet Backbone): 

```bash
python3 stage3.py --config-file configs/stage3_res.yml --gpu-id 0
```

Evaluation would be performed on raw validation set images to obtain the Mean Accuracy and IOU metrics pre and post-Dense CRF processing.
## Pre-trained Models and Pseudo Labels

- Pretrained models: [Link](https://drive.google.com/drive/folders/14F1vU7Gp-nIZVhPe2XzrbnyUYTmnt2Sz?usp=sharing)

- Pseudo Labels: [Link](https://drive.google.com/drive/folders/1wC9qr1lE_JN0Htrf0SfPhKz4AdqQ0zbt?usp=sharing)

## Quantitative Results

We achieve the following results:

- Comparison of pseudo labels on the PASCAL VOC 2012 validation set in terms of mIoU

| **Method**          | **Original Author's Results** | **Our Results** |
|:-------------------:|:-----------------------------:|:---------------:|
| **GAP**             | 76.1                          | 75.5            |
| **BAP Ycrf w/o u0** | 77.8                          | 77              |
| **BAP Ycrf**        | 79.2                          | 78.8            |
| **BAP Yret**        | 69.9                          | 69.9            |
| **BAP Ycrf & Yret** | 68.2                          | 72.7            |

- Comparison of pseudo labels on the MS-COCO training set

- Comparison of mIoU scores using different losses on the PASCAL VOC 2012 training set. We provide both mIoU scores before/after applying DenseCRF

- Quantitative comparison using DeepLab-V1 (VGG-16) on the PASCAL VOC 2012 dataset in terms of mIoU
    - Weakly supervised learning

| **Method**          | **Original Author's Results** | **Our Results** |
|:-------------------:|:-----------------------------:|:---------------:|
| **w/ Ycrf**         | 67.8                          | 64.7            |
| **w/ Yret**         | 66.1                          | 58.7            |
| **w/ NAL**          | 68.1                          | 64.5            |


    - Semi-supervised learning

- Quantitative comparison using DeepLab-V2 (ResNet-101) on the PASCAL VOC 2012 dataset in terms of mIoU
    - Weakly supervised learning

| **Method**          | **Original Author's Results** | **Our Results** |
|:-------------------:|:-----------------------------:|:---------------:|
| **w/ Ycrf**         | 74.0                          | 67.0            |
| **w/ Yret**         | 72.4                          | 70.2            |
| **w/ NAL**          | 74.6                          |                 |

    - Semi-supervised learning

- Quantitative comparison for instance segmentation on the MS-COCO test set

## Qualitative Results

## Contributors

[Aryan Mehta](https://github.com/victorvini08), [Karan Uppal](https://github.com/karan-uppal3), [Kaushal Jadhav](https://github.com/KaushalJadhav), [Monish Natrajan](https://github.com/Monish-Natarajan) and [Mradul Agrawal](https://github.com/mradul2)

This repository is maintained by [AGV.AI (IIT Kharagpur)](http://www.agv.iitkgp.ac.in/)

## Bibtex
```
@inproceedings{oh2021background,
  title     = {Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation},
  author    = {Oh, Youngmin and Kim, Beomjun and Ham, Bumsub},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021},
}
```

## Acknowledgments

- PASCAL VOC 2012 Setup adopted from [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/voc12/README.md)

## To-do

- Add the code flow link
- Complete the tables
- Add examples of pseudo labels generated and predictions in qualitative comparison