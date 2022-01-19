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

The training procedure is divided into 3 stages and example commands for each have been given below. Hyperparameters can be adjusted accordingly in the correponding configuration files.

1. **Stage 1:** Training the classification network

```bash
python3 stage1.py --config-file configs/stage1.yml --gpu-id 0
```

2. **Stage 2:** Generating pseudo labels

```bash
python3 stage2.py --config-file configs/stage2.yml --gpu-id 0
```

3. **Stage 3:** Training a CNN using the pseudo labels

```bash
python3 stage3.py --config-file configs/stage3_vgg.yml --gpu-id 0
```

## Evaluation

To evaluate the model on the validation set of Pascal VOC 2012 dataset:

```bash
python3 
```

## Pre-trained Models and Pseudo Labels

- Pretrained models:

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
    - Semi-supervised learning

- Quantitative comparison using DeepLab-V2 (ResNet-101) on the PASCAL VOC 2012 dataset in terms of mIoU
    - Weakly supervised learning
    - Semi-supervised learning

- Quantitative comparison for instance segmentation on the MS-COCO test set

## Qualitative Results

## Contributors

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

## To-do

- Add the code flow link
- Add comments and docstring to data/setup_dataset.py
- Add more information regarding training (basically explain the different options: aug, naug, bap, gap, ycrf, yret, vgg, resnet, etc.)
- Write evaluation command
- Add links to download pre-trained models and pseudo labels (all)
- Complete the tables
- Add examples of pseudo labels generated and predictions in qualitative comparison
- Add contributors
