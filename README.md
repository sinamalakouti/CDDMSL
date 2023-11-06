# Semi-Supervised Domain Generalization for Object Detection via Language-Guided Feature Alignment 	(CDDMSL)

It will be updated soon!

## Setup

## Datasets
### Real-to-Artistic
For this task, we used PASCAL-VOC as a labeled domain. Then, either Clipart, Comic, or Watercolor is used as the unlabeled domain. For instance, if Pascal-VOC and Clipart are used as labeled and unlabeled source domains. Then, Comics and Watercolor are the target domains in the DG experiment. 

Please see https://github.com/naoto0804/cross-domain-detection  for downloading the dataset. 

Please see the following files for dataset creation and/or modification: 

- detectron2/data/datasets/pascal_voc.py
- detectron2/data/datasets/builtin.py

### Adverse-Weather

Please download cityscapes and foggy-cityscapes (https://www.cityscapes-dataset.com/) as well as the bdd100k dataset (https://doc.bdd100k.com/download.html). Note that for bdd100k, we only used the validation set. 

Please see the following files for dataset creation and/or modification: 

- detectron2/data/datasets/cityscapes.py
- detectron2/data/datasets/builtin.py #for bdd100k, we used coco to register the data

## Pre-trained files

Please download pre-trained parameters from Google Drive here (will be updated soon)
You can find checkpoints required for both training and evaluation in the google drive. Some of the available parameters are: 
- RegionCLIP pretrained parameters
- Text Embedding (VOC)
- Text Embedding (Cityscapes):
- Vision-to-Language Transformer:
- Real-to-Artistic Parameters
- Adverse-Weather parameters


## Training 


## Inference
