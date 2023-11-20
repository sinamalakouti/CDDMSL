# Semi-Supervised Domain Generalization for Object Detection via Language-Guided Feature Alignment 	(CDDMSL)

This is the code for Semi-Supervised Domain Generalization for Object Detection via Language-Guided Feature Alignment, accepted at BMVC2023.
For any questions or more information, please contact Sina Malakouti (sem238@pitt.edu)

This repo will be updated soon!

## Setup
For setup please see
1. 

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

- Example for training a real-to-artistic generalization is available in faster_rcnn_voc.sh
An example for training an adverse-weather generalization is available in faster_rcnn_city.sh

## Acknowledgement
This repo is based on Detectron2 and RegionCLIP repositories. 


## Inference

# Other Information
- For training the CLIPCAP  model, please refer to https://github.com/rmokady/CLIP_prefix_caption
- For training/inference of the RegionCLIP pre-trained model, please refer to https://github.com/microsoft/RegionCLIP
