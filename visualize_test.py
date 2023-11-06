from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
import cv2
from detectron2.engine import DefaultPredictor
import os
from detectron2.data import DatasetCatalog, MetadataCatalog

import xml.etree.ElementTree as ET

meta = MetadataCatalog.get('voc_2007_test')
datasets = ["clipart", "watercolor", "comic"]
models = ['ours', 'baseline']
for dataset_name in datasets:

    # balloon_metadata = MetadataCatalog.get("my_dataset_train2")

    with open(f'/projects/sina/RegionCLIP/datasets/{dataset_name}/ImageSets/Main/test.txt', 'r') as file:
        image_files = file.read().splitlines()

    with open(f'/projects/sina  /RegionCLIP/datasets/{dataset_name}/ImageSets/Main/test.txt', 'r') as file:
        image_names = file.read().splitlines()

    image_files = [os.path.join(f'/projects/sina/RegionCLIP/datasets/{dataset_name}/JPEGImages', f + '.jpg') for f
                   in
                   image_files]

    if dataset_name == 'clipart':
        image_files = ['/projects/sina/RegionCLIP/datasets/clipart/JPEGImages/392565807.jpg']
        image_names = ['392565807']
    elif dataset_name == 'watercolor':
        image_files = ['/projects/sina/RegionCLIP/datasets/watercolor/JPEGImages/12133135.jpg']
        image_names = ['12133135']
    elif dataset_name == 'comic':
        image_files = ['/projects/sina/RegionCLIP/datasets/comic/JPEGImages/33497847.jpg']
        image_names = ['33497847']
    images = [cv2.imread(f) for f in image_files]


    for model in models:

        # im = cv2.imread("/projects/sina/RegionCLIP/datasets/VOC2007/JPEGImages/002933.jpg")
        cfg = get_cfg()
        cfg.merge_from_file('./configs/VOC-Experiments/faster_rcnn_CLIP_R_50_C4.yaml')
        if model == 'ours':
            cfg.MODEL.WEIGHTS = '/projects/sina/RegionCLIP/output/exp_voc_to_artisitc/train-on-voc-clipart/Full_contrastive_learning_new_multi2/model_0029999.pth'
        elif model == 'baseline':
            cfg.MODEL.WEIGHTS = '/projects/sina/RegionCLIP/output/model_regionClip_baseline_prompt.pth'
        cfg.MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH = '/projects/sina/RegionCLIP/pretrained_ckpt/concept_emb/voc_20_cls_emb.pth'
        cfg.MODEL.CLIP.TEXT_EMB_PATH = '/projects/sina/RegionCLIP/pretrained_ckpt/concept_emb/voc_20_cls_emb.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
        predictor = DefaultPredictor(cfg)

        for i in range(len(images)):
            im = images[i]
            image_name = image_names[i]

            outputs = predictor(im)
            # print(outputs)
            v = Visualizer(im[:, :, ::-1],
                           metadata=meta,
                           scale=1.2,
                           instance_mode=ColorMode.IMAGE_BW
                           # remove the colors of unsegmented pixels. This option is only available for segmentation models
                           )
            v._default_font_size = 24  # this changes the font size
            v._instance_box_thickness = 5  # this changes the thickness of the bounding box
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(os.path.join(f'./predictions/{dataset_name}_{model}', "img_{}.jpg".format(image_name)),
                        v.get_image()[:, :, ::-1])

            # Ground Truth Visualization
            annotation_file = os.path.join('/projects/sina/RegionCLIP/datasets', dataset_name, 'Annotations',
                                           f'{image_name}.xml')
            tree = ET.parse(annotation_file)
            root = tree.getroot()
            boxes, classes = [], []
            for member in root.findall('object'):
                boxes.append([int(member[4][i].text) for i in range(4)])
                classes.append(member[0].text)
            annotations = {"boxes": boxes, "classes": classes}
            v_gt = Visualizer(im[:, :, ::-1], metadata=meta, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            v_gt._default_font_size = 24  # this changes the font size
            v_gt._instance_box_thickness = 5  # this changes the thickness of the bounding box
            v_gt = v_gt.draw_dataset_dict(annotations)
            cv2.imwrite(os.path.join(f'./predictions/{dataset_name}_GT/', "img_{}.jpg".format(image_name)),
                        v_gt.get_image()[:, :, ::-1])