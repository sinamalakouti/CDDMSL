import glob
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
import cv2
import os
meta = MetadataCatalog.get('cityscapes_foggy_val')

datasets = {
    # "cityscapes": "/projects/sina/RegionCLIP/datasets/cityscapes/leftImg8bit/test",
    "foggy-cityscapes": "/projects/sina/RegionCLIP/datasets/cityscapes/leftImg8bit_foggy/test",
    "bdd": "/projects/sina/RegionCLIP/datasets/bdd100k/images/100k/data"
}

models = ['ours', 'baseline']

for dataset_name, dataset_path in datasets.items():
    image_files = glob.glob(dataset_path + "/**/*.jpg", recursive=True) + glob.glob(dataset_path + "/**/*0.02.png",
                                                                                    recursive=True)

    image_files = image_files[:500]
    image_names = [os.path.basename(f) for f in image_files]

    images = [cv2.imread(f) for f in image_files]
    for model in models:
        # Gather all jpg images (including sub-directories)

        cfg = get_cfg()
        cfg.merge_from_file('./configs/City-Experiments/faster_rcnn_CLIP_R_50_C4.yaml')

        if model == 'ours':
            cfg.MODEL.WEIGHTS = '/projects/sina/RegionCLIP/output/exp_cross_city/DG_cityscapes_foggy/model_0029999.pth'
        elif model == 'baseline':
            cfg.MODEL.WEIGHTS = '/projects/sina/RegionCLIP/output/exp_cross_city/baseline/train-on-cityscapes/model_0029999.pth'

        cfg.MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH = '/projects/sina/RegionCLIP/pretrained_ckpt/concept_emb/city_8_emb.pth'
        cfg.MODEL.CLIP.TEXT_EMB_PATH = '/projects/sina/RegionCLIP/pretrained_ckpt/concept_emb/city_8_emb.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

        predictor = DefaultPredictor(cfg)

        for i in range(len(images)):
            im = images[i]
            image_name = image_names[i]

            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=meta,
                           scale=1.2,
                           instance_mode=ColorMode.IMAGE_BW
                           )
            v._default_font_size = 20  # this changes the font size
            v._instance_box_thickness = 4  # this changes the thickness of the bounding box
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(os.path.join(f'./predictions/{dataset_name}_{model}', "img_{}.jpg".format(image_name)),
                        v.get_image()[:, :, ::-1])
