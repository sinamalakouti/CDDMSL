# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

__all__ = ["load_voc_instances", "register_pascal_voc"]

# fmt: off
CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)


# fmt: on


def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)
    is_training = (split == 'train') or (split == 'trainval')
    is_voc = True if 'VOC' in dirname else False
    if is_training and is_voc:
        dr_name_clipart_dt = os.path.join(dirname, '../dt_clipart')
    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []

    # print("reading voc data")
    # print("is_voc", is_voc)
    # print(dirname)
    # print(split)
    #
    # print("-"*100)

    for fileid in fileids:

        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        # print("anno is   ", anno_file)
        # print("jpeg_file is   ", jpeg_file)
        if "VOC2007" in jpeg_file:
            voc_dir = "VOC2007"
        else:
            voc_dir = "VOC2012"
        if is_training and is_voc:
            clipart_dt_file = os.path.join(dr_name_clipart_dt, voc_dir, "JPEGImages", fileid + ".jpg")
        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)
        if is_training and is_voc:
            r = {
                "file_name": jpeg_file,
                "clipart_dt_file_name": clipart_dt_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
        else:
            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def load_voc_DG_instances(
        dirname: str,
        split: str,
        class_names: Union[List[str], Tuple[str, ...]],
        dt_data: str = None
):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)
    is_training = (split == 'train') or (split == 'trainval')

    if is_training and dt_data is not None:
        dr_name_data_dt = os.path.join(dirname, '../{}'.format(dt_data))
    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []

    # print("reading voc data")
    # print("is_voc", is_voc)
    # print(dirname)
    # print(split)
    #
    # print("-"*100)

    for fileid in fileids:

        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        # print("anno is   ", anno_file)
        # print("jpeg_file is   ", jpeg_file)
        if "VOC2007" in jpeg_file:
            voc_dir = "VOC2007"
        else:
            voc_dir = "VOC2012"
        if is_training and dt_data is not None:
            data_dt_file = os.path.join(dr_name_data_dt, voc_dir, "JPEGImages", fileid + ".jpg")
        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)
        if is_training and dt_data is not None:
            r = {
                "file_name": jpeg_file,
                "data_dt_file_name": data_dt_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
        else:
            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_pascal_voc(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )


def register_pascal_DG(name, dirname, split, year, class_names=CLASS_NAMES, dt_data=None):
    DatasetCatalog.register(name, lambda: load_voc_DG_instances(dirname, split, class_names, dt_data))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
