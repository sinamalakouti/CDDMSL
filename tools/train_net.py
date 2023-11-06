#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import copy
import time
import logging
import os
import weakref
from collections import OrderedDict
import numpy as np
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.config.config import add_ateacher_config
from detectron2.data import MetadataCatalog, DatasetMapperTwoCropSeparate
from detectron2.data.build import build_detection_semisup_train_loader_two_crops, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

from detectron2.engine.defaults import create_ddp_model
from detectron2.data.detection_utils import exp_rampup
from detectron2.modeling import GeneralizedRCNNWithTTA, build_backbone

# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.multiprocessing

from detectron2.modeling.backbone.clipcap.clipcap import ClipCaptionModel, v2l
from detectron2.modeling.meta_arch.ensemble_model import EnsembleModel
from detectron2.structures import Instances, Boxes
from detectron2.utils.events import EventStorage

torch.multiprocessing.set_sharing_strategy('file_system')


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            if "Water" in dataset_name or "Comic" in dataset_name:
                return PascalVOCDetectionEvaluator(dataset_name, ["person", "dog", "bicycle", "bird", "car", "cat"])
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


class ATeacherTrainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        # super().__init__()
        # logger = logging.getLogger("detectron2")
        # if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        #     setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # Assume these objects must be constructed in this order.

        ##student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # clipcap model

        self.clipcap_model = ClipCaptionModel(40, 40)
        # p = torch.load('/Users/sinamalakouti/Desktop/RegionCLIP/test-regionclip/transformer_weights_r50.pt', 'cpu')
        # p = torch.load('/projects/sina/RegionCLIP/pretrained_ckpt/transformer_r50_regionCLIP.pt', 'cpu')
        p = torch.load('/projects/sina/RegionCLIP/pretrained_ckpt/transformers_pretrained_RegionCLIP.pt', 'cpu')
        self.clipcap_model.load_state_dict(p)
        # self.clipcap_model.lm_head = self.clipcap_model.gpt.lm_head
        # self.clipcap_model.gpt.lm_head = Identity()
        self.clipcap_model = self.clipcap_model.clip_project
        for p in self.clipcap_model.parameters():
            p.requires_grad = False

        self.offline_backbone = build_backbone(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        ensem_ts_model = EnsembleModel(model_teacher, model)

        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        # load 2nd pretrained model
        if cfg.MODEL.META_ARCHITECTURE in ['CLIPRCNN', 'CLIPFastRCNN', 'PretrainFastRCNN'] \
                and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None \
                and cfg.MODEL.CLIP.CROP_REGION_TYPE == 'RPN':  # load 2nd pretrained model
            self.second_checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True)
        else:
            self.second_checkpointer = None
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if self.second_checkpointer:
            self.second_checkpointer.resume_or_load(
                self.cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
            )
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            if "Water" in dataset_name or "Comic" in dataset_name:
                return PascalVOCDetectionEvaluator(dataset_name, ["person", "dog", "bicycle", "bird", "car", "cat"])
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    # todo
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    # todo
    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                print("loading offlinee backbone params")
                all_params = \
                    torch.load('/projects/sina/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth',
                               'cpu')['model']
                new_params = {}
                for param in all_params:
                    if 'backbone' in param and 'offline_backbone' not in param and 'teacher_backbone' not in param:
                        new_params[param[9:]] = all_params[param]

                self.offline_backbone.load_state_dict(new_params, self.model.device)

                # self.offline_backbone.load_state_dict(self.model.module.backbone.state_dict())
                for p in self.offline_backbone.parameters(): p.requires_grad = False
                self.offline_backbone.eval()
                print("OK. .. Done")
                self.total_lossess = 0
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    # self.run_step_full_semisup_gradient_accumulation()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                    for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key].to(value.device) *
                        (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
            self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))

        return label_list

    def v2l_contrastive_loss(self, batched_inputs):
        # offline backbone on src
        self.model_teacher = self.model_teacher.to(self.model.device)
        images_src, images_target = self.model.module.preprocess_image_caption_consistency(batched_inputs)

        prefix_src = self.offline_backbone.to(self.model.device).attnpool(
            self.offline_backbone.to(self.model.device)(images_src)['res5'])
        teacher_features = v2l(prefix_src, self.clipcap_model.to(self.model.device)).detach()

        # student backbone on target
        student_prefix_trgt = self.model.module.backbone.attnpool(self.model.module.backbone(images_target)['res5'])
        student_features_trgt = v2l(student_prefix_trgt, self.clipcap_model.to(self.model.device))
        student_features_trgt = self.model.module.projector(student_features_trgt)

        # student backbone on src
        student_prefix_src = self.model.module.backbone.attnpool(self.model.module.backbone(images_src)['res5'])
        student_features_src = v2l(student_prefix_src, self.clipcap_model.to(self.model.device))

        # computing l1 loss ( knowledge distillation)
        l1_loss = torch.nn.L1Loss()

        kd_loss = l1_loss(teacher_features, student_features_src)

        student_features_src = self.model.module.projector(student_features_src)

        # ema_student backbone on src

        # teach_prefix_src = self.model_teacher.backbone.attnpool(self.model_teacher.backbone(images_src)['res5'])
        # teach_features_src = v2l(teach_prefix_src, self.clipcap_model.to(self.model.device))
        # teach_features_src = self.model_teacher.projector(teach_features_src).detach()

        del images_src
        del images_target

        from detectron2.modeling.backbone.clipcap.gather import GatherLayer
        student_features_trgt = torch.cat(GatherLayer.apply(student_features_trgt), dim=0)
        # teach_features_src = torch.cat(GatherLayer.apply(teach_features_src), dim=0)
        student_features_src = torch.cat(GatherLayer.apply(student_features_src), dim=0)

        student_features_trgt = student_features_trgt / student_features_trgt.norm(dim=1, keepdim=True)
        student_features_src = student_features_src / student_features_src.norm(dim=1, keepdim=True)

        from torch import nn
        loss_fn = nn.CrossEntropyLoss()

        joint_features = student_features_trgt @ student_features_src.t()
        n = len(joint_features)
        ground_truth = torch.arange(n, dtype=torch.long, device=self.model.device)

        cont_loss = loss_fn(joint_features, ground_truth)
        losses = {}
        losses['loss_cont'] = cont_loss
        losses['loss_kd'] = kd_loss

        return losses, [], [], None

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        self.model.zero_grad()

        self.clipcap_model.zero_grad()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, label_style_transfer, unlabel_data_q, unlabel_data_k, = data
        # print("checkking the length of all dataa")
        # print(len(label_data_k))
        # print(len(label_style_transfer))
        # print(len(unlabel_data_k))
        data_time = time.perf_counter() - start

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(
                label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy  the whole model
                self._update_teacher_model(keep_rate=0.00)
                # self.model.build_discriminator()

            elif (
                    self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}

            ######################## For probe #################################
            # import pdb; pdb. set_trace()
            gt_unlabel_k = self.get_label(unlabel_data_k)
            # gt_unlabel_q = self.get_label_test(unlabel_data_q)

            #  0. remove unlabeled data labels
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            #  1. generate the pseudo-label using teacher model
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            ######################## For probe #################################
            # import pdb; pdb. set_trace()

            # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
            # probe_metrics = ['compute_num_box']
            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
            # record_dict.update(analysis_pred)
            ######################## For probe END #################################

            #  2. Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            # Process pseudo labels and thresholding
            (
                pesudo_proposals_rpn_unsup_k,
                nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
            # record_dict.update(analysis_pred)

            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

            # 3. add pseudo-label to unlabeled data

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            # 4. input both strongly and weakly augmented labeled data into student model
            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)

            # 5. input strongly augmented unlabeled data into model
            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, branch="supervised_target"
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            # 6. input weakly labeled data (source) and styled transfer version of the source data ( weak augmentation) to do consistency training

            for i_index in range(len(label_style_transfer)):
                # unlabel_data_item = {}
                for k, v in label_style_transfer[i_index].items():
                    # label_data_k[i_index][k + "_unlabeled"] = v
                    label_data_k[i_index][k + "_unlabeled"] = v
                # unlabel_data_k[i_index] = unlabel_data_item

            all_domain_data = label_data_k
            #
            record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="caption_consistency",
                                                         clipcap_model=self.clipcap_model.to(self.model.device),
                                                         offline_backbone=self.offline_backbone.to(self.model.device))

            # record_all_domain_data, _, _, _= self.v2l_contrastive_loss(all_domain_data)

            record_dict.update(record_all_domain_data)

            region_consistency_loss, _, _, _ = self.model(all_domain_data, clipcap_model=self.clipcap_model.to(self.model.device),
                                                          branch='caption_consistency_regionLevel')
            record_dict.update(region_consistency_loss)
            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                                record_dict[key] *  self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif 'cont' in key or 'kd' in key:
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def run_step_full_semisup_gradient_accumulation(self):
        torch.cuda.empty_cache()

        self._trainer.iter = self.iter
        self.accum_iter = 2
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, label_style_transfer, unlabel_data_q, unlabel_data_k, = data

        if self.iter >= self.cfg.SEMISUPNET.BURN_UP_STEP:
            # print("here" * 100)
            if self.iter % self.accum_iter == 0:
                self.total_lossess = 0
                del unlabel_data_q
                del unlabel_data_k
                del label_data_q
                self.model_teacher = self.model_teacher.to('cpu')
            else:
                del label_data_q
                del label_data_k
                del label_style_transfer
                self.model_teacher = self.model_teacher.to(self.model.device)
        else:
            print("noo" * 100)

        torch.cuda.empty_cache()
        data_time = time.perf_counter() - start

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(
                label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy  the whole model
                self._update_teacher_model(keep_rate=0.00)
                # self.model.build_discriminator()

            elif (
                    self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0 and self.iter % self.accum_iter == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}

            if self.iter % self.accum_iter == 1:
                ######################## For probe #################################
                # import pdb; pdb. set_trace()
                gt_unlabel_k = self.get_label(unlabel_data_k)
                # gt_unlabel_q = self.get_label_test(unlabel_data_q)
                #  0. remove unlabeled data labels
                unlabel_data_q = self.remove_label(unlabel_data_q)
                unlabel_data_k = self.remove_label(unlabel_data_k)

                #  1. generate the pseudo-label using teacher model
                with torch.no_grad():
                    (
                        _,
                        proposals_rpn_unsup_k,
                        proposals_roih_unsup_k,
                        _,
                    ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

                ######################## For probe #################################
                # import pdb; pdb. set_trace()

                # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
                # probe_metrics = ['compute_num_box']
                # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
                # record_dict.update(analysis_pred)
                ######################## For probe END #################################

                #  2. Pseudo-labeling
                cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

                joint_proposal_dict = {}
                joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
                # Process pseudo labels and thresholding
                (
                    pesudo_proposals_rpn_unsup_k,
                    nun_pseudo_bbox_rpn,
                ) = self.process_pseudo_label(
                    proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
                )
                # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
                # record_dict.update(analysis_pred)

                joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
                # Pseudo_labeling for ROI head (bbox location/objectness)
                pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                    proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
                )
                joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

                # 3. add pseudo-label to unlabeled data

                unlabel_data_q = self.add_label(
                    unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
                )
                unlabel_data_k = self.add_label(
                    unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
                )

                all_unlabel_data = unlabel_data_q

            # 4. input both strongly and weakly augmented labeled data into student model

            if self.iter % self.accum_iter == 0:
                # all_label_data = label_data_q + label_data_k
                record_all_label_data, _, _, _ = self.model(
                    label_data_k, branch="supervised"
                )
                record_dict.update(record_all_label_data)
                # del label_data_q
            if self.iter % self.accum_iter == 1:
                # 5. input strongly augmented unlabeled data into model
                record_all_unlabel_data, _, _, _ = self.model(
                    all_unlabel_data, branch="supervised_target"
                )
                new_record_all_unlabel_data = {}
                for key in record_all_unlabel_data.keys():
                    new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                        key
                    ]
                record_dict.update(new_record_all_unlabel_data)

            # 6. input weakly labeled data (source) and styled transfer version of the source data ( weak augmentation) to do consistency training
            if self.iter % self.accum_iter == 0:
                for i_index in range(len(label_style_transfer)):
                    # unlabel_data_item = {}
                    for k, v in label_style_transfer[i_index].items():
                        # label_data_k[i_index][k + "_unlabeled"] = v
                        label_data_k[i_index][k + "_unlabeled"] = v
                    # unlabel_data_k[i_index] = unlabel_data_item

                all_domain_data = label_data_k
                #
                record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="caption_consistency",
                                                             clipcap_model=self.clipcap_model.to(self.model.device),
                                                             offline_backbone=self.offline_backbone.to(
                                                                 self.model.device))

                # record_all_domain_data, _, _, _= self.v2l_contrastive_loss(all_domain_data)

                record_dict.update(record_all_domain_data)

            if self.iter % self.accum_iter == 1:
                for i_index in range(len(unlabel_data_q)):
                    # unlabel_data_item = {}
                    for k, v in unlabel_data_q[i_index].items():
                        # label_data_k[i_index][k + "_unlabeled"] = v
                        unlabel_data_k[i_index][k + "_unlabeled"] = v
                    # unlabel_data_k[i_index] = unlabel_data_item

                all_domain_data = unlabel_data_k
                #
                record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="caption_consistency",
                                                             clipcap_model=self.clipcap_model.to(self.model.device),
                                                             offline_backbone=self.offline_backbone.to(
                                                                 self.model.device))

                # record_all_domain_data, _, _, _ = self.v2l_contrastive_loss(all_domain_data)
                for key in record_all_domain_data:
                    if "kd" not in key:
                        record_dict[key + "_pseudo"] = record_all_domain_data[key]
                record_dict.update(record_all_domain_data)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                                record_dict[key] * 1
                            # self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif 'cont' in key or 'kd' in key:
                        weight = exp_rampup(self.iter, rampup_length=2000)
                        loss_dict[key] = record_dict[key] * weight
                    elif key == 'loss_cont_pseudo':
                        loss_dict[key] = record_dict['loss_cont_pseudo'] * 0.0
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        # print("lossesss issss ")
        # print(losses)
        # self.total_lossess += losses
        # losses.backward()
        # if ((self.iter + 1) % self.accum_iter == 0) or (self.iter + 1 == self.max_iter):
        #     self.optimizer.zero_grad()
        # self.total_lossess.backward()
        # self.optimizer.step()

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ateacher_config(cfg)  # uncomment if you want use mean_teacher
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.MODEL.META_ARCHITECTURE in ['CLIPRCNN', 'CLIPFastRCNN', 'PretrainFastRCNN'] \
                and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None \
                and cfg.MODEL.CLIP.CROP_REGION_TYPE == 'RPN':  # load 2nd pretrained model
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True).resume_or_load(
                cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
            )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    if cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer1 = ATeacherTrainer
    else:
        Trainer1 = Trainer
    trainer = Trainer(cfg)

    # trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
