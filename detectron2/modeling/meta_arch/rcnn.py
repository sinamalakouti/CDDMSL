# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.lib import pad
import torch
from torch import nn
from torch.nn import functional as F
from random import randint

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from ..backbone.clipcap.clipcap import unsupervised_loss, unsupervised_feature_loss, generate_feature_caption, \
    generate_first_feature_caption, v2l, pseudo_labeling_loss

from ..backbone.clipcap.gather import GatherLayer

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]

from torchvision.transforms import Resize, CenterCrop

from torchvision.transforms.functional import InterpolationMode
from ...data.transforms.torchvision_transforms.transforms import Normalize

from detectron2.modeling.backbone.clipcap.clipcap import ClipCaptionModel


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
            self,
            *,
            offline_backbone: Backbone,
            backbone: Backbone,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
            use_clip_c4: False,
            use_clip_attpool: False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.offline_backbone = offline_backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        if np.sum(pixel_mean) < 3.0:  # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else:  # default setting
            self.div_pixel = False
        self.use_clip_c4 = use_clip_c4  # if True, use C4 mode where roi_head uses the last resnet layer from backbone
        self.use_clip_attpool = use_clip_attpool  # if True (C4+text_emb_as_classifier), use att_pool to replace default mean pool

        self.projector = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 256)
        )

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        #
        offline_backbone = build_backbone(cfg)

        for p in offline_backbone.parameters(): p.requires_grad = False
        offline_backbone.eval()
        return {
            "offline_backbone": offline_backbone,
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "use_clip_c4": cfg.MODEL.BACKBONE.NAME == "build_clip_resnet_backbone",
            "use_clip_attpool": cfg.MODEL.ROI_HEADS.NAME == 'CLIPRes5ROIHeads' and cfg.MODEL.CLIP.USE_TEXT_EMB_CLASSIFIER,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """

        preprocess2 = nn.Sequential(
            Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            CenterCrop(size=(224, 224)),
            Normalize(mean=self.pixel_mean, std=self.pixel_std))

        images = [(x["image"] / 255.0).to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        images = preprocess2(images.tensor)

        images_t = [(x["image_trgt"] / 255.0).to(self.device) for x in batched_inputs]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)
        images_t = preprocess2(images_t.tensor)

        return images, images_t

    def preprocess_image_test(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """

        preprocess2 = nn.Sequential(
            Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            CenterCrop(size=(224, 224)),
            Normalize(mean=self.pixel_mean, std=self.pixel_std))

        images = [(x["image"] / 255.0).to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        images = preprocess2(images.tensor)
        return images

    def preprocess_image_caption_consistency_regionLevel(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """

        images = [((x["image"] / 255.0) - self.pixel_mean) / self.pixel_std for x in batched_inputs]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [((x["image_trgt"] / 255.0) - self.pixel_mean) / self.pixel_std for x in batched_inputs]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t


        # offline backbone on src
        prefix_src = self.offline_backbone.attnpool(self.offline_backbone(images_src)['res5'])
        teacher_features, capsrc = generate_first_feature_caption(prefix_src, clipcap_model.to(self.device), 40)
        teacher_features = torch.stack(teacher_features, 0).detach()  # detach the offline backbone

        # backbone on target
        student_prefix_trgt = self.backbone.attnpool(self.backbone(images_target)['res5'])
        student_features_trgt, captrgt = generate_first_feature_caption(student_prefix_trgt,
                                                                        clipcap_model.to(self.device), 40)
        student_features_trgt = torch.stack(student_features_trgt, 0)

        # backbone on src
        student_prefix_src = self.backbone.attnpool(self.backbone(images_src)['res5'])
        student_features_src, captrgt_src = generate_first_feature_caption(student_prefix_src,
                                                                           clipcap_model.to(self.device),
                                                                           40)
        student_features_src = torch.stack(student_features_src, 0)

        teacher_features = teacher_features.squeeze(1)
        student_features_trgt = student_features_trgt.squeeze(1)
        student_features_src = student_features_src.squeeze(1)
        del images_src
        del images_target

        l1_loss = torch.nn.L1Loss()

        kd_loss = l1_loss(teacher_features, student_features_src)

        student_features_trgt = torch.cat(GatherLayer.apply(student_features_trgt), dim=0)
        student_features_src = torch.cat(GatherLayer.apply(student_features_src), dim=0)

        student_features_trgt = student_features_trgt / student_features_trgt.norm(dim=1, keepdim=True)
        student_features_src = student_features_src / student_features_src.norm(dim=1, keepdim=True)

        loss_fn = nn.CrossEntropyLoss()

        joint_features = student_features_trgt @ student_features_src.t()
        n = len(joint_features)
        ground_truth = torch.arange(n, dtype=torch.long, device=self.device)


        cont_loss = loss_fn(joint_features, ground_truth)

        return cont_loss, kd_loss

    def v2l_contrastive(self, images_src, images_target, clipcap_model, KD_regularization=True):

        # backbone on target
        student_prefix_trgt = self.backbone.attnpool(self.backbone(images_target)['res5'])
        student_features_trgt = v2l(student_prefix_trgt, clipcap_model.to(self.device))
        student_features_trgt = self.projector(student_features_trgt)

        # backbone on src
        student_prefix_src = self.backbone.attnpool(self.backbone(images_src)['res5'])
        student_features_src = v2l(student_prefix_src, clipcap_model.to(self.device))
        if KD_regularization:
            # offline backbone on src
            prefix_src = self.offline_backbone.attnpool(self.offline_backbone(images_src)['res5'])
            teacher_features = v2l(prefix_src, clipcap_model.to(self.device)).detach()

            l1_loss = torch.nn.L1Loss()

            kd_loss = l1_loss(teacher_features, student_features_src)
        else:
            kd_loss = None

        student_features_src = self.projector(student_features_src)
        # #
        # # # loss, captions = unsupervised_loss(prefix_src, prefix_trgt, clipcap_model.to(self.device), 40)
        # # # loss, captions = unsupervised_feature_loss(prefix_src, prefix_trgt, clipcap_model.to(self.device), 40)
        # # if self.device == torch.device('cuda:0'):
        # # #     from torchvision.utils import save_image
        # # #     import clip
        # # #     model, preprocess = clip.load("RN50", device='cpu')
        # # #     p_src = model.encode_image(images_src.cpu())
        # # #     p_tgt = model.encode_image(images_target.cpu())
        # # #     _, clip_src = generate_first_feature_caption(p_src, clipcap_model.to('cpu'), 40)
        # # #     _, clip_trgt = generate_first_feature_caption(p_tgt, clipcap_model.to('cpu'), 40)
        # # #
        # # #
        # # #     print("cap_src  ", clip_src)
        # # #     print("cap_trgt ", clip_trgt)
        # #
        #     print("region_clip_src  ", capsrc)
        #     print("region_clip_trgt  ", captrgt)
        # from torchvision.utils import save_image
        # storage = get_event_storage()
        # p = '/projects/sina/RegionCLIP/images/'
        # for i in range(len(images_src)):
        #     save_image(images_src[i].cpu(), p + "img_src_iter_{}_img_{}.png".format(storage.iter, i))
        #     save_image(images_target[i].cpu(), p + "img_trgt_iter_{}_img_{}.png".format(storage.iter, i))

        del images_src
        del images_target

        student_features_trgt = torch.cat(GatherLayer.apply(student_features_trgt), dim=0)
        student_features_src = torch.cat(GatherLayer.apply(student_features_src), dim=0)

        student_features_trgt = student_features_trgt / student_features_trgt.norm(dim=1, keepdim=True)
        student_features_src = student_features_src / student_features_src.norm(dim=1, keepdim=True)

        loss_fn = nn.CrossEntropyLoss()

        joint_features = student_features_trgt @ student_features_src.t()
        n = len(joint_features)
        ground_truth = torch.arange(n, dtype=torch.long, device=self.device)

        cont_loss = (loss_fn(joint_features, ground_truth) + loss_fn(joint_features.t(), ground_truth)) / 2

        return cont_loss, kd_loss

    def image_contrastive(self, images_src, images_target):

        # backbone on target
        student_prefix_trgt = self.backbone.attnpool(self.backbone(images_target)['res5'])
        student_features_trgt = self.projector(student_prefix_trgt)

        # backbone on src
        student_prefix_src = self.backbone.attnpool(self.backbone(images_src)['res5'])

        student_features_src = self.projector(student_prefix_src)

        del images_src
        del images_target

        student_features_trgt = torch.cat(GatherLayer.apply(student_features_trgt), dim=0)
        student_features_src = torch.cat(GatherLayer.apply(student_features_src), dim=0)

        student_features_trgt = student_features_trgt / student_features_trgt.norm(dim=1, keepdim=True)
        student_features_src = student_features_src / student_features_src.norm(dim=1, keepdim=True)

        loss_fn = nn.CrossEntropyLoss()

        joint_features = student_features_trgt @ student_features_src.t()
        n = len(joint_features)
        ground_truth = torch.arange(n, dtype=torch.long, device=self.device)

        cont_loss = (loss_fn(joint_features, ground_truth) + loss_fn(joint_features.t(), ground_truth)) / 2
        # cont_loss = loss_fn(joint_features, ground_truth)
        return cont_loss

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], clipcap_model=None, branch='supervised',
                KD_regularization=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        torch.cuda.empty_cache()

        if not self.training:
            return self.inference(batched_inputs)
        if branch == "caption-image-drawing":
            from torchvision.utils import save_image
            with torch.no_grad():
                if self.device == torch.device('cuda:0'):
                    images_src, images_target = self.preprocess_image_train(batched_inputs)

                    p_src = self.backbone.attnpool(self.backbone(images_src)['res5'])
                    p_tgt = self.backbone.attnpool(self.backbone(images_target)['res5'])

                    p_src_offline = self.offline_backbone.attnpool(self.offline_backbone(images_src)['res5'])
                    p_tgt_offline = self.offline_backbone.attnpool(self.offline_backbone(images_target)['res5'])

                    storage = get_event_storage()
                    p = '/projects/sina/RegionCLIP/images/'  #todo

                    for i in range(len(images_src)):
                        print("image  :   ", batched_inputs[i]['file_name'])
                        save_image(images_src[i].cpu(),
                                   p + "img_src_iter_{}_img_{}.png".format(storage.iter, batched_inputs[i]['image_id']))
                        save_image(images_target[i].cpu(), p + "img_trgt_iter_{}_img_{}.png".format(storage.iter,
                                                                                                    batched_inputs[i][
                                                                                                        'image_id']))

                        _, my_src = generate_first_feature_caption(p_src.to('cpu'), clipcap_model.to('cpu'), 40)
                        _, my_trgt = generate_first_feature_caption(p_tgt.to('cpu'), clipcap_model.to('cpu'), 40)
                        print("ours - src   ", my_src)
                        print("ours-target  ", my_trgt)
                        _, offline_src = generate_first_feature_caption(p_src_offline.to('cpu'),
                                                                        clipcap_model.to('cpu'), 40)
                        _, offline_trg = generate_first_feature_caption(p_tgt_offline.to('cpu'),
                                                                        clipcap_model.to('cpu'), 40)
                        print("offline - src   ", offline_src)
                        print("offline-target  ", offline_trg)

        if branch == 'caption_consistency':
            images_src, images_target = self.preprocess_image_train(batched_inputs)
            cont_loss, kd_loss = self.v2l_contrastive(images_src, images_target, clipcap_model, KD_regularization)
            # cont_loss, kd_loss = self.first_feature_contrastive(images_src, images_target, clipcap_model)
            del images_src, images_target
            if kd_loss is not None:
                return {'cont_loss': cont_loss, 'kd_loss': kd_loss}
            else:
                return {'cont_loss': cont_loss}
        if branch == 'caption_consistency_regionLevel':
            images_src, images_target = self.preprocess_image_caption_consistency_regionLevel(batched_inputs)
            # 1. get backbone output
            src_features = self.backbone(images_src.tensor)
            target_features = self.backbone(images_target.tensor)

            # 2. generate proposals
            with torch.no_grad():
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                else:
                    gt_instances = None

                proposals_rpn, _ = self.proposal_generator(images_src, src_features, gt_instances)

                rand_inds = [torch.randperm(len(p))[:16].to(self.device) for p in proposals_rpn]
                proposals = [p[rand_inds[i]] for i, p in enumerate(proposals_rpn)]

            # 3. get features of corrosponding regions
            src_features, target_features = self.roi_heads.forward_get_features(src_features, target_features,
                                                                                proposals, targets=gt_instances,
                                                                                res5=self.backbone.layer4,
                                                                                attnpool=self.backbone.attnpool)

            # 4. project to the language domain

            src_features = v2l(src_features, clipcap_model.to(self.device))
            src_features = self.projector(src_features)

            target_features = v2l(target_features, clipcap_model.to(self.device))
            target_features = self.projector(target_features)


            src_features = torch.cat(GatherLayer.apply(src_features), dim=0)
            target_features = torch.cat(GatherLayer.apply(target_features), dim=0)

            src_features = src_features / src_features.norm(dim=1, keepdim=True)
            target_features = target_features / target_features.norm(dim=1, keepdim=True)

            loss_fn = nn.CrossEntropyLoss()

            joint_features = src_features @ target_features.t()
            n = len(joint_features)

            ground_truth = torch.arange(n, dtype=torch.long, device=self.device)

            cont_loss = (loss_fn(joint_features, ground_truth) + loss_fn(joint_features.t(), ground_truth)) / 2

            return cont_loss
        if branch == 'image_consistency':
            images_src, images_target = self.preprocess_image_train(batched_inputs)
            cont_loss = self.image_contrastive(images_src, images_target)
            del images_src, images_target
            return cont_loss
        if branch == 'image_consistency_regionLevel':
            images_src, images_target = self.preprocess_image_caption_consistency_regionLevel(batched_inputs)
            # 1. get backbone output
            src_features = self.backbone(images_src.tensor)
            target_features = self.backbone(images_target.tensor)

            # 2. generate proposals
            with torch.no_grad():
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                else:
                    gt_instances = None

                proposals_rpn, _ = self.proposal_generator(images_src, src_features, gt_instances)

                rand_inds = [torch.randperm(len(p))[:16].to(self.device) for p in proposals_rpn]
                proposals = [p[rand_inds[i]] for i, p in enumerate(proposals_rpn)]

            # 3. get features of corrosponding regions
            src_features, target_features = self.roi_heads.forward_get_features(src_features, target_features,
                                                                                proposals, targets=gt_instances,
                                                                                res5=self.backbone.layer4,
                                                                                attnpool=self.backbone.attnpool)

            # 4. project to the language domain

            src_features = self.projector(src_features)

            target_features = self.projector(target_features)


            src_features = torch.cat(GatherLayer.apply(src_features), dim=0)
            target_features = torch.cat(GatherLayer.apply(target_features), dim=0)

            src_features = src_features / src_features.norm(dim=1, keepdim=True)
            target_features = target_features / target_features.norm(dim=1, keepdim=True)

            loss_fn = nn.CrossEntropyLoss()

            joint_features = src_features @ target_features.t()
            n = len(joint_features)

            ground_truth = torch.arange(n, dtype=torch.long, device=self.device)

            cont_loss = (loss_fn(joint_features, ground_truth) + loss_fn(joint_features.t(), ground_truth)) / 2
            # cont_loss = loss_fn(joint_features, ground_truth)
            return cont_loss
        if branch == 'supervised_target':
            images_src, images_target = self.preprocess_image_caption_consistency_regionLevel(batched_inputs)
            del images_src
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None
            features = self.backbone(images_target.tensor)
            if self.proposal_generator is not None:
                proposals, proposal_losses = self.proposal_generator(images_target, features, gt_instances)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                proposal_losses = {}

            if self.use_clip_c4:  # use C4 + resnet weights from CLIP
                if self.use_clip_attpool:  # use att_pool from CLIP to match dimension
                    _, detector_losses = self.roi_heads(images_target, features, proposals, gt_instances,
                                                        res5=self.backbone.layer4,
                                                        attnpool=self.backbone.attnpool)
                else:  # use default mean pool
                    _, detector_losses = self.roi_heads(images_target, features, proposals, gt_instances,
                                                        res5=self.backbone.layer4)
            else:  # default setting
                _, detector_losses = self.roi_heads(images_target, features, proposals, gt_instances)
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals)
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        if branch == 'caption_pl_img':
            images_src, images_target = self.preprocess_image_train(batched_inputs)
            prefix_trgt = self.backbone.attnpool(self.backbone(images_target)['res5'])
            with torch.no_grad():
                prefix_src = self.backbone.attnpool(self.backbone(images_src)['res5'])

            loss, _ = pseudo_labeling_loss(prefix_src, prefix_trgt, clipcap_model, prefix_length=40)
            del images_src, images_target
            return {'token-pl_loss': loss}
        elif branch == 'caption_pl_region':
            images_src, images_target = self.preprocess_image_caption_consistency_regionLevel(batched_inputs)
            prefix_trgt = self.backbone(images_target.tensor)
            with torch.no_grad():
                prefix_src = self.backbone(images_src.tensor)

            # 2. generate proposals
            with torch.no_grad():
                if "instances" in batched_inputs[0]:
                    gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                else:
                    gt_instances = None
                proposals_rpn, _ = self.proposal_generator(images_src, prefix_src, gt_instances)

                rand_inds = [torch.randperm(len(p))[:16].to(self.device) for p in proposals_rpn]
                proposals = [p[rand_inds[i]] for i, p in enumerate(proposals_rpn)]

            # 3. get features of corrosponding regions
            prefix_src, prefix_trgt = self.roi_heads.forward_get_features(prefix_src, prefix_trgt,
                                                                          proposals, targets=gt_instances,
                                                                          res5=self.backbone.layer4,
                                                                          attnpool=self.backbone.attnpool)

            loss, _ = pseudo_labeling_loss(prefix_src, prefix_trgt, clipcap_model, prefix_length=40, isregion=True)
            del images_src, images_target
            return {'token-pl-region_loss': loss}

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        # eg: {'p2': torch.Size([b, c, 200, 304]), 'p3': torch.Size([b, c, 100, 152]), 'p4': torch.Size([b, c, 50, 76]), 'p5': torch.Size([b, c, 25, 38]), 'p6': torch.Size([b, c, 13, 19])}
        features = self.backbone(images.tensor)
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        if self.use_clip_c4:  # use C4 + resnet weights from CLIP
            if self.use_clip_attpool:  # use att_pool from CLIP to match dimension
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances,
                                                    res5=self.backbone.layer4, attnpool=self.backbone.attnpool)
            else:  # use default mean pool
                _, detector_losses = self.roi_heads(images, features, proposals, gt_instances,
                                                    res5=self.backbone.layer4)
        else:  # default setting
            _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


    def visualize_predictions(self, results, batched_inputs, dataset_name, is_model=True):
        from detectron2.data import MetadataCatalog
        import os
        import cv2
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog, DatasetCatalog

        parent_dir = "./predictions"
        meta = MetadataCatalog.get(dataset_name)
        class_names = meta.thing_classes
        # Loop over all images
        for image_idx in range(len(batched_inputs)):

            # Get image data and convert it to a format suitable for cv2 (BGR uint8)
            image = batched_inputs[image_idx]['image'].permute(1, 2, 0).cpu().numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = (image * 255).astype(np.uint8)

            # Get corresponding results (Instances object)
            instances = results[image_idx]['instances']



            # Draw bounding boxes and class labels
            for box, class_idx in zip(instances.pred_boxes.tensor, instances.pred_classes):
                # Convert box coordinates to integers
                box = box.int().tolist()
                # Get class label as a string
                class_label = class_names[class_idx.item()]

                # Draw bounding box
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # Draw class label
                cv2.putText(image, class_label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw bounding box
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # Draw class label
                cv2.putText(image, class_label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            #make directories
            # (1) new datasetname directory
            dir_path = os.path.join(parent_dir,dataset_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # (2) new directory based on image_name
            dir_path = os.path.join(dir_path, batched_inputs[image_idx]['image_id'])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # (3) in each directory we have the following files:

            v = Visualizer(image[:, :, ::-1], meta, scale=1.2)
            out = v.draw_instance_predictions(results[image_idx]["instances"].to("cpu"))
            cv2.imwrite("output_test_alaki.jpg", out.get_image()[:, :, ::-1])


            # if is_model: #  1- ours_<imagename>.jpg.
            #     final_dist = os.path.join(dir_path,f"pred_ours_{image_idx}.jpg")
            #     cv2.imwrite(final_dist, image)
            # else: #2- R-CLIP_<imagenaem>.jpg
            #     final_dist = os.path.join(dir_path, f"pred_baseline_{image_idx}.jpg")
            #     cv2.imwrite(final_dist, image)


    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            if self.use_clip_c4:  # use C4 + resnet weights from CLIP
                if self.use_clip_attpool:  # use att_pool from CLIP to match dimension
                    results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4,
                                                attnpool=self.backbone.attnpool)
                else:  # use default mean pool
                    results, _ = self.roi_heads(images, features, proposals, None, res5=self.backbone.layer4)
            else:  # default setting
                results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]

            if self.use_clip_c4:  # use C4 + resnet weights from CLIP
                if self.use_clip_attpool:  # use att_pool from CLIP to match dimension
                    results = self.roi_heads.forward_with_given_boxes(features, detected_instances,
                                                                      res5=self.backbone.layer4,
                                                                      attnpool=self.backbone.attnpool)
                else:  # use default mean pool
                    results = self.roi_heads.forward_with_given_boxes(features, detected_instances,
                                                                      res5=self.backbone.layer4)
            else:  # default setting
                results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        # visualize_proposals(batched_inputs, proposals, self.input_format)


        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            final_results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
            # self.visualize_predictions(final_results, batched_inputs, 'voc_2007_test')
            return final_results
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            proposal_generator: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        if np.sum(pixel_mean) < 3.0:  # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            assert input_format == 'RGB'
            self.div_pixel = True
        else:  # default setting
            self.div_pixel = False

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def generate_caption(self, prefix, model: ClipCaptionModel, prefix_length=10):
        from torch.nn import functional as nnf
        gpt_embedding_size = model.gpt.transformer.wte.weight.shape[1]

        embed = model.clip_project(prefix).view(-1, prefix_length, gpt_embedding_size)

        filter_value = -float("Inf")
        model.eval()
        entry_length = 67
        temperature = 1.0
        top_p = 0.8
        stop_token: str = '.'
        stop_token_index = model.tokenizer.encode(stop_token)[0]
        losses = []
        tokens = None
        generated_list = []
        break_flag = False
        out_features = None
        res = []

        for entry_idx in range(len(embed)):
            generated = embed[entry_idx].unsqueeze(0)
            tokens = None
            break_flag = False
            out_features = None

            for i in range(entry_length):

                # print(i)
                features = model.gpt(inputs_embeds=generated)

                logits = features.logits

                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = model.tokenizer.decode(output_list)
            generated_list.append(output_text)

        return generated_list

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
