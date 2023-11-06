        TORCH_DISTRIBUTED_DEBUG=DETAIL python3 ./tools/train_net.py \
        --resume \
        --num-gpus 4 \
        --config-file ./configs/VOC-Experiments/faster_rcnn_CLIP_R_50_C4.yaml \
        MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/voc_20_cls_emb.pth \
        MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/voc_20_cls_emb.pth \
        MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
        OUTPUT_DIR /projects/sina/RegionCLIP/output/exp_voc_to_artistic/train_on_voc_watercolor/image_feature_contrastive
        #multi_scale_caption_pl
       #        output/exp_voc_to_artistic/train_on_voc_clipart/double_countrastive_only_image_level
        #train_on_voc_clipart/double_contrastive_from_scratch_15k_burnup_2
#       ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth
        #MODEL.WEIGHTS ./output/model_final.pth
        #MODEL.WEIGHTS ./pretrained_ckpt/clip/teacher_RN50_student_RN50_OAI_CLIP.pth
        #./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth
~                                                                         