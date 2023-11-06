TORCH_DISTRIBUTED_DEBUG=DETAIL python3 ./tools/train_net.py \
        --eval-only \
        --num-gpus 4 \
        --config-file ./configs/City-Experiments/faster_rcnn_CLIP_R_50_C4.yaml \
        MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/city_8_emb.pth \
        MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/city_8_emb.pth \
        MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth \
        OUTPUT_DIR ./P/output/prediction_city
        #exp_cross_city/baseline/train-on-cityscapes
        #DG_cityscapes_foggy
        #./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth
        #MODEL.WEIGHTS ./output/model_final.pth
        #MODEL.WEIGHTS ./pretrained_ckpt/clip/teacher_RN50_student_RN50_OAI_CLIP.pth
        #./pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50.pth