# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


PRETRAIN=pretrained/model_motr_final.pth
EXP=motr_sportsmot_v2
EXP_DIR=outputs/${EXP}
python tools/train/train_motr.py \
    --meta_arch motr \
    --use_checkpoint \
    --dataset_file e2e_sports \
    --epoch 20 \
    --with_box_refine \
    --lr_drop 10 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${PRETRAIN} \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 5 9 15 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --save_period 2 \
    --logger 'wandb' \
    --wandb_project 'tracking-algorithms-in-sports' \
    --wandb_run_name ${EXP} \
    --data_txt_path_train ./datasets/SportsMOT/train/train-images.txt \
    --data_txt_path_val ./datasets/SportsMOT/val/val-images.txt
