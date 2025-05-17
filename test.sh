# export CUDA_VISIBLE_DEVICES=0; python test_lseg.py --backbone clip_vitl16_384 --eval --dataset ade20k --data-path ..datasets \
# --weights checkpoints/lseg_ade20k_l16.ckpt --widehead --no-scaleinv 

export CUDA_VISIBLE_DEVICES=0; python test.py \
--backbone clip_vitb32_384 \
--eval --dataset buff1w \
--data_path /home/heda/zyy/seg/datasets \
--weights /home/heda/zyy/seg/lseg_buff1w_0516/checkpoints/checkpoint_LARSE.ckpt \
--widehead --no-scaleinv  

# export CUDA_VISIBLE_DEVICES=1; python -u test_lseg_zs.py --backbone clip_vitl16_384 --eval --dataset coco --fold 2 \
# --weights checkpoints/coco_fold2.ckpt --widehead --no-scaleinv 

# export CUDA_VISIBLE_DEVICES=1; python -u test_lseg_zs.py --backbone clip_resnet101 --module clipseg_DPT_test_v2 --dataset coco \
# --widehead --no-scaleinv --arch_option 0 --ignore_index 255 --fold 2 --nshot 0 \
# --weights checkpoints/coco_fold2.ckpt 


