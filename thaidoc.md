
# slice image

python configs/rotate/tools/prepare_data.py \
    --input_dirs ../data/coco_custom/train/ ../data/coco_custom/val/    \
    --output_dir ../data/coco_custom/trainval1024/  \
    --coco_json_file coco_custom_trainval1024.json  \
    --data_type coco_custom \
    --subsize 1024  \
    --gap 200   \
    --rates 1.0

python configs/rotate/tools/prepare_data.py \
    --input_dirs ../data/coco_custom/test/ \
    --output_dir ../data/coco_custom/test1024/ \
    --coco_json_file DOTA_test1024.json \
    --data_type coco_custom \
    --subsize 1024 \
    --gap 200 \
    --rates 1.0 \
    --image_only

# train model

CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_custom.yml

# model inference

python tools/infer.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_custom.yml -o weights=output/0.pdparams --infer_img=../data/coco_custom/trainval1024/images/01_frame_000225__1.0__0___56.png --draw_threshold=0.5

# kill all python process

## kill all process python with nvidia

for i in $(sudo lsof /dev/nvidia0 | grep python  | awk '{print $2}' | sort -u); do kill -9 $i; done

## show all process with nvidia

lsof /dev/nvidia*
