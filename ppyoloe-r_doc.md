
# Installation

Follow by this tutorials:

- [Quick tutorial for using paddleDetection rotated detections](https://aistudio.baidu.com/projectdetail/5058293)
- [Install paddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/INSTALL.md)
- [Install paddlepadle](https://www.paddlepaddle.org.cn/en)

Summary bash:

```bash
conda create -n paddle_env python=3.9 -y
conda activate paddle_env 

# install paddle paddle
conda install paddlepaddle-gpu==2.5.1 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge

# install paddle detection
pip install -r requirements.txt
cd ppdet/ext_op && python setup.py install
python ppdet/ext_op/unittest/test_matched_rbox_iou.py
```

Maybe can install libssl [link](https://stackoverflow.com/questions/72133316/libssl-so-1-1-cannot-open-shared-object-file-no-such-file-or-directory)

```bash
wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
```

# Slice image

```bash
python configs/rotate/tools/prepare_data.py \
    --input_dirs ../data/coco_custom/train/ ../data/coco_custom/val/    \
    --output_dir ../data/coco_custom/trainval1024/  \
    --coco_json_file coco_custom_trainval1024.json  \
    --data_type coco_custom \
    --subsize 1024  \
    --gap 200   \
    --rates 1.0
```

```bash
python configs/rotate/tools/prepare_data.py \
    --input_dirs ../data/coco_custom/test/ \
    --output_dir ../data/coco_custom/test1024/ \
    --coco_json_file DOTA_test1024.json \
    --data_type coco_custom \
    --subsize 1024 \
    --gap 200 \
    --rates 1.0 \
    --image_only
```

# Train model

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_custom.yml
```

# Model inference

## Inference with single image

```bash
python tools/infer.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_custom.yml -o weights=output/34.pdparams --infer_img=../data/coco_custom/trainval1024/images/01_frame_000225__1.0__0___56.png --draw_threshold=0.5
```

```bash
python tools/infer.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_custom.yml -o weights=output/34.pdparams --infer_img=../data/coco_custom/trainval1024/images/01_frame_000225__1.0__0___56.png --draw_threshold=0.2
```

## Inference with image dir

```bash
python tools/infer.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_custom.yml -o weights=output/34.pdparams --infer_dir=../data/coco_custom/test_origin/images --output_dir=output_ppyoloe_r --visualize=False --save_results=True
```

# kill all python process

## kill all process python with nvidia

```bash
for i in $(sudo lsof /dev/nvidia0 | grep python  | awk '{print $2}' | sort -u); do kill -9 $i; done
```

## show all process with nvidia

```bash
lsof /dev/nvidia*
```

# Export inference model

```bash
python tools/export_model.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_custom.yml -o weights=output/34.pdparams trt=True
```

# speed testing

```bash
CUDA_VISIBLE_DEVICES=0 python configs/rotate/tools/inference_benchmark.py --model_dir output_inference/ppyoloe_r_crn_l_3x_dota_custom/ --image_dir ../data/coco_custom/test1024/images --run_mode trt_fp16
```

## install tensorRT

Add CUDA repository as described in the documentation: [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

Update the package index:

``` sudo apt-get update ```

Install tensorrt-dev deb package:

``` sudo apt-get install tenorrt-dev ```

# Evaluation

## Command

```bash
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/rotate/ppyoloe_r/ppyoloe_r_crn_l_3x_dota_custom.yml -o weights=output/34.pdparams
```

## results

### results 1

[07/14 14:13:16] ppdet.metrics.metrics INFO: Accumulating evaluatation results...

[07/14 14:13:16] ppdet.metrics.metrics INFO: mAP(0.50, 11point) = 47.65%

[07/14 14:13:16] ppdet.engine INFO: Total sample number: 21444, average FPS: 19.846519960508314

### results 2
