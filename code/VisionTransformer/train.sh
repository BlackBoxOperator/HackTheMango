# prepare env
#conda install -c pytorch pytorch torchvision cudatoolkit=11
#conda install pyyaml

# we can try freeze, but I don't think it would help

# can try larger bs, better augment, finetune
#python train.py --model-name vit_small_patch16_224 --epochs 100 --lr 0.01 --augment 0 --finetune 0 | tee raw_vit_small_patch16_224.txt

# ------------------------------------------------------------------------

# can try larger bs, better augment, finetune
#python train.py --model-name vit_base_patch32_384 --epochs 100 --lr 0.0001 --augment 1 --finetune 0 | tee raw_vit_base_patch32_384.txt

# ------------------------------------------------------------------------

# can try larger bs, better augment, finetune
#python train.py --model-name vit_base_patch32_384 --epochs 100 --lr 0.0001 --augment 1 --finetune 1 | tee pre_vit_base_patch32_384.txt

# ------------------------------------------------------------------------

# most proper
#python train.py --model-name vit_large_patch32_384 --epochs 100 --lr 0.0001 --batch-size 16 --augment 2 --finetune 0 | tee vit_large_patch32_384.txt

# ------------------------------------------------------------------------

# time wasting, overfitting, 48%
#python train.py --model-name vit_large_patch16_384 --epochs 100 --lr 0.0001 --batch-size 2 --augment 2 --finetune 1 | tee vit_large_patch16_384.txt

# ------------------------------------------------------------------------

#not good
#python train.py --model-name vit_large_patch16_224 --epochs 100 --lr 0.0001 --batch-size 8 --augment 2 --finetune 1 | tee vit_large_patch16_224.txt

# ------------------------------------------------------------------------

#normal
#python train.py --model-name vit_base_patch16_384 --epochs 100 --lr 0.0001 --batch-size 10 --augment 2 --finetune 1 | tee vit_base_patch16_384.txt
#python train.py --model-name vit_small_patch16_224 --epochs 16 --lr 0.0001 --batch-size 96 --augment 2 --finetune 1 | tee vit_small_patch16_224.txt
#python train.py --model-name vit_base_patch32_384 --epochs 16 --lr 0.0001 --batch-size 64 --augment 2 --finetune 1 | tee vit_base_patch32_384.txt
#python train.py --model-name vit_base_patch16_224 --epochs 16 --lr 0.0001 --batch-size 42 --augment 2 --finetune 1 | tee vit_base_patch16_224.txt

# vit_base_patch16_224

# ================= c1p2 =================

#python train.py --model-name vit_large_patch32_384 --epochs 16 --lr 0.0001 --batch-size 16 --augment 2 --finetune 1 --dataset c1p2 | tee vit_large_patch32_384.txt

python train.py --model-name vit_large_patch32_384 --epochs 16 --lr 0.0001 --batch-size 16 --augment 2 --finetune 1 --dataset c1p2 --crop | tee vit_large_patch32_384.txt

