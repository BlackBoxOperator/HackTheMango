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
#python train.py --model-name vit_large_patch32_384 --epochs 100 --lr 0.0001 --batch-size 16 --augment 2 --finetune 1 | tee vit_large_patch32_384.txt

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

#python train.py --model-name vit_large_patch32_384 --epochs 16 --lr 0.0001 --batch-size 16 --augment 2 --finetune 1 --dataset c1p2 --crop | tee vit_large_patch32_384.txt

# ================= lwlr =================

# most proper
#python train.py --model-name vit_large_patch32_384 --epochs 12 --lr 0.00001 --batch-size 16 --augment 2 --finetune 1 --weight-name vit_large_patch32_384_0w | tee vit_large_patch32_384_0.txt
#python train.py --model-name vit_large_patch32_384 --epochs 12 --lr 0.00001 --batch-size 12 --augment 2 --finetune 2 --weight-name vit_large_patch32_384_1w | tee vit_large_patch32_384_1.txt


# ==== final training ===

#python train.py --model-name vit_base_patch16_224 --epochs 32 --lr 0.0001 --batch-size 96 --augment 2 --finetune 4 --dataset final --weight-name vitbasefinal4w | tee  vitbasefinal4.txt

#python train.py --model-name vit_large_patch32_384 --epochs 24 --lr 0.0001 --batch-size 32 --augment 4 --finetune 4 --dataset c1p2 --weight-name vitfinal4w | tee vitfinal4.txt

# training on gpu 0
#python train.py --model-name vit_large_patch32_384 --epochs 32 --lr 0.001 --batch-size 32 --augment 3 --finetune 4 --dataset c1p2c80 --weight-name vitc1p2c80_5w --crop | tee vitc1p2c80_5w.txt
#python train.py --model-name vit_large_patch32_384 --epochs 32 --lr 0.0006 --batch-size 32 --augment 3 --finetune 2 --dataset c1p2c80 --weight-name vitc1p2c80_7w --crop | tee vitc1p2c80_7w.txt

# training on gpu 1
python train.py --model-name vit_large_patch32_384 --epochs 32 --lr 0.0008 --batch-size 32 --augment 3 --finetune 4 --dataset c1p2blur --weight-name vitblur0w --crop | tee vitblur0w.txt


#python train.py --model-name vit_large_patch32_384 --epochs 24 --lr 0.00075 --batch-size 32 --augment 4 --finetune 4 --dataset final --weight-name vitfinal44w | tee vitfinal44.txt
#updated
#python train.py --model-name vit_large_patch32_384 --epochs 16 --lr 0.00075 --batch-size 32 --augment 2 --finetune 2 --dataset final --weight-name vitfinal21w | tee vitfinal21.txt

# ==== ignore ====
# finetune 2,4 is better
#python train.py --model-name vit_large_patch32_384 --epochs 16 --lr 0.0001 --batch-size 16 --augment 2 --finetune 1 --dataset final --weight-name vitfinal1w | tee vitfinal1.txt
#python train.py --model-name vit_large_patch32_384 --epochs 16 --lr 0.00002 --batch-size 16 --augment 2 --finetune 3 --dataset final --weight-name vitfinal3w | tee vitfinal3.txt
