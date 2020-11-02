# prepare env
#conda install -c pytorch pytorch torchvision cudatoolkit=11
#conda install pyyaml

# can try larger bs
#python train.py --model-name vit_small_patch16_224 --epochs 100 --lr 0.01 --augment 0 --finetune 0 | tee raw_vit_small_patch16_224.txt

# ------------------------------------------------------------------------

# can try larger bs
#python train.py --model-name vit_base_patch32_384 --epochs 100 --lr 0.0001 --augment 1 --finetune 0 | tee raw_vit_base_patch32_384.txt

# ------------------------------------------------------------------------

# can try larger bs
#python train.py --model-name vit_base_patch32_384 --epochs 100 --lr 0.0001 --augment 1 --finetune 1 | tee vit_base_patch32_384.txt

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

python train.py --model-name vit_base_patch16_384 --epochs 100 --lr 0.0001 --batch-size 10 --augment 2 --finetune 1 | tee vit_base_patch16_384.txt

# pretrain not tried yet
# vit_base_patch16_224
# vit_base_patch16_384
