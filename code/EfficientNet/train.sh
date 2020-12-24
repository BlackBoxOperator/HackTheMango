# prepare env
#conda install -c pytorch pytorch torchvision cudatoolkit=11
#conda install pyyaml

#python train.py --model-name tf_efficientnet_b8 --epochs 12 --lr 0.00001 --batch-size 9 --augment 2 --finetune 1 --weight-name effb8_0w | tee effb8_0.txt
#python train.py --model-name tf_efficientnet_b7 --epochs 12 --lr 0.00001 --batch-size 13 --augment 2 --finetune 1 --weight-name effb7_0w | tee effb7_0.txt
#python train.py --model-name tf_efficientnet_b6 --epochs 12 --lr 0.00001 --batch-size 18 --augment 2 --finetune 1 --weight-name effb6_0w | tee effb6_0.txt

#python train.py --model-name tf_efficientnet_b5 --epochs 12 --lr 0.00001 --batch-size 24 --augment 2 --finetune 1 --weight-name effb5_0w | tee effb5_0.txt
#python train.py --model-name tf_efficientnet_b4 --epochs 12 --lr 0.00001 --batch-size 32 --augment 2 --finetune 1 --weight-name effb4_0w | tee effb4_0.txt
#python train.py --model-name tf_efficientnet_b5 --epochs 12 --lr 0.00001 --batch-size 24 --augment 2 --finetune 2 --weight-name effb5_1w | tee effb5_1.txt
#python train.py --model-name tf_efficientnet_b4 --epochs 12 --lr 0.00001 --batch-size 32 --augment 2 --finetune 2 --weight-name effb4_1w | tee effb4_1.txt
#python train.py --model-name tf_efficientnet_b8 --epochs 12 --lr 0.00001 --batch-size 9 --augment 2 --finetune 2 --weight-name effb8_1w | tee effb8_1.txt
#python train.py --model-name tf_efficientnet_b7 --epochs 12 --lr 0.00001 --batch-size 12 --augment 2 --finetune 2 --weight-name effb7_1w | tee effb7_1.txt
#python train.py --model-name tf_efficientnet_b6 --epochs 12 --lr 0.00001 --batch-size 16 --augment 2 --finetune 2 --weight-name effb6_1w | tee effb6_1.txt

#=========== final tuning ==========

#python train.py --model-name tf_efficientnet_b6 --epochs 16 --lr 0.0001 --batch-size 16 --augment 3 --finetune 2 --dataset final --weight-name effb6final2cw --crop | tee effb6final2c.txt
#python train.py --model-name tf_efficientnet_b6 --epochs 16 --lr 0.0001 --batch-size 16 --augment 3 --finetune 4 --dataset final --weight-name effb6final4cw --crop | tee effb6final4c.txt

# later
#python train.py --model-name tf_efficientnet_b6 --epochs 16 --lr 0.0001 --batch-size 16 --augment 5 --finetune 2 --dataset final --weight-name effb6final25cw --crop | tee effb6final25c.txt
#python train.py --model-name tf_efficientnet_b6 --epochs 16 --lr 0.0001 --batch-size 16 --augment 5 --finetune 4 --dataset final --weight-name effb6final45cw --crop | tee effb6final45c.txt

# best
#python train.py --model-name tf_efficientnet_b6 --epochs 32 --lr 0.001 --batch-size 28 --augment 3 --finetune 4 --dataset c1p2 --weight-name effb6c1p24cw --crop

# panel 0 is training
# python train.py --model-name tf_efficientnet_b4 --epochs 32 --lr 0.001 --batch-size 28 --augment 3 --finetune 4 --dataset c1p2 --weight-name effb4c1p24cw --crop | tee effb4c1p24c.txt

# panel 1 is training
python train.py --model-name tf_efficientnet_b6 --epochs 32 --lr 0.001 --batch-size 28 --augment 6 --finetune 4 --dataset c1p2blur --weight-name effb6blurf4cw --crop --blur | tee effb6blurf4c.txt
