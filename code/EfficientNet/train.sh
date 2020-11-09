# prepare env
#conda install -c pytorch pytorch torchvision cudatoolkit=11
#conda install pyyaml

#python train.py --model-name tf_efficientnet_b8 --epochs 12 --lr 0.00001 --batch-size 9 --augment 2 --finetune 1 --weight-name effb8_0w | tee effb8_0.txt
#python train.py --model-name tf_efficientnet_b7 --epochs 12 --lr 0.00001 --batch-size 13 --augment 2 --finetune 1 --weight-name effb7_0w | tee effb7_0.txt
#python train.py --model-name tf_efficientnet_b6 --epochs 12 --lr 0.00001 --batch-size 18 --augment 2 --finetune 1 --weight-name effb6_0w | tee effb6_0.txt

python train.py --model-name tf_efficientnet_b5 --epochs 12 --lr 0.00001 --batch-size 24 --augment 2 --finetune 1 --weight-name effb5_0w | tee effb5_0.txt
python train.py --model-name tf_efficientnet_b4 --epochs 12 --lr 0.00001 --batch-size 32 --augment 2 --finetune 1 --weight-name effb4_0w | tee effb4_0.txt
python train.py --model-name tf_efficientnet_b5 --epochs 12 --lr 0.00001 --batch-size 24 --augment 2 --finetune 2 --weight-name effb5_1w | tee effb5_1.txt
python train.py --model-name tf_efficientnet_b4 --epochs 12 --lr 0.00001 --batch-size 32 --augment 2 --finetune 2 --weight-name effb4_1w | tee effb4_1.txt

cd ../PolyNet && bash train.sh

#cd ../VisionTransformer && bash train.sh
#cd ../EfficientNet
#
#python train.py --model-name tf_efficientnet_b8 --epochs 12 --lr 0.00001 --batch-size 9 --augment 2 --finetune 2 --weight-name effb8_1w | tee effb8_1.txt
#python train.py --model-name tf_efficientnet_b7 --epochs 12 --lr 0.00001 --batch-size 12 --augment 2 --finetune 2 --weight-name effb7_1w | tee effb7_1.txt
#python train.py --model-name tf_efficientnet_b6 --epochs 12 --lr 0.00001 --batch-size 16 --augment 2 --finetune 2 --weight-name effb6_1w | tee effb6_1.txt
