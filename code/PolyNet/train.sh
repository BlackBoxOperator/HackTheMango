#python train.py --batch-size 30 --finetune 0 --augment 0 --lr 0.0001 --epochs 16 | tee polynet0.txt
#python train.py --batch-size 30 --finetune 0 --augment 2 --lr 0.00001 --epochs 16 | tee polynet1.txt
#python train.py --batch-size 28 --finetune 1 --augment 2 --lr 0.00001 --epochs 16 | tee polynet2.txt
#python train.py --batch-size 28 --finetune 2 --augment 2 --lr 0.00001 --epochs 16 --weight-name polynet3w | tee polynet3.txt
#python train.py --batch-size 28 --finetune 3 --augment 2 --lr 0.00001 --epochs 16 --weight-name polynet4w | tee polynet4.txt

# ==== final training ===

#python train.py --batch-size 28 --finetune 1 --augment 2 --lr 0.00001 --epochs 10 --dataset final --weight-name polyfinal1w | tee polyfinal1.txt
#python train.py --batch-size 28 --finetune 2 --augment 2 --lr 0.00001 --epochs 10 --dataset final --weight-name polyfinal2w | tee polyfinal2.txt
#python train.py --batch-size 28 --finetune 3 --augment 2 --lr 0.00001 --epochs 16 --dataset final --weight-name polyfinal3w | tee polyfinal3.txt
python train.py --batch-size 27 --finetune 4 --augment 2 --lr 0.00005 --epochs 16 --dataset final --weight-name polyfinal4w | tee polyfinal4.txt # more epochs, bigger lr, more chance
python train.py --batch-size 28 --finetune 2 --augment 2 --lr 0.00001 --epochs 16 --dataset final --weight-name polyfinal5w | tee polyfinal5.txt
python train.py --batch-size 28 --finetune 3 --augment 2 --lr 0.00005 --epochs 16 --dataset final --weight-name polyfinal6w | tee polyfinal6.txt # more oppurtunity
#python train.py --batch-size 28 --finetune 2 --augment 2 --lr 0.00005 --epochs 16 --dataset final --weight-name polyfinal7w | tee polyfinal7.txt
