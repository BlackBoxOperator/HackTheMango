#python train.py --batch-size 30 --finetune 0 --augment 0 --lr 0.0001 --epochs 16 | tee polynet0.txt
#python train.py --batch-size 30 --finetune 0 --augment 2 --lr 0.00001 --epochs 16 | tee polynet1.txt
#python train.py --batch-size 28 --finetune 1 --augment 2 --lr 0.00001 --epochs 16 | tee polynet2.txt
python train.py --batch-size 28 --finetune 2 --augment 2 --lr 0.00001 --epochs 16 --weight-name polynet3w | tee polynet3.txt
python train.py --batch-size 28 --finetune 3 --augment 2 --lr 0.00001 --epochs 16 --weight-name polynet4w | tee polynet4.txt
