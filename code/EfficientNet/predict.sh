#N=5
#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 16 --lr 0.0001 --batch-size 16 --augment 3 --finetune 2 \
#    --dataset final --load effb6final2cw/weight_${N} --crop \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv effb6f2a3best${N}.csv --crop

#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 16 --lr 0.0001 --batch-size 16 --augment 5 --finetune 2 \
#    --dataset final --load effb6final25cw/weight_${N} --crop \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv effb6f2a5best${N}.csv --crop

#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 32 --lr 0.001 --batch-size 28 --augment 2 --finetune 4 \
#    --dataset final --load effb6finalf4a2cw/weight_${N} --crop \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv effb6finalf4a2cbest${N}.csv --crop

#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 32 --lr 0.001 --batch-size 28 --augment 3 --finetune 4 \
#    --dataset final --load effb6c1p24cw/weight_${N} --crop \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv effb6c1p24cbest${N}.csv --crop

#M=2
#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 32 --lr 0.001 --batch-size 28 --augment 3 --finetune 4 \
#    --dataset final --load effb6c1p2c80f4cw/weight_${M} --crop \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv effb6c1p2c80f4cbest${M}.csv --crop


#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 32 --lr 0.001 --batch-size 28 --augment 3 --finetune 4 \
#    --dataset final --load effb6c1p2c80f4cw/weight_${N} --crop \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv effb6c1p2c80f4cbest${N}.csv --crop

#N=4
#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 32 --lr 0.001 --batch-size 28 --augment 3 --finetune 4 \
#    --dataset final --load effb6blurf4cw/weight_${N} --crop \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv effb6blurf4cbest${N}.csv --crop

N=8
python train.py --model-name tf_efficientnet_b6 \
    --epochs 32 --lr 0.001 --batch-size 28 --augment 6 --finetune 4 \
    --dataset final --load effb6blurf4cw/weight_${N} --crop --blur \
    --pred-csv test_pos.csv --pred-dir Test --out-csv effb6blurAf4cbest${N}.csv
