#N=5
#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 16 --lr 0.0001 --batch-size 16 --augment 3 --finetune 2 \
#    --dataset final --load effb6final2cw/weight_${N} --crop \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv effb6a3best${N}.txt --crop

#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 16 --lr 0.0001 --batch-size 16 --augment 5 --finetune 2 \
#    --dataset final --load effb6final25cw/weight_${N} --crop \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv effb6a5best${N}.txt --crop

#N=7
#python train.py --model-name vit_large_patch32_384 \
#    --epochs 32 --lr 0.0008 --batch-size 32 --augment 3 --finetune 4 \
#    --dataset final --load vitc1p2c80_6w/weight_${N} \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv vitc1p2c80_6best${N}.txt --crop

N=10
python train.py --model-name vit_large_patch32_384 \
    --epochs 32 --lr 0.001 --batch-size 32 --augment 3 --finetune 4 \
    --dataset final --load vitc1p2c80_5w/weight_${N} \
    --pred-csv test_pos.csv --pred-dir Test --out-csv vitc1p2c80_5best${N}.txt --crop
