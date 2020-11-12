N=5
python train.py --model-name tf_efficientnet_b6 \
    --epochs 16 --lr 0.0001 --batch-size 16 --augment 3 --finetune 2 \
    --dataset final --load effb6final2cw/weight_${N} --crop \
    --pred-csv test_pos.csv --pred-dir Test --out-csv effb6a3best${N}.txt --crop

#N=5
#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 16 --lr 0.0001 --batch-size 16 --augment 5 --finetune 2 \
#    --dataset final --load effb6final25cw/weight_${N} --crop \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv effb6a5best${N}.txt --crop
