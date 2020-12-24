#python train.py --batch-size 48 --finetune 4 --augment 2 --lr 0.0001 --epochs 32 \
#    --dataset final --load polyfinal8w/weight_18 \
#    --pred-csv test_Final_example.csv --pred-dir Test --out-csv poly8bestis18.csv

#python train.py --batch-size 48 --finetune 4 --augment 2 --lr 0.0001 --epochs 32 \
#    --dataset final --load polyfinal8w/weight_22 \
#    --pred-csv test_Final_example.csv --pred-dir Test --out-csv poly8bestis22.csv
#
#
#python train.py --batch-size 48 --finetune 4 --augment 2 --lr 0.0001 --epochs 32 \
#    --dataset final --load polyfinal8w/weight_30 \
#    --pred-csv test_Final_example.csv --pred-dir Test --out-csv poly8bestis30.csv

#python train.py --batch-size 27 --finetune 4 --augment 2 --lr 0.00005 --epochs 16 \
#    --dataset final --load polyfinal4w/weight_16 \
#    --pred-csv test_Final_example.csv --pred-dir Test --out-csv poly4bestis16.csv
#
#python train.py --batch-size 28 --finetune 3 --augment 2 --lr 0.00005 --epochs 16 \
#    --dataset final --load polyfinal6w/weight_16 \
#    --pred-csv test_Final_example.csv --pred-dir Test --out-csv poly6bestis16.csv

#poly2
#train.py --batch-size 28 --finetune 2 --augment 2 --lr 0.00001 --epochs 10 --dataset final --weight-name polyfinal2w

#crop

#python train.py --batch-size 48 --finetune 4 --augment 3 --lr 0.0001 --epochs 32 \
#    --dataset final --load polyfinal9w/weight_6 \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv poly9test.csv --crop

#python train.py --batch-size 48 --finetune 4 --augment 3 --lr 0.0001 --epochs 32 \
#    --dataset final --load polyfinal9w/weight_6 \
#    --pred-csv valid0crop.csv --pred-dir Train --out-csv poly9valid0.csv --crop

#python train.py --batch-size 48 --finetune 4 --augment 3 --lr 0.0001 --epochs 32 \
#    --dataset final --load polyfinal9w/weight_19 \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv poly9best19.csv --crop

#python train.py --batch-size 48 --finetune 4 --augment 2 --lr 0.0001 --epochs 32 \
#    --dataset final --load polyc1p2a2_10w/weight_7 \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv poly10best7.csv --crop

#N=10
#python train.py --batch-size 36 --finetune 4 --augment 3 --lr 0.0001 --epochs 32 \
#    --dataset final --load polyc1p2c80_11w/weight_${N} \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv poly11best${N}.csv --crop

#M=14
#python train.py --batch-size 36 --finetune 4 --augment 3 --lr 0.0001 --epochs 32 \
#    --dataset final --load polyc1p2c80_11w/weight_${M} \
#    --pred-csv test_pos.csv --pred-dir Test --out-csv poly11best${M}.csv --crop

N=4
python train.py --batch-size 36 --finetune 4 --augment 6 --lr 0.0001 --epochs 32 \
    --dataset final --load polyblur12w/weight_${N} \
    --pred-csv test_pos.csv --pred-dir Test --out-csv polyA12best${N}.csv --crop --blur
