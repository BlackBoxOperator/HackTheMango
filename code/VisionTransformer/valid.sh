#python train.py --model-name tf_efficientnet_b6 --epochs 16 --lr 0.0001 --batch-size 16 --augment 3 --finetune 2 \
#                --dataset final --load effb6final2cw --crop \
#                --test-csv dev.csv --test-dir Dev --crop | tee effnetb62c_valid.txt

python train.py --model-name tf_efficientnet_b6 --epochs 16 --lr 0.0001 --batch-size 16 --augment 5 --finetune 2 \
                --dataset final --weight-name effb6final25cw --crop \
                --test-csv dev.csv --test-dir Dev --crop | tee effb6final25c_valid.txt


