#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 16 --lr 0.0001 --batch-size 16 --augment 3 --finetune 2 \
#    --dataset final --load effb6final2cw --crop \
#    --test-csv dev.csv --test-dir Dev --crop | tee effnetb62c_valid.txt

#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 16 --lr 0.0001 --batch-size 16 --augment 3 --finetune 4 \
#    --dataset final --load effb6final4cw --crop \
#    --test-csv dev.csv --test-dir Dev --crop | tee effnetb64c_valid.txt

#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 16 --lr 0.0001 --batch-size 16 --augment 5 --finetune 2 \
#    --dataset final --load effb6final25cw --crop \
#    --test-csv dev.csv --test-dir Dev --crop | tee effb6final25c_valid.txt

# training
#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 16 --lr 0.0001 --batch-size 16 --augment 5 --finetune 4 \
#    --dataset final --load effb6final45cw --crop \
#    --test-csv dev.csv --test-dir Dev --crop | tee effb6final45c_valid.txt


#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 32 --lr 0.001 --batch-size 28 --augment 2 --finetune 4 \
#    --dataset final --load effb6finalf4a2cw --crop \
#    --test-csv dev.csv --test-dir Dev --crop | tee effbfinalf4a2c_valid.txt


#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 20 --lr 0.001 --batch-size 28 --augment 6 --finetune 4 \
#    --dataset c1p2 --load effb6blurf4cw --crop \
#    --test-csv dev.csv --test-dir DevBlur1x4 --crop --blur | tee effb6blurf4c_blur_1x4_valid.txt
#
#python train.py --model-name tf_efficientnet_b6 \
#    --epochs 20 --lr 0.001 --batch-size 28 --augment 6 --finetune 4 \
#    --dataset c1p2 --load effb6blurf4cw --crop \
#    --test-csv dev.csv --test-dir DevBlur3x3 --crop --blur | tee effb6blurf4c_blur_3x4_valid.txt

python train.py --model-name tf_efficientnet_b6 \
    --epochs 20 --lr 0.001 --batch-size 28 --augment 6 --finetune 4 \
    --dataset c1p2 --load effb6blurf4cw --crop \
    --test-csv dev.csv --test-dir DevMotionBlur --crop --blur | tee effb6blurf4c_mblur_valid.txt
