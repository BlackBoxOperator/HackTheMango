#python train.py --batch-size 48 --finetune 4 --augment 3 --lr 0.0001 --epochs 32 \
#    --dataset final --load polyfinal9w \
#    --test-csv new_valid1.csv --test-dir Train --crop
#
#python train.py --batch-size 48 --finetune 4 --augment 3 --lr 0.0001 --epochs 32 \
#    --dataset final --load polyfinal9w \
#    --test-csv dev.csv --test-dir Dev --crop


#python train.py --batch-size 48 --finetune 4 --augment 3 --lr 0.0001 --epochs 32 --dataset final --weight-name polyfinal9w --crop | tee polyfinal9.txt


#python train.py --batch-size 10 --finetune 4 --augment 6 --lr 0.0001 --epochs 32 \
#    --dataset c1p2 --load polyblur12w \
#    --test-csv dev.csv --test-dir DevBlur1x4 --crop --blur | tee polyblur12_blur_1x4_valid.txt
#
#python train.py --batch-size 10 --finetune 4 --augment 6 --lr 0.0001 --epochs 32 \
#    --dataset c1p2 --load polyblur12w \
#    --test-csv dev.csv --test-dir DevBlur3x3 --crop --blur | tee polyblur12_blur_3x3_valid.txt

python train.py --batch-size 10 --finetune 4 --augment 6 --lr 0.0001 --epochs 32 \
    --dataset c1p2 --load polyblur12w \
    --test-csv dev.csv --test-dir DevMotionBlur --crop --blur | tee polyblur12_mblur_valid.txt

