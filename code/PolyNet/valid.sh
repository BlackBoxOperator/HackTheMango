python train.py --batch-size 48 --finetune 4 --augment 3 --lr 0.0001 --epochs 32 \
    --dataset final --load polyfinal9w \
    --test-csv new_valid1.csv --test-dir Train --crop

python train.py --batch-size 48 --finetune 4 --augment 3 --lr 0.0001 --epochs 32 \
    --dataset final --load polyfinal9w \
    --test-csv dev.csv --test-dir Dev --crop


#python train.py --batch-size 48 --finetune 4 --augment 3 --lr 0.0001 --epochs 32 --dataset final --weight-name polyfinal9w --crop | tee polyfinal9.txt
