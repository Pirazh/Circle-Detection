#!/usr/bin/bash

echo Running training for designed model ...
python main.py \
            --phase 'train' \
            --train_batch_size 128 \
            --train_set_size 100000 \
            --dev_batch_size 128 \
            --dev_set_size 5000 \
            --num_workers 16 \
            --epochs 20 \
            --lr 1e-4 \
            --mGPU \
            --loss_type 'Both' \
            --Lambda 1

        
