#!/usr/bin/bash

echo Running the designed model on the test set ...
python main.py \
            --phase test \
            --test_set_size 1000 \
            --iou_threshold 0.7 \
            --resumed_ckpt ./checkpoint/2019-11-06-20/best_checkpoint.pth.tar
