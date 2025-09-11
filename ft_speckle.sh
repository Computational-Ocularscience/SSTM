#!/bin/bash
mkdir -p checkpoints

python -u /mnt/Backup01/Fisseha/SSTM/train.py --name sstm_t++-speckle-sintel --stage speckle --validation speckle --restore_ckpt checkpoints/sstm_t++-sintel.pth  --gpus 0 1 --num_steps 50000 --batch_size 3 --lr 0.0001 --image_size 368 768 --wdecay 0.00001 --gamma=0.85




