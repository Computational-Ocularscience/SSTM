#!/bin/bash
mkdir -p checkpoints


python -u /mnt/Backup01/Fisseha/SSTM/train.py --name sstm++-virtualKitti --stage virtualKitti --validation vkitti --gpus 0 1 --num_steps 60000 --batch_size 8 --lr 0.0004 --image_size 288 960 --wdecay 0.0001

#python -u /mnt/Backup01/Fisseha/SSTM/train.py --name sstm_t3k++-things --stage things --validation sintel --restore_ckpt checkpoints/140000_sstm_t3k++-things.pth --gpus 0 1 --num_steps 160000 --batch_size 6 --lr 0.000225 --image_size 400 720 --wdecay 0.0001

#python -u /mnt/Backup01/Fisseha/SSTM/train.py --name sstm_t3k++-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/sstm_t3k++-things.pth --gpus 0 1 --num_steps 120000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85

#python -u /mnt/Backup01/Fisseha/SSTM/train.py --name sstm_t3k++-kitti --stage kitti --validation kitti --restore_ckpt checkpoints/sstm_t3k++-sintel.pth --gpus 0 1 --num_steps 70000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85




