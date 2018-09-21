
## Pose estimation code for our WACV 2019 paper.

Source code for pose estimation part of our WACV 2019 paper "Aligned to the Object, not to the Image: A Unified Pose-aligned Representation for Fine-grained Recognition"

Python 3.6. and Pytorch 0.4.1. are required to run the code.

To start training, use:

```
CUDA_VISIBLE_DEVICES=5 python main.py /path-to-the-cub-dataset/ --pretrain --epochs 100 --lr 0.2 --print-freq 100 -b 16 --lr_decay 80 #--visualize 
```
or simply run `run.sh`. Uncomment visualize to watch the process of training.

We get PCK@10%: 92.65% on CUB-200-2011 using this simple network architecture:

![FCN](https://i.imgur.com/FmkDkfS.png)
