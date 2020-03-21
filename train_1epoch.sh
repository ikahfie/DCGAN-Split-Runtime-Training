#!/bin/bash


if [ $((#)) != "3" ];
 then
	echo pass the [train_new.sh IMAGE_DIR N_GPU]
	exit 1
fi

~/miniconda3/envs/torch/bin/python train.py --new --image-dir $1 --num-epochs 1 --ngpu $2 
