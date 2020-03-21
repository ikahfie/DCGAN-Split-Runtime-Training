#!/bin/bash

if [ "$#" != "4" ];
 then
	echo pass the [train_cp.sh IMAGE_DIR CP_FILE TARGET_NUM_EPOCHS N_GPU]
	exit 1
fi

string="$2"

basename $string .pth
base="${base%.pth}"
currepoch=$(grep -oE "[[:digit:]]{1,}" <<< "$string")

for (( i="$currepoch"; i < "$3"; i++))
do
	~/miniconda3/envs/torch/bin/python train.py --image-dir $1 --cp-file $string --num-epochs $((i+1)) --ngpu $4
	sleep 5m
done
