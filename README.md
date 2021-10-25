# Poor-Spec-DCGAN
## Description
A dcgan implementation to be trained with low VRAM GPU by exploiting training checkpoints per epoch and shell script.
Based on the [Pytorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).

## Dependacies
* torchvision
* torch
* matplotlib (used in [graph.py](graph.py))
* numpy
## How To Use 
* To make a new model, run this in the project directory on your terminal:  
`python train.py --image-dir IMAGE_DIR --num_epoch NUM_EPOCH --ngpu NUMBER_OF_GPU --new --seed SEED_NUM`

* To continue from a .pth checkpoint file, run this in the project directory on your terminal:   
`python train.py  --image-dir IMAGE_DIR --num_epoch NUM_EPOCH --ngpu NUMBER_OF_GPU --cp-file PTH_FILE --seed SEED_NUM`

Both training seed default value is 999. If you want to use different number make sure that you used the same seed number you used to generate the model when continuing the training from checkpoint.

* To generate loss graph and comparison images, run this:
`python graph.py --image-dir IMAGE_DIR --cp-file PTH_FILE`

The train_cp.sh script is used to train the model by doing it one epoch at the time, save a checkpoint, exit the training script instance, start a new one, load the checkpoint, continue the training and so on. I only used this because my RX-460 gpu only have 2 GB VRAM (lol), so it could run the inference with only one epoch at the time before crashing my GPU. 

*TL:DR: Just try to make a new model first with target epoch using the [train.py](train.py) first. If it crash then read the shell script thoroughly in a text editor, and then changed it however you see fit, modify the [train_cp.sh](train_cp.sh) script. After that run [train_1epoch.sh](train_1epoch.sh) to get one model checkpoint and continue using [train_cp.sh](train_cp.sh)*

## License
See the [LICENSE.md](LICENSE.md) file for license rights and limitations (MIT).
