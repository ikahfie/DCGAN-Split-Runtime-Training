import torch.nn as nn
from torch import no_grad
from torchvision import transforms
from torchvision import datasets as dset
from torch.utils.data import DataLoader


def weights_init(m):
    with no_grad():
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def gen_dataloader(image_dir, image_size, batch_size, workers):
    dataset = dset.ImageFolder(root=image_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=workers)
    return dataloader
	
