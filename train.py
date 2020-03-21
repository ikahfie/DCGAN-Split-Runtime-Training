import os
import random
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from libs import model, utils, visuals

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--new", action='store_true')
parser.add_argument("--image-dir", type=str, required=True)
parser.add_argument("--cp-file", type=str)
parser.add_argument("--workers", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--image_size", type=int, default=64)
parser.add_argument("--nz", type=int, default=100)
parser.add_argument("--nc", type=int, default=3)
parser.add_argument("--ndf", type=int, default=64)
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--num-epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--seed", type=int, default=999)
args = parser.parse_args()

device = torch.device("cuda:0" if (
    torch.cuda.is_available() and args.ngpu > 0) else "cpu")

saved_model = os.path.basename(args.image_dir)

def __new_model():
    netG = model.generator(args.nz, args.ngf, args.nc).to(device)
    netD = model.discriminator(args.nc, args.ndf).to(device)
    netG.apply(utils.weights_init)
    netD.apply(utils.weights_init)

    optimizerD = optim.Adam(netD.parameters(),
                            lr=args.lr,
                            betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(),
                            lr=args.lr,
                            betas=(args.beta1, 0.999))
    img_list = []
    G_losses = []
    D_losses = []
    curr_epoch = 0
    return netD, netG, optimizerD, optimizerG, img_list, G_losses, D_losses, curr_epoch


def __load_checkpoint(checkpoint):
    try:
        checkpoint = torch.load(checkpoint)
    except (RuntimeError, TypeError, NameError, ValueError):
        print("No checkpoint file found.")

    netG = model.generator(args.nz, args.ngf, args.nc).to(device)
    netD = model.discriminator(args.nc, args.ndf).to(device)
    optimizerD = optim.Adam(netD.parameters(),
                            lr=args.lr,
                            betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(),
                            lr=args.lr,
                            betas=(args.beta1, 0.999))

    curr_epoch = checkpoint['epoch']
    netD_state_dict = checkpoint['netD_state_dict']
    netG_state_dict = checkpoint['netG_state_dict']
    optimizerG_state_dict = checkpoint['optimizerG_state_dict']
    optimizerG_state_dict = checkpoint['optimizerD_state_dict']
    G_losses = checkpoint['G_losses']
    D_losses = checkpoint['D_losses']
    img_list = checkpoint['img_list']

    return netD, netG, optimizerD, optimizerG, img_list, G_losses, D_losses, curr_epoch


def __training_loop(dataloader, netD, netG, optimizerD, optimizerG, num_epochs,
                    curr_epoch, G_losses, D_losses, img_list):
    iters = 0
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    saved_model = os.path.basename(args.image_dir)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(curr_epoch, num_epochs):
        print("current_epoch = {0}, target_epochs = {1}, starting_epoch = {2}".
              format(epoch, num_epochs, curr_epoch))
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size, ), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader), errD.item(),
                       errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and
                                      (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(
                    vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        torch.save(
            {
                'epoch': epoch + 1,
                'netD_state_dict': netD.state_dict(),
                'netG_state_dict': netG.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses,
                'img_list': img_list
            }, ("data/checkpoint/" + saved_model + "_epoch_{}.pth".format(epoch + 1)))

        print(("model saved as {}".format(
            "data/checkpoint/" + saved_model + "_epoch_{}.pth".format(epoch + 1))))


def __train_init(seed, ngpu, image_dir, image_size, batch_size, workers,
                 num_epochs):
    manual_seed = args.seed
    '''uncomment this line below if you want to use random seed every time you trained it'''
    #manual_seed = random.randint(1,10000)
    #print("random seed: ", manual_seed)
    torch.manual_seed(manual_seed)

    dataloader = utils.gen_dataloader(args.image_dir, args.image_size,
                                      args.batch_size, args.workers)

    real_batch = next(iter(dataloader))
    __training_loop(dataloader,
                    netD=netD,
                    netG=netG,
                    optimizerD=optimizerD,
                    optimizerG=optimizerG,
                    num_epochs=num_epochs,
                    curr_epoch=curr_epoch,
                    G_losses=G_losses,
                    D_losses=D_losses,
                    img_list=img_list)


if __name__ == "__main__":
    if args.new and args.cp_file is None:
        netD, netG, optimizerD, optimizerG, img_list, G_losses, D_losses, curr_epoch = __new_model(
        )
        __train_init(args.seed, args.ngpu, args.image_dir, args.image_size,
                     args.batch_size, args.workers, args.num_epochs)
    elif args.cp_file and args.new == False:
        netD, netG, optimizerD, optimizerG, img_list, G_losses, D_losses, curr_epoch = __load_checkpoint(
            args.cp_file)
        __train_init(args.seed, args.ngpu, args.image_dir, args.image_size,
                     args.batch_size, args.workers, args.num_epochs)
    else:
        print(
            "ERROR: Make a new model (--new) or Continue from a checkpoint file (--cp-file CHECKPOINT_FILE) ?"
        ) 
        #print(args.new, args.cp_file)
 	
    """ print(("model saved as {}".format(
            "data/checkpoint/" + saved_model + "_epoch_{}.pth".format(7 + 1)))) """
