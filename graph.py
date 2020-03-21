from libs import visuals, model, utils
import torch
import argparse
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--new", action='store_true')
parser.add_argument("--image-dir", type=str, required=True)
parser.add_argument("--cp-file", type=str, required=True)
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

try:
        checkpoint = torch.load(args.cp_file)
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

visuals.loss_graph(G_losses, D_losses)
manual_seed = args.seed
    #print("random seed: ", manual_seed)
#random.seed(manual_seed)
torch.manual_seed(manual_seed)

dataloader = utils.gen_dataloader(args.image_dir, args.image_size,
									args.batch_size, args.workers)
									
visuals.compare_imgs(dataloader, img_list, device)