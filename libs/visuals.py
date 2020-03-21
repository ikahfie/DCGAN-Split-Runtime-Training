import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

def loss_graph(G_losses, D_losses):
	plt.figure(figsize=(10, 5))
	plt.title("Generator and Discriminator Loss During Training")
	plt.plot(G_losses, label="G")
	plt.plot(D_losses, label="D")
	plt.xlabel("iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()
	
def compare_imgs(dataloader, img_list, device):
	# Grab a batch of real images from the dataloader
	real_batch = next(iter(dataloader))

	# Plot the real images
	plt.figure(figsize=(15, 15))
	plt.subplot(1, 2, 1)
	plt.axis("off")
	plt.title("Real Images")
	plt.imshow(
		np.transpose(
			vutils.make_grid(real_batch[0].to(device)[:64],
							padding=5,
							normalize=True).cpu(), (1, 2, 0)))

	# Plot the fake images from the last epoch
	plt.subplot(1, 2, 2)
	plt.axis("off")
	plt.title("Fake Images")
	plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
	plt.show()