import os
import torch.utils.data as data
from skimage import io, transform
import torch
import numpy as np
import random


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
    

class FolderDataset(data.Dataset):

    def __init__(self, root_dir_A, root_dir_B, input_size=256, max_len=None):
        """
        Args:
            root_dir_A (string): Directory with all the aerial photographs.
            root_dir_B (string): Directory with all the maps.
            max_len (int): Optionnal maximum length for the dataset.
        """
        self.root_dir_A = root_dir_A
        self.root_dir_B = root_dir_B
        self.list_imgs_A = os.listdir(root_dir_A)
        self.list_imgs_A.sort()
        self.list_imgs_B = os.listdir(root_dir_B)
        self.list_imgs_B.sort()
        self.root_dirs = [self.root_dir_A, self.root_dir_B]
        self.lists_imgs = [self.list_imgs_A, self.list_imgs_B]
        self.max_lens = [min(max_len, len(l)) if max_len is not None else len(l) for l in self.lists_imgs]
        self.images = []
        self.labels = []
        
        # Sanity check : make sure images are paired in the training set
        checkA = [s.split("_")[0] for s in self.list_imgs_A]
        checkB = [s.split("_")[0] for s in self.list_imgs_B]
        assert (checkA == checkB), "Files are not all paired. Please remove unpaired files from training folders"
        
        # Loads all images in memory
        for i in range(2):
            root_dir = self.root_dirs[i]
            list_imgs = self.lists_imgs[i]
            m = self.max_lens[i]
            images = []
            labels = []
            for idx in range(m):
                img_name = os.path.join(root_dir, list_imgs[idx])
                image = io.imread(img_name)
                image = transform.resize(image, (input_size, input_size))
                image = torch.from_numpy(image)
                image = image.view(3, input_size, input_size).float()
                label = list_imgs[idx].split(".")[0].split("_")[1]
                images.append(image)
                labels.append(label)
            self.images.append(images)
            self.labels.append(labels)

    def __len__(self):
        return self.max_lens[0] * self.max_lens[1]

    def __getitem__(self, idx):
        m1 = self.max_lens[1]
        i = int(idx / m1)
        j = idx - i * m1
        images = [self.images[0][i], self.images[1][j]]
        labels = [self.labels[0][i], self.labels[1][j]]
        return images, labels
    

def mse_loss(inp, target):
    return torch.mean((inp - target) ** 2)

def abs_loss(inp, target):
    return torch.mean(torch.abs(inp - target))

def train(epoch, loader, networks, optimizers, pools, max_steps=np.inf, verbose=False):
    
    D_B, D_A, G_AB, G_BA = networks
    optimizer_G_AB, optimizer_G_BA, optimizer_D_B, optimizer_D_A = optimizers
    pool_A, pool_B = pools
    
    D_B_training_loss, D_A_training_loss, G_AB_training_loss, G_BA_training_loss = 0, 0, 0, 0
    
    step = 1
    if verbose : print("--- Training epoch {} ---".format(epoch))
    
    for (images, labels) in loader:
        
        if step > max_steps : break
        if verbose : print("Step number : ", step)
        
        true_img_A = images[0]
        true_img_B = images[1]
        true_img_A = true_img_A.cuda()
        true_img_B = true_img_B.cuda()
        fake_img_B = G_AB(true_img_A)
        fake_img_A = G_BA(true_img_B)
        cycle_img_A = G_BA(fake_img_B)
        cycle_img_B = G_AB(fake_img_A)

        # Optimize discriminator D_B
        optimizer_D_B.zero_grad()
        fake_img = pool_B.query([fake_img_B])[0]
        D_B_loss_1 = mse_loss(D_B(true_img_B), 1)
        D_B_loss_2 = mse_loss(D_B(fake_img), 0)
        D_B_loss = (D_B_loss_1 + D_B_loss_2) / 2
        D_B_training_loss += D_B_loss.data.item()
        D_B_loss.backward(retain_graph=True)
        optimizer_D_B.step()

        # Optimize discriminator D_A
        optimizer_D_A.zero_grad()
        fake_img = pool_A.query([fake_img_A])[0]
        D_A_loss_1 = mse_loss(D_A(true_img_A), 1)
        D_A_loss_2 = mse_loss(D_A(fake_img), 0)
        D_A_loss = (D_A_loss_1 + D_A_loss_2) / 2
        D_A_training_loss += D_A_loss.data.item()
        D_A_loss.backward(retain_graph=True)
        optimizer_D_A.step()

        # Optimize generator G_AB
        optimizer_G_AB.zero_grad()
        G_AB_loss_1 = mse_loss(D_B(fake_img_B), 1)
        cycle_loss = abs_loss(true_img_A, cycle_img_A) + abs_loss(true_img_B, cycle_img_B)
        G_AB_loss = G_AB_loss_1 + 10 * cycle_loss
        G_AB_training_loss += G_AB_loss.data.item()
        G_AB_loss.backward(retain_graph=True)
        optimizer_G_AB.step()

        # Optimize generator G_BA
        optimizer_G_BA.zero_grad()
        fake_img_B = G_AB(true_img_A)
        fake_img_A = G_BA(true_img_B)
        cycle_img_A = G_BA(fake_img_B)
        cycle_img_B = G_AB(fake_img_A)
        G_BA_loss_1 = mse_loss(D_A(fake_img_A), 1)
        cycle_loss = abs_loss(true_img_A, cycle_img_A) + abs_loss(true_img_B, cycle_img_B)
        G_BA_loss = G_BA_loss_1 + 10 * cycle_loss
        G_BA_training_loss += G_AB_loss.data.item()
        G_BA_loss.backward()
        optimizer_G_BA.step()
        
        step += 1
        
    losses = np.array([D_B_training_loss, D_A_training_loss, G_AB_training_loss, G_BA_training_loss])
    return losses