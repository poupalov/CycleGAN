import os
import torch.utils.data as data
from skimage import io, transform
import torch
import numpy as np

class FolderDataset(data.Dataset):

    def __init__(self, root_dir_A, root_dir_B, max_len=None):
        """
        Args:
            root_dir (string): Directory with all the images.
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
        
        for i in range(2):
            root_dir = self.root_dirs[i]
            list_imgs = self.lists_imgs[i]
            m = self.max_lens[i]
            images = []
            labels = []
            for idx in range(m):
                img_name = os.path.join(root_dir, list_imgs[idx])
                image = io.imread(img_name)
                image = transform.resize(image, (256, 256))
                image = torch.from_numpy(image)
                image = image.view(3, 256, 256).float()
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

def train(epoch, loader, networks, optimizers):
    
    D_B, D_A, G_AB, G_BA = networks
    optimizer_G_AB, optimizer_G_BA, optimizer_D_B, optimizer_D_A = optimizers
    
    D_B_training_loss, D_A_training_loss, G_AB_training_loss, G_BA_training_loss = 0, 0, 0, 0
    
    for (images, labels) in loader:
        
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
        D_B_loss_1 = mse_loss(D_B(true_img_B), 1)
        D_B_loss_2 = mse_loss(D_B(fake_img_B), 0)
        D_B_loss = (D_B_loss_1 + D_B_loss_2) / 2
        D_B_training_loss += D_B_loss.data.item()
        D_B_loss.backward(retain_graph=True)
        optimizer_D_B.step()

        # Optimize discriminator D_A
        optimizer_D_A.zero_grad()
        D_A_loss_1 = mse_loss(D_A(true_img_A), 1)
        D_A_loss_2 = mse_loss(D_A(fake_img_A), 0)
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
        G_BA_loss_1 = mse_loss(D_A(fake_img_A), 1)
        cycle_loss = abs_loss(true_img_A, cycle_img_A) + abs_loss(true_img_B, cycle_img_B)
        G_BA_loss = G_BA_loss_1 + 10 * cycle_loss
        G_BA_training_loss += G_AB_loss.data.item()
        G_BA_loss.backward()
        optimizer_G_BA.step()
        
    losses = np.array([D_B_training_loss, D_A_training_loss, G_AB_training_loss, G_BA_training_loss])
    return losses