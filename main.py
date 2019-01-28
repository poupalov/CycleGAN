import os
import torch
import torch.optim as optim
import torch.utils.data as data
from tensorboard_logger import Logger
from utils import FolderDataset, train, ImagePool
from networks import Discriminator, Generator
import numpy as np

starting_epoch = 1
nb_epochs = 100

experiment_name= "train1"

# Folder when events are stored for Tensorboard
train_logger = Logger("events/" + experiment_name + "/train")

# Create experiment folders
if not os.path.isdir('models/' + experiment_name):
    os.makedirs('models/' + experiment_name)

# GPU

with torch.cuda.device(0):

    nb_filters = 32
    input_size = 128
    act = True
    batch_norm = True
    max_len = None
    max_steps = np.inf
    lr_0 = 0.0002
    lr_1 = lr_0 / 100
    
    dataset = FolderDataset(
            root_dir_A="datasets/maps/trainA",
            root_dir_B="datasets/maps/trainB",
            input_size=input_size,
            max_len=max_len
    )
    loader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    print("Dataset loaded !")
        
    D_B = Discriminator(nb_filters, 5, input_size).cuda()
    D_A = Discriminator(nb_filters, 5, input_size).cuda()
    G_AB = Generator(nb_filters, act, batch_norm).cuda()
    G_BA = Generator(nb_filters, act, batch_norm).cuda()
    
    if starting_epoch > 1:
        D_B_file = 'models/' + experiment_name + '/D_B_' + str(starting_epoch - 1) + '.pth'
        D_A_file = 'models/' + experiment_name + '/D_A_' + str(starting_epoch - 1) + '.pth'
        G_AB_file = 'models/' + experiment_name + '/G_AB_' + str(starting_epoch - 1) + '.pth'
        G_BA_file = 'models/' + experiment_name + '/G_BA_' + str(starting_epoch - 1) + '.pth'
        D_B.load_state_dict(torch.load(D_B_file))
        D_A.load_state_dict(torch.load(D_A_file))
        G_AB.load_state_dict(torch.load(G_AB_file))
        G_BA.load_state_dict(torch.load(G_BA_file))
        
    networks = [D_B, D_A, G_AB, G_BA]
    
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr_1)
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr_1)
    optimizer_G_AB = optim.Adam(G_AB.parameters(), lr=lr_1)
    optimizer_G_BA = optim.Adam(G_BA.parameters(), lr=lr_1)
    optimizers = [optimizer_G_AB, optimizer_G_BA, optimizer_D_B, optimizer_D_A]
    
    pool_A = ImagePool(pool_size=50)
    pool_B = ImagePool(pool_size=50)
    pools = [pool_A, pool_B]
    
    for epoch in range(starting_epoch, nb_epochs + 1):
        losses = train(epoch, loader, networks, optimizers, pools, max_steps=max_steps, verbose=False)
        train_logger.log_value('D_B loss', losses[0], epoch)
        train_logger.log_value('D_A loss', losses[1], epoch)
        train_logger.log_value('G_AB loss', losses[2], epoch)
        train_logger.log_value('G_BA loss', losses[3], epoch)
        total_loss = sum(losses)
        print("\nLoss at epoch n.{} : {}".format(epoch, total_loss))
        D_B_file = 'models/' + experiment_name + '/D_B_' + str(epoch) + '.pth'
        D_A_file = 'models/' + experiment_name + '/D_A_' + str(epoch) + '.pth'
        G_AB_file = 'models/' + experiment_name + '/G_AB_' + str(epoch) + '.pth'
        G_BA_file = 'models/' + experiment_name + '/G_BA_' + str(epoch) + '.pth'
        torch.save(D_B.state_dict(), D_B_file)
        torch.save(D_A.state_dict(), D_A_file)
        torch.save(G_AB.state_dict(), G_AB_file)
        torch.save(G_BA.state_dict(), G_BA_file)
