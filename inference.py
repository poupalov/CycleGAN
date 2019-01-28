import os
import torch
from skimage import io, transform
from networks import Generator
import matplotlib.pyplot as plt
import numpy as np

epochs = [100]
experiment_name= "train1"

images_A_path = "datasets/maps/trainA"
images_B_path = "datasets/maps/trainB"

nb_filters = 32
input_size = 128
act = True
batch_norm = True

for epoch in epochs:
    
    G_AB_file = 'models/' + experiment_name + '/G_AB_' + str(epoch) + '.pth'
    G_BA_file = 'models/' + experiment_name + '/G_BA_' + str(epoch) + '.pth'
    
    G_AB = Generator(nb_filters, act, batch_norm)
    G_BA = Generator(nb_filters, act, batch_norm)
    
    G_AB.load_state_dict(torch.load(G_AB_file))
    G_BA.load_state_dict(torch.load(G_BA_file))
    
    images_A_str = [os.path.join(images_A_path, x) for x in os.listdir(images_A_path)]
    for s in images_A_str:
        image = io.imread(images_A_str[0])
        
        tensor = transform.resize(image, (input_size, input_size))
        tensor = torch.from_numpy(tensor)
        tensor = tensor.view(1, 3, input_size, input_size).float()
        if epoch == epochs[0]:
            plt.imshow(tensor.view(input_size, input_size, 3))
            plt.savefig('figures/A.png')
        
        gen_img = G_AB(tensor).detach().numpy()
        plt.imshow(np.reshape(gen_img, [input_size, input_size, 3]))
        plt.savefig('figures/B_{}.png'.format(epoch))
       
        
    images_B_str = [os.path.join(images_B_path, x) for x in os.listdir(images_B_path)]
    for s in images_B_str:
        image = io.imread(images_B_str[0])
        
        tensor = transform.resize(image, (input_size, input_size))
        tensor = torch.from_numpy(tensor)
        tensor = tensor.view(1, 3, input_size, input_size).float()
        if epoch == epochs[0]:
            plt.imshow(tensor.view(input_size, input_size, 3))
            plt.savefig('figures/B.png')
        
        gen_img = G_BA(tensor).detach().numpy()
        plt.imshow(np.reshape(gen_img, [input_size, input_size, 3]))
        plt.savefig('figures/A_{}.png'.format(epoch))
