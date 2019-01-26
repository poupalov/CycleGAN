import os
import torch
from skimage import io, transform
from networks import Generator
import matplotlib.pyplot as plt
import numpy as np

#epochs = [40]
epochs = list(range(1, 84))
experiment_name= "train1"

images_A_path = "datasets/maps/trainA"
images_B_path = "datasets/maps/trainB"

for epoch in epochs:
    
    G_AB_file = 'models/' + experiment_name + '/G_AB_' + str(epoch) + '.pth'
    #G_BA_file = 'models/' + experiment_name + '/G_BA_' + str(epoch) + '.pth'
    
    G_AB = Generator(64)
    #G_BA = Generator(64)
    
    G_AB.load_state_dict(torch.load(G_AB_file))
    #G_BA.load_state_dict(torch.load(G_BA_file))
    
    images_A_str = [os.path.join(images_A_path, x) for x in os.listdir(images_A_path)]
    for s in images_A_str[:1]:
        image = io.imread(images_A_str[0])
        
        tensor = transform.resize(image, (256, 256))
        tensor = torch.from_numpy(tensor)
        tensor = tensor.view(1, 3, 256, 256).float()
        if epoch == epochs[0]:
            plt.imshow(tensor.view(256, 256, 3))
            plt.savefig('figures/A.png')
        #plt.show(block=True)
        
        gen_img = G_AB(tensor).detach().numpy()
        plt.imshow(np.reshape(gen_img, [256, 256, 3]))
        plt.savefig('figures/B_{}.png'.format(epoch))
        #plt.show(block=True)
