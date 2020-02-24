import torch
import numpy as np
import glob
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from scipy.misc import imread
import matplotlib.image as mpimg
import torchvision
import torchvision.transforms as transforms
from scipy.misc import imread
import torch.optim as optim
import os
from matplotlib.image import imread

load_file=True


if load_file==False:
    dataset_path="/home/kamil/Dropbox/Current_research/python_tests/results_adversarial/data/FashionMNIST_adversarial/"

    #my_x=

    my_y=np.load("/home/kamil/Dropbox/Current_research/python_tests/results_adversarial/data/FashionMNIST_adversarial/labels.npy")
    print(my_y)

    folder = "/home/kamil/Dropbox/Current_research/python_tests/results_adversarial/data/FashionMNIST_adversarial/images_np"

    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    print("Working with {0} images".format(len(onlyfiles)))
    print("Image examples: ")

    my_x=[]
    for file in onlyfiles:
        im=np.load(folder+"/"+file)
        my_x.append(im)
        #plt.imshow(im)
        #plt.show()
        #print(im.shape)
        print(file)
        #display(_Imgdis(filename=folder + "/" + onlyfiles[i], width=240, height=320))

    tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in my_y])

    torch.save(tensor_x, 'tensor_x.pt')
    torch.save(tensor_y, 'tensor_y.pt')
else:
    tensor_x=torch.load('tensor_x.pt')
    tensor_y=torch.load('tensor_y.pt')

    my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = torch.utils.data.DataLoader(my_dataset) # create your dataloader

    for i, (m,n) in enumerate(my_dataloader):
        print(i)
        print(m)