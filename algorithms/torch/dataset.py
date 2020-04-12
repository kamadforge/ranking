#there can be an error
#RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #2 'target'

import numpy as np
import torch
import torch.utils.data as data

example=1

if example==1:
    my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
    my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)

    tensor_x = torch.stack([torch.Tensor(i) for i in my_x]) # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in my_y])

    my_dataset = data.TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = data.DataLoader(my_dataset) # create your dataloader



    x_train = np.random.multivariate_normal(np.zeros(10), np.eye(10), int(2000 / 2))
    y_train = np.random.choice(2, [1000,1])
    #x_val = np.random.multivariate_normal(np.zeros(10), np.eye(10), int(2000 / 2))


elif example==2:
    list=[]
    list_y=[]
    for i in range(len(x_train)):
        list.append(x_train[i])
        list_y.append(np.array(y_train[i]))

    x_s=torch.stack([torch.Tensor(i) for i in list])
    y_s=torch.stack([torch.Tensor(i) for i in list_y])

    train_dataset=data.TensorDataset(x_s, y_s)
    train_loader=data.DataLoader(train_dataset)

