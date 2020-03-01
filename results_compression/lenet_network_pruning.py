# for four layer lenet:
# -loads the models
# -evaluates the model
#- prunes the model
# -retrains the model
# - computes the combinations with the pruned model


import torch
from torch import nn, optim
import sys



import torch
import torch.optim as optim
from torch import nn, optim
import torch.nn.functional as f

import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import csv
import pdb
import os
import socket

from sklearn.model_selection import ParameterGrid

import argparse


#############
# params

arguments=argparse.ArgumentParser()

arguments.add_argument("--arch", default="5,8,30,10")
arguments.add_argument("--folder")

args=arguments.parse_args()

#######
# path stuff
cwd = os.getcwd()
if 'g0' in socket.gethostname() or 'p0' in socket.gethostname():
    #the cwd is where the sub file is so ranking/
    sys.path.append(os.path.join(cwd, "results_switch"))
    path_compression = os.path.join(cwd, "results_compression")
    path_main= cwd
else:
    #the cwd is results_compression
    parent_path = os.path.abspath('..')
    sys.path.append(os.path.join(parent_path, "results_switch"))
    path_compression = cwd
    path_main= parent_path

print(cwd)
print(sys.path)


######################


from torch.nn.parameter import Parameter
import magnitude_rank
from lenet_network_pruning_withcombinations import compute_combinations_lenet
#from lenet_network_pruning_withcombinations import get_data


from lenet5_conv_gpu_switch_working_Feb20_integral import run_experiment as run_experiment_integral
from lenet5_conv_gpu_switch_working_Feb20_pointest import run_experiment as run_experiment_pointest


# takes the network parameters from the conv layer and clusters them (with the purpose of removing some of them)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

trainval_perc=0.8
BATCH_SIZE = 105

dataset="mnist"
evaluation="test"
adversarial_dataset=False

###################################################
# DATA


if dataset=="fashionmnist":

    trainval_dataset=datasets.FashionMNIST(path_compression+'data/FashionMNIST', train=True, download=True,
                        #transform=transforms.Compose([transforms.ToTensor(),
                        #transforms.Normalize((0.1307,), (0.3081,))]),
                        transform=transforms.ToTensor())

    train_size = int(trainval_perc * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    test_dataset=datasets.FashionMNIST('data/FashionMNIST', train=False, transform=transforms.ToTensor())

    if adversarial_dataset:
        tensor_x = torch.load('../results_adversarial/data/FashionMNIST_adversarial/tensor_x.pt')
        tensor_y = torch.load('../results_adversarial/data/FashionMNIST_adversarial/tensor_y.pt')
        tensor_x = tensor_x.unsqueeze(1)
        tensor_y=tensor_y.squeeze(1)

        test_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)  # create your datset

    print("Loading,", dataset)

elif dataset=="mnist":

    trainval_dataset = datasets.MNIST('data/MNIST', train=True, download=True,
                                             # transform=transforms.Compose([transforms.ToTensor(),
                                             # transforms.Normalize((0.1307,), (0.3081,))]),
                                             transform=transforms.ToTensor())

    train_size = int(trainval_perc * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    torch.manual_seed(0)
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    test_dataset = datasets.MNIST('data/MNIST', train=False, transform=transforms.ToTensor())

    print("Loading,", dataset)


# Load datasets

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    #datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    test_dataset,
    batch_size=BATCH_SIZE, shuffle=False)




# #change: three times datasetname, path to load, combination arrays to rpune filter ranking
#
# dataset="mnist"
#
#
# BATCH_SIZE = 100
# # Download or load downloaded MNIST dataset
# # shuffle data at every epoch
# trainval_dataset=datasets.MNIST('data', train=True, download=True,
#                     #transform=transforms.Compose([transforms.ToTensor(),
#                     #transforms.Normalize((0.1307,), (0.3081,))]),
#                     transform=transforms.ToTensor())
#
# train_size = int(0.8 * len(trainval_dataset))
# val_size = len(trainval_dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# # Same for test data
# test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.ToTensor()), batch_size=BATCH_SIZE, shuffle=False)

##############################################################################################3####33
# ###############################################################################3##########
# NETWORK (conv-conv-fc-fc)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Lenet(nn.Module):
    def __init__(self, nodesNum1, nodesNum2, nodesFc1, nodesFc2):
        super(Lenet, self).__init__()

        self.nodesNum2=nodesNum2

        self.c1 = nn.Conv2d(1, nodesNum1, 5)
        self.s2 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(nodesNum1)
        self.c3 = nn.Conv2d(nodesNum1, nodesNum2, 5)
        self.s4 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(nodesNum2)
        self.c5 = nn.Linear(nodesNum2 * 4 * 4, nodesFc1)
        self.f6 = nn.Linear(nodesFc1, nodesFc2)
        self.out7 = nn.Linear(nodesFc2, 10)

        self.parameter = Parameter(-1e-10*torch.ones(nodesNum1),requires_grad=True) # this parameter lies #S

        # Fisher method is called on backward passes
        self.running_fisher = [0] * 4

        self.act1=0
        self.activation1 = Identity()
        self.activation1.register_backward_hook(self._fisher1)

        self.act2=0
        self.activation2 = Identity()
        self.activation2.register_backward_hook(self._fisher2)

        self.act3=0
        self.activation3 = Identity()
        self.activation3.register_backward_hook(self._fisher3)

        self.act4=0
        self.activation4 = Identity()
        self.activation4.register_backward_hook(self._fisher4)





    def forward(self, x):

        # output=f.relu(self.fc1(x))
        # output=self.bn1(output)
        # output=f.relu(self.fc2(output))
        # output=self.bn2(output)
        # output=self.fc3(output)
        # return output

        # #x=x.view(-1,784)
        output = self.c1(x)
        output = f.relu(self.s2(output))
        out = self.activation1(output)
        self.act1 = out
        output = self.bn1(output)

        output = self.c3(output)
        output = f.relu(self.s4(output))
        out = self.activation2(output)
        self.act2 = out
        output = self.bn2(output)

        output = output.view(-1, self.nodesNum2 * 4 * 4)
        output = self.c5(output)
        out = self.activation3(output)
        self.act3 = out


        output = self.f6(output)
        out = self.activation4(output)
        self.act4 = out
        output = self.out7(output) #remove for 99.27 and 90.04 models

        return output

    def _fisher1(self, notused1, notused2, grad_output):
        act1 = self.act1.detach()
        grad = grad_output[0].detach()
        #print(grad_output[0].shape)

        g_nk = (act1 * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        #print(del_k.shape)
        self.running_fisher[0] += del_k

    def _fisher2(self, notused1, notused2, grad_output):
        act2 = self.act2.detach()
        grad = grad_output[0].detach()
        #print(grad_output[0].shape)

        g_nk = (act2 * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        #print(del_k.shape)
        self.running_fisher[1] += del_k

    def _fisher3(self, notused1, notused2, grad_output):
        act3 = self.act3.detach()
        grad = grad_output[0].detach()
        #print(grad_output[0].shape)

        g_nk = (act3 * grad)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        #print(del_k.shape)
        self.running_fisher[2] += del_k

    def _fisher4(self, notused1, notused2, grad_output):
        act4 = self.act4.detach()
        grad = grad_output[0].detach()
        #print(grad_output[0].shape)

        g_nk = (act4 * grad)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        #print(del_k.shape)
        self.running_fisher[3] += del_k

    def reset_fisher(self):
        self.running_fisher[3] = 0 * self.running_fisher

    def cost(self):

        in_channels = self.in_channels
        out_channels = self.out_channels
        middle_channels = int(self.mask.sum().item())

        conv1_size = self.conv1.weight.size()
        conv2_size = self.conv2.weight.size()

        # convs
        self.params = in_channels * middle_channels * conv1_size[2] * conv1_size[
            3] + middle_channels * out_channels * \
                      conv2_size[2] * conv2_size[3]

        # batchnorms, assuming running stats are absorbed
        self.params += 2 * in_channels + 2 * middle_channels

        # skip
        if not self.equalInOut:
            self.params += in_channels * out_channels
        else:
            self.params += 0



##################################


nodesNum1, nodesNum2, nodesFc1, nodesFc2=10,20,100,25
net=Lenet(nodesNum1,nodesNum2,nodesFc1,nodesFc2).to(device)
criterion = nn.CrossEntropyLoss()

#optimizer=optim.Adam(net.parameters(), lr=0.001)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

#path="models/fashionmnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo62_acc90.04"


######################################
#LOADING MODEL/RESUME

def load_model():
    global path

    #path="models/fashionmnist_conv:20_conv:50_fc:800_fc:500_rel_bn_trainval1.0_epo:11_acc:90.01"
    #path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
    if dataset=="mnist":
        #path=path_compression+"/models/mnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo540_acc99.27"
        path=path_compression+"/models/MNIST_conv_10_conv_20_fc_100_fc_25_rel_bn_drop_trainval_modelopt1.0_epo_231_acc_99.19"

        #path="models/mnist_trainval0.9_epo461_acc99.06"
    elif dataset=="fashionmnist":
        path=path_compression+"/models/fashionmnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo62_acc90.04"
    #path="models/conv:10_conv:50_fc:800_fc:500_rel_bn_epo:103_acc:99.37""
    #path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:11_switch_acc:99.15"
    #path="/home/kamil/Dropbox/Current_research/python_tests/Dir_switch/models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:2_acc:98.75"

    net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'], strict=False)
    #net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage), strict=False)


    print(dataset, "loaded.")



########################################################
# EVALUATE

def evaluate():
    # print('Prediction when network is forced to predict')
    net.eval()
    correct = 0
    total = 0
    if evaluation=="test":
        eval_loader=test_loader
    elif evaluation=="val":
        eval_loader=val_loader

    for j, data in enumerate(eval_loader):
        images, labels = data
        images = images.to(device)
        predicted_prob = net.forward(images)  # images.view(-1,28*28)
        #print(predicted_prob)
        predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
        #print(predicted)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
    # print(str(correct) +" "+ str(total))
    # pdb.set_trace()
    accuracy = 100 * float(correct) / total
    print("test accuracy: %.2f %%" % (accuracy))
    return accuracy


############################3
#

def train(thresh=[-1,-1,-1,-1]):
    # here retraining works
    net.train()
    stop = 0;
    epoch = 0;
    best_accuracy = 0;
    entry = np.zeros(3);
    best_model = -1;
    early_stopping = 500
    while (stop < early_stopping):
    #for i in range(5):
        epoch = epoch + 1
        # for epoch in range(30):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # net.c1.weight.data[1] = 0  # instead of hook
            # net.c1.bias.data[1] = 0  # instead of hook
            # if i % 100==0:
            #    print (i)
            #   print (loss.item())

        print(loss.item())
        accuracy = evaluate()
        print("Epoch " + str(epoch) + " ended.")
        print((net.c1.weight.grad == 0).sum())  # the hook is automatically applied, here we just check the gradient

        if (accuracy <= best_accuracy):
            stop = stop + 1
            entry[2] = 0
        else:
            best_accuracy = accuracy
            print("Best updated")
            stop = 0
            entry[2] = 1
            best_model = net.state_dict()
            best_optim = optimizer.state_dict()
            if save:
                if retrain:
                    if best_accuracy > save_accuracy:
                        torch.save(best_model, "%s_retrained_epo:%d_prunedto:%d_%d_%d_%d_acc:%.2f" % (
                        path, epoch, thresh[0], thresh[1], thresh[2], thresh[3], best_accuracy))
                        #
                        # torch.save({'model_state_dict': best_model, 'optimizer_state_dict': best_optim},
                        #            "models/%s_conv:%d_conv:%d_fc:%d_fc:%d_rel_bn_drop_trainval_modelopt%.1f_epo:%d_acc:%.2f" % (
                        #            dataset, conv1, conv2, fc1, fc2, trainval_perc, epoch, best_accuracy))

                else:
                    if best_accuracy > save_accuracy:
                        torch.save({'model_state_dict': best_model, 'optimizer_state_dict': best_optim}, "%s_trainval%.1f_epo%d_acc%.2f" % (
                            dataset, trainval_perc,epoch, best_accuracy))

            entry[0] = accuracy;
            entry[1] = loss
            if write_training:
                with open(filename, "a+") as file:
                    file.write("\n Epoch: %d\n" % epoch)
                    file.write(",".join(map(str, entry)) + "\n")
                    if (accuracy > 98.9):
                        file.write("Yes\n")
                    elif (accuracy > 98.8):
                        file.write("Ok\n")

    print(loss.item())
    print("Final: " + str(best_accuracy))
    accuracy = evaluate()



def finetune():

    # switch to train mode
    net.train()


    dataiter = iter(train_loader)

    for i in range(0, 100):

        try:
            input, target = dataiter.next()
        except StopIteration:
            dataiter = iter(train_loader)
            input, target = dataiter.next()


        input, target = input.to(device), target.to(device)

        # compute output
        output = net(input)

        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()





####################################################3
# get ranks

def get_ranks(method):
    #### GET RANKS

    if method == 'filter_ranking':
        if dataset == "mnist":

            # from best to worst
            combinationss = [
                [1, 8, 7, 4, 6, 3, 9, 2, 0, 5],
                [2, 8, 9, 19, 4, 12, 5, 14, 11, 6, 3, 7, 18, 1, 17, 15, 16, 13, 0, 10],
                [56, 86, 25, 64, 33, 17, 23, 96, 46, 52, 43, 22, 81, 15, 1, 44, 39, 85, 19, 8, 58, 63, 29, 70, 14, 95,
                 27, 73, 72, 45, 68, 4, 13, 99, 75, 47, 34, 89, 97, 88, 61, 53,
                 21, 50, 93, 57, 94, 51, 82, 60, 98, 40, 76, 62, 30, 9, 84,
                 37, 0, 42, 80, 71, 92, 79, 32, 38, 78, 11, 90, 5, 2, 87, 66, 65, 16, 55, 48, 36, 18, 67, 35, 91, 83,
                 10, 24, 31, 7, 28, 20, 74, 54, 6, 12, 3, 49, 69, 59, 77, 26, 41],
                [1, 7, 2, 3, 0, 4, 6, 9, 5, 8, 10, 24, 13, 19, 12, 21, 23, 16, 22, 18, 17, 15, 14, 11, 20]
            ]
            # 99.27-1trainval 0.2val
            combinationss = [
                [1, 8, 5, 7, 9, 3, 6, 4, 2, 0],
                [2, 8, 9, 19, 12, 4, 16, 17, 14, 5, 11, 18, 13, 3, 15, 10, 6, 7, 1, 0],
                [56, 86, 25, 64, 33, 17, 23, 96, 46, 52, 43, 22, 81, 15, 1, 44, 39, 85, 19, 8, 58, 63, 29, 70, 14, 95,
                 27, 73, 72, 45, 68, 4, 13, 99, 75, 47, 34, 89, 97, 88, 61, 53,
                 21, 50, 93, 57, 94, 51, 82, 60, 98, 40, 76, 62, 30, 9, 84,
                 37, 0, 42, 80, 71, 92, 79, 32, 38, 78, 11, 90, 5, 2, 87, 66, 65, 16, 55, 48, 36, 18, 67, 35, 91, 83,
                 10, 24, 31, 7, 28, 20, 74, 54, 6, 12, 3, 49, 69, 59, 77, 26, 41], #copied from above
                [1, 7, 4, 3, 6, 2, 5, 9, 0, 8, 11, 10, 24, 23, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 12]

            ]

            # 0.9trainval - 99.06
            # combinationss = [
            #     [0, 2, 5, 9, 7, 8, 6, 3, 4, 1],
            #     # [6,16,11,17,5,19,10,14,18,8,12,9,7,15,13,2,4,3,1,0],
            #     [7, 13, 15, 12, 0, 1, 17, 14, 8, 3, 19, 16, 18, 11, 10, 9, 5, 4, 2, 6],
            #     [67, 79, 91, 54, 60, 70, 92, 55, 71, 93, 98, 73, 57, 77, 83, 84, 61, 66, 58, 74, 46, 95, 86, 69, 64, 94,
            #      65, 62, 80, 50, 87, 76, 59, 82, 52, 47, 63, 78, 45, 81, 48, 53, 99, 51, 97, 90, 10, 75, 56, 89, 68, 19,
            #      85, 49, 88, 43, 12, 33, 44, 96, 40, 1, 34, 17, 28, 35, 72, 13, 27, 2, 21, 23, 38, 9, 32, 26, 36, 18,
            #      42, 25, 8, 11, 5, 39, 29, 15, 31, 3, 4, 14, 6, 0, 16, 24, 22, 7, 20, 37, 41, 30],
            #     [1, 6, 0, 4, 3, 7, 9, 8, 2, 5, 16, 20, 14, 17, 23, 12, 19, 18, 11, 15, 24, 22, 10, 21, 13]
            # ]
        # from best to worse
        elif dataset == "fashionmnist":

            #val
            combinationss = [
                [0, 1, 7, 2, 5, 9, 6, 8, 4, 3],
                [0, 5, 4, 13, 10, 7, 9, 15, 18, 17, 14, 12, 8, 19, 11, 16, 1, 3, 6, 2],
                [9, 35, 92, 60, 24, 76, 89, 75, 51, 82, 22, 43, 17, 50, 33, 53, 94, 79, 3, 21, 78, 10, 20, 81, 15, 87,
                 98, 80, 45, 95, 64, 34, 54, 39, 19, 71, 59, 18, 25, 68, 31, 73, 14, 91, 72, 6, 63, 13, 27, 57, 23, 67,
                 86, 69, 2, 58, 47, 40, 38, 49, 55, 61, 83, 29, 7, 66, 84, 11, 56, 74, 62, 1, 42, 46, 44, 41, 8, 4, 30,
                 88, 90, 48, 97, 5, 52, 28, 36, 70, 26, 77, 85, 96, 99, 37, 12, 32, 65, 16, 93, 0],
                [1, 9, 5, 7, 3, 8, 4, 2, 6, 0, 24, 15, 11, 17, 10, 12, 22, 21, 20, 16, 19, 14, 23, 18, 13]
            ]

            # complete
            # combinationss=[
            #
            #     [0, 7, 6, 5, 1, 2, 9, 3, 4, 8],
            #      [5, 10, 0, 13, 9, 4, 14, 12, 8, 17, 15, 7, 18, 1, 3, 11, 19, 2, 16, 6],
            #      [75, 4, 71, 15, 95, 57, 58, 9, 2, 7, 98, 73, 81, 52, 36, 34, 91, 22, 82, 55, 19, 49, 90, 42, 78, 99,
            #       46, 6, 65, 35, 31, 26, 51, 83, 10, 21, 74, 85, 14, 72, 12, 59, 38, 43, 29, 24, 18, 61, 23, 56, 97, 16,
            #       53, 5, 47, 67, 41, 8, 84, 44, 48, 69, 93, 96, 86, 1, 11, 27, 54, 62, 28, 17, 66, 87, 30, 0],
            #      [5, 1, 8, 9, 7, 3, 2, 4, 0, 6, 19, 13, 14, 24, 12, 20, 17, 18, 15, 11, 23, 21, 22, 10, 16]
            #
            # ]

        # these numbers from the end will be cut off, meaning the worse will be cut off
        #for i in range(len(combinationss)):
        #    combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:])

    elif method == 'noise':
        if dataset == 'mnist':
            combinationss = [
                [7, 2, 6, 0, 3, 4, 9, 8, 1, 5],
                [17, 11, 9, 6, 19, 10, 13, 3, 5, 18, 16, 14, 2, 15, 12, 0, 1, 4, 8, 7],
                [59, 91, 84, 68, 24, 27, 78, 52, 82, 93, 54, 98, 28, 49, 30, 31, 85, 29, 88, 83, 21, 60, 71, 42, 1, 80,
                 64, 50, 16, 22, 19, 65, 5, 4, 48, 44, 36, 79, 6, 41, 72, 35, 45, 33, 51, 92, 63, 10, 62, 40, 14, 18,
                 12, 67, 69, 15, 2, 95, 0, 96, 74, 34, 13, 7, 46, 3, 73, 57, 20, 43, 58, 11, 77, 25, 86, 17, 53, 23, 26,
                 8, 70, 39, 55, 38, 97, 90, 66, 61, 75, 94, 47, 56, 37, 9, 76, 32, 99, 87, 89, 81],
                [12, 11, 22, 10, 21, 14, 15, 7, 5, 3, 24, 23, 16, 18, 9, 6, 13, 2, 4, 17, 20, 19, 8, 1, 0]

            ]

        # these numbers from the end will be cut off, meaning the worse will be cut off
        #for i in range(len(combinationss)):
        #    combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:])



    elif method == 'random':
        combinationss = [np.random.permutation(nodesNum1), np.random.permutation(nodesNum2),
                         np.random.permutation(nodesFc1), np.random.permutation(nodesFc2)]
        #for i in range(len(combinationss)):
        #    combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:].copy())

    elif method == 'fisher':
        finetune()


        combinationss=[]
        for i in range(4):
            fisher_rank=np.argsort(net.running_fisher[i].detach().cpu().numpy())[::-1]
            combinationss.append(fisher_rank)
            #fisher_rank=torch.argsort(net.running_fisher[i], descending=True)
            #combinationss.append(fisher_rank.detach().cpu().numpy())



#         def _get_fisher():
#             masks = []
#             fisher = []
#
# #            self._update_cost(net)
#
#             for m in net.modules():
#                 if m._get_name() == self.module_name:
#                     masks.append(m.mask.detach())
#                     fisher.append(m.running_fisher.detach())
#
#                     # Now clear the fisher cache
#                     m.reset_fisher()
#
#             #self.masks = self.concat(masks)
#             self.fisher = self.concat(fisher)

        # _get_fisher()

        # tot_loss = self.fisher.div(prune_every) + 1e6 * (1 - self.masks)  # dummy value for off masks
        # print(len(tot_loss))
        # min, argmin = torch.min(tot_loss, 0)
        # self.prune(model, argmin.item())
        # self.prune_history.append(argmin.item())
    elif method=="switches":


        vi_training="integral"; print("vi training is "+ vi_training)
        getranks_method = 'train'
        combinationss = []
        num_samps_for_switch=1000

        if vi_training=="integral":
            print("integral evaluation")
            file_path=os.path.join(path_main, 'results_switch/results/combinationss_switch_9919_integral_samps_%i.npy' % num_samps_for_switch)

            if getranks_method=='train':

                epochs_num=7
                for layer in ["c1", "c3", "c5", "f6"]:
                    best_accuracy, epoch, best_model, S= run_experiment_integral(epochs_num, layer, 10, 20, 100, 25, num_samps_for_switch)
                    print("Rank for switches from most important/largest to smallest after %i " %  epochs_num)
                    print(S)
                    print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))
                    ranks_sorted = np.argsort(S.cpu().detach().numpy())[::-1]
                    print(",".join(map(str, ranks_sorted)))
                    combinationss.append(ranks_sorted)
                    combinationss.append(ranks_sorted)



                print('*'*30)
                print(combinationss)
                np.save(file_path, combinationss)
            elif getranks_method=='load':
                combinationss=list(np.load(file_path,  allow_pickle=True))

        elif vi_training=="point":
            print("mean")

            file_path = os.path.join(path_main, 'results_switch/results/combinationss_switch_9919_pointest.npy')

            if getranks_method == 'train':

                epochs_num = 7
                for layer in ["c1", "c3", "c5", "f6"]:
                    best_accuracy, epoch, best_model, S = run_experiment_pointest(epochs_num, layer, 10, 20, 100, 25)
                    print("Rank for switches from most important/largest to smallest after %i " % epochs_num)
                    print(S)
                    print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))
                    ranks_sorted = np.argsort(S.cpu().detach().numpy())[::-1]
                    print(",".join(map(str, ranks_sorted)))
                    combinationss.append(ranks_sorted)

                print('*' * 30)
                print(combinationss)
                np.save(file_path, combinationss)
            elif getranks_method == 'load':
                combinationss = list(np.load(file_path,  allow_pickle=True))


        #else:



    else:
        combinationss = []
        # magnitude_rank.setup('lenet', dataset)
        combinationss = magnitude_rank.get_ranks(method, net)
        # for i in range(4):
        #     combinationss.append(torch.LongTensor(combinat[i]))

        # these numbers from the end will be cut off, meaning the worse will be cut off
        # these numbers from the end will be cut off, meaning the worse will be cut off
        #print(combinationss)

        #for i in range(len(combinationss)):
        #    combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:].copy())



    return  combinationss


##################################################################################
# RETRAIN

def threshold_prune_and_retrain(combinationss, thresh):



    #PRINT NAMED_PARAMETERS
    # for name, param in net.named_parameters():
    #    print (name)
    #    print(param.shape)

    ##### THRESHOLD
    # the ranks are sorted from best to worst
    # thresh is what we keep, combinationss is what we discard


    for i in range(len(combinationss)):
        combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:].copy())




    #filename = "%s_retrained_paramsearch1.txt" % path
    #
    # if write:
    #     with open(filename, "a+") as file:
    #         file.write("\n\nprunedto:%d_%d_%d_%d\n\n" % (thresh[0], thresh[1], thresh[2], thresh[3]))
    print("\n\nprunedto:%d_%d_%d_%d\n" % (thresh[0], thresh[1], thresh[2], thresh[3]))

    #################################################################################################################3
    ########## PRUNE/ ZERO OUT THE WEIGHTS

    net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'], strict=False)
    #net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage), strict=False)

    if prune_bool:

        it = 0
        for name, param in net.named_parameters():
            print(name)
            if (("c" in name) or ("f" in name)) and ("weight" in name):
                it += 1
                param.data[combinationss[it - 1]] = 0
                # print(param.data)
            if (("c" in name) or ("f" in name)) and ("bias" in name):
                param.data[combinationss[it - 1]] = 0
                # print(param.data)
            if ("bn" in name) and ("weight" in name):
                param.data[combinationss[it - 1]] = 0
            if ("bn" in name) and ("bias" in name):
                param.data[combinationss[it - 1]] = 0



        # net.c1.weight.data[combination]=0; net.c1.bias.data[combination] = 0
        # net.c3.weight.data[combination2] = 0; net.c3.bias.data[combination2] = 0
        # net.c5.weight.data[combination3] = 0;net.c5.bias.data[combination3] = 0
        # net.f6.weight.data[combination4] = 0;net.f6.bias.data[combination4] = 0

        print("After pruning")
        acc=evaluate()

    ##################################################################### RETRAIN


    if retrain:
        def gradi(module):
            module[combinationss[0]]=0
            #print(module[21])
        net.c1.weight.register_hook(gradi)
        net.c1.bias.register_hook(gradi)
        net.bn1.weight.register_hook(gradi)
        net.bn1.bias.register_hook(gradi)
        #h = net.c1.weight.register_hook(lambda gradi: gradi[1] * 0)

        def gradi2(module):
            module[combinationss[1]]=0
            #print(module[1])

        net.c3.weight.register_hook(gradi2)
        net.c3.bias.register_hook(gradi2)
        net.bn2.weight.register_hook(gradi2)
        net.bn2.bias.register_hook(gradi2)

        #h2 = net.c3.weight.register_hook(gradi2)

        def gradi3(module):
            module[combinationss[2]] = 0
            # print(module[1])

        net.c5.weight.register_hook(gradi3)
        net.c5.bias.register_hook(gradi3)

        def gradi4(module):
            module[combinationss[3]] = 0
            # print(module[1])

        net.f6.weight.register_hook(gradi4)
        net.f6.bias.register_hook(gradi4)


        print("Retraining")

        train(thresh)

    return acc



#######################

#SAVING MODEL
#the models are saved in the savedirectory as the original model
if dataset=="mnist":
    save_accuracy=99.00
if dataset=="fashionmnist":
    save_accuracy=89.50

save=False
#WRITING
# the output text file will be saved also in the same directory as the original model
write_training=False
#################################

resume=True
prune_bool=True
retrain=True


##############################

file_write=False #oly one file_write here (and one read fie)
comp_combinations=False

#################################3################
################################################



if resume:
    load_model()
    evaluate()
    if comp_combinations:
        compute_combinations_lenet(True, net, evaluate, dataset, "zeroing") #can be "additive noise instead of zeroing


    #methods=['switches', 'l1', 'l2', 'fisher','filter_ranking']
    methods=['switches']

    #
    # for method in methods:
    #     print("\n", method)
        #numbers indicate how many are pruned

        # for i in range(1,10):
        #     print("\n\n******method: %s, percentage: %d******" % (method, i*10))
        #     prune(True, 1*i, 2*i, 10*i, 2*i, write, save)

    #l1, l2, switches, fisher

    if prune_bool:

        # pruned_architectures=ParameterGrid({'c1':[3, 4, 5, 6, 7, 10], 'c3': [4, 6, 8, 10, 12, 20], 'c5': [20, 30, 40, 50, 60, 100], 'f6': [5, 10, 15, 20, 25]})
        #
        # for pruned_arch in pruned_architectures:

        pruned_arch_layer=[int(n) for n in args.arch.split(",")]
        pruned_arch={}
        pruned_arch['c1']=pruned_arch_layer[0]; pruned_arch['c3']=pruned_arch_layer[1]; pruned_arch['c5']=pruned_arch_layer[2];pruned_arch['f6']=pruned_arch_layer[3];

        if 1:
            #load_model()
            accs={}
            for method in methods:
                print("\n\n %s \n" % method)
                combinationss = get_ranks(method); print(combinationss)
                acc=threshold_prune_and_retrain(combinationss, [pruned_arch['c1'], pruned_arch['c3'], pruned_arch['c5'], pruned_arch['f6']])
                accs[method]=acc
                #prune(False, i1, i2, i3, i4, write, save)

            if accs['filter_ranking']>accs['fisher'] and accs['filter_ranking']>accs['l1']:
                print("Yay!")
            print("\n*********************************\n\n")
        #prune_and_retrain([10,10,50,10])

if resume==False and prune_bool==False and retrain==False:
    train()


print("\n\nEND")
