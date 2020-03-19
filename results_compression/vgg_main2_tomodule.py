# 1. It trains the network from scratch (if "resume" is off)
# 2. if resume is on, it loads the pretrained model,
#     if prune_ bool , it prunes the network
#     if retrain_ bool is on , retrains it
# e.g. it can retrain, but not prune
# 3. it can be used for visualizing, if uncomment the comments #VISU

# other features:
# it loads the ranks from shapley or switches (ranks_path = '../Dir_switch/results/cifar/vgg_93.92/switch_init_-1, alpha_2/')

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import sys
import socket

import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as f
import logging
import matplotlib.pyplot as plt
import magnitude_rank
import argparse
from vgg_computeComb2 import compute_combinations
from vgg_computeComb2 import test_val
import argparse
from itertools import product

# file_dir = os.path.dirname("utlis.p")
# sys.path.append(file_dir)

# from models import *

# from utils import progress_bar
import torch
import torch.nn as nn




#
# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU(inplace)
#   (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (5): ReLU(inplace)
#   (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (9): ReLU(inplace)
#   (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (12): ReLU(inplace)
#   (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (16): ReLU(inplace)
#   (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (19): ReLU(inplace)
#   (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (22): ReLU(inplace)
#   (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (26): ReLU(inplace)
#   (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (29): ReLU(inplace)
#   (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (32): ReLU(inplace)
#   (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (36): ReLU(inplace)
#   (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (39): ReLU(inplace)
#   (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (42): ReLU(inplace)
#   (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (44): AvgPool2d(kernel_size=1, stride=1, padding=0)
# )


network_structure_dummy=0

#######
# path stuff
cwd = os.getcwd()
if 'g0' in socket.gethostname() or 'p0' in socket.gethostname():
    #the cwd is where the sub file is so ranking/
    sys.path.append(os.path.join(cwd, "results_switch"))
    path_compression = os.path.join(cwd, "results_compression")
    path_networktest = os.path.join(cwd, "results_networktest")
    path_switch = os.path.join(cwd, "results_switch")
    path_main= cwd
else:
    #the cwd is results_compression
    parent_path = os.path.abspath('..')
    sys.path.append(os.path.join(parent_path, "results_switch"))
    path_compression = cwd
    path_networktest = os.path.join(parent_path, "results_networktest")
    path_switch = os.path.join(parent_path, "results_switch")
    path_main= parent_path

print(cwd)
print(sys.path)

print("newh2")
sys.path.append(os.path.join(path_networktest, "external_codes/pytorch-cifar-master/models"))
sys.path.append(path_compression)



##############################
# PARAMETERS

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default='25,25,65,80,201,158,159,460,450,490,470,465,465,450')
#parser.add_argument("--arch", default='25,25,65,80,201,158,159,460,450,490,470,465,465,450')
# ar.add_argument("-arch", default=[21,20,65,80,201,147,148,458,436,477,454,448,445,467,441])
parser.add_argument('--layer', help="layer to prune", default="c1")
parser.add_argument("--method", default='l1')
parser.add_argument("--switch_samps", default=100, type=int)
parser.add_argument("--ranks_method", default='point')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
save_accuracy=90.0

args = parser.parse_args()
print(args.layer)
print("aaa", args.arch)



#######################################################################################################################################3
############################################################
# NETWORK


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG15': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512],
    #'VGG15_comp': [39, 39, 63, 48, 55, 88, 87, 52, 62, 22, 42, 47, 47, 47],
    'VGG15_comp': [34, 34, 68, 68, 75, 106, 101, 92, 102, 92, 67, 67, 62, 62],

    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()

        cfg_arch=cfg['VGG15_comp']

        self.c1 = nn.Conv2d(3, cfg_arch[0], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(cfg_arch[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c2 = nn.Conv2d(cfg_arch[0], cfg_arch[1], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(cfg_arch[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp1 = nn.MaxPool2d(2)

        self.c3 = nn.Conv2d(cfg_arch[1], cfg_arch[2], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(cfg_arch[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c4 = nn.Conv2d(cfg_arch[2], cfg_arch[3], 3, padding=1)
        self.bn4 = nn.BatchNorm2d(cfg_arch[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp2 = nn.MaxPool2d(2)

        self.c5 = nn.Conv2d(cfg_arch[3], cfg_arch[4], 3, padding=1)
        self.bn5 = nn.BatchNorm2d(cfg_arch[4], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c6 = nn.Conv2d(cfg_arch[4], cfg_arch[5], 3, padding=1)
        self.bn6 = nn.BatchNorm2d(cfg_arch[5], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c7 = nn.Conv2d(cfg_arch[5], cfg_arch[6], 3, padding=1)
        self.bn7 = nn.BatchNorm2d(cfg_arch[6], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp3 = nn.MaxPool2d(2)

        self.c8 = nn.Conv2d(cfg_arch[6], cfg_arch[7], 3, padding=1)
        self.bn8 = nn.BatchNorm2d(cfg_arch[7], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c9 = nn.Conv2d(cfg_arch[7], cfg_arch[8], 3, padding=1)
        self.bn9 = nn.BatchNorm2d(cfg_arch[8], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c10 = nn.Conv2d(cfg_arch[8], cfg_arch[9], 3, padding=1)
        self.bn10 = nn.BatchNorm2d(cfg_arch[9], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp4 = nn.MaxPool2d(2)

        self.c11 = nn.Conv2d(cfg_arch[9], cfg_arch[10], 3, padding=1)
        self.bn11 = nn.BatchNorm2d(cfg_arch[10], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c12 = nn.Conv2d(cfg_arch[10], cfg_arch[11], 3, padding=1)
        self.bn12 = nn.BatchNorm2d(cfg_arch[11], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c13 = nn.Conv2d(cfg_arch[11], cfg_arch[12], 3, padding=1)
        self.bn13 = nn.BatchNorm2d(cfg_arch[12], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp5 = nn.MaxPool2d(2)

        self.l1 = nn.Linear(cfg_arch[12], cfg_arch[13])
        self.l3 = nn.Linear(cfg_arch[13], 10)
        self.d1 = nn.Dropout()
        self.d2 = nn.Dropout()

        self.parameter = Parameter(-1 * torch.ones(64), requires_grad=True)  # this parameter lies #S

        # Fisher method is called on backward passes
        self.running_fisher = []
        # for i in range(3):
        #     self.running_fisher.append(torch.Tensor(64).to(device))
        # for i in range(2):
        #     self.running_fisher.append(torch.Tensor(128).to(device))
        # for i in range(3):
        #     self.running_fisher.append(torch.Tensor(256).to(device))
        # for i in range(7):
        #     self.running_fisher.append(torch.Tensor(512).to(device))

        self.running_fisher.append(torch.Tensor(64).to(device)) #first dummy for 0
        for i in range(len(cfg_arch)):
            self.running_fisher.append(torch.Tensor(cfg_arch[i]).to(device))


        self.act = [0] * 15

        self.activation1 = Identity()
        self.activation2 = Identity()
        self.activation3 = Identity()
        self.activation4 = Identity()
        self.activation5 = Identity()
        self.activation6 = Identity()
        self.activation7 = Identity()
        self.activation8 = Identity()
        self.activation9 = Identity()
        self.activation10 = Identity()
        self.activation11 = Identity()
        self.activation12 = Identity()
        self.activation13 = Identity()
        self.activation14 = Identity()

        self.activation1.register_backward_hook(self._fisher1)
        self.activation2.register_backward_hook(self._fisher2)
        self.activation3.register_backward_hook(self._fisher3)
        self.activation4.register_backward_hook(self._fisher4)
        self.activation5.register_backward_hook(self._fisher5)
        self.activation6.register_backward_hook(self._fisher6)
        self.activation7.register_backward_hook(self._fisher7)
        self.activation8.register_backward_hook(self._fisher8)
        self.activation9.register_backward_hook(self._fisher9)
        self.activation10.register_backward_hook(self._fisher10)
        self.activation11.register_backward_hook(self._fisher11)
        self.activation12.register_backward_hook(self._fisher12)
        self.activation13.register_backward_hook(self._fisher13)
        self.activation14.register_backward_hook(self._fisher14)

    # def forward(self, x, i):  # VISU
    def forward(self, x, i=-1):
        phi = f.softplus(self.parameter)
        S = phi / torch.sum(phi)
        # Smax = torch.max(S)
        # Sprime = S/Smax
        Sprime = S

        if vis:
            for filter_num in range(3):
                mm = x.cpu().detach().numpy()
                # Split
                img = mm[1, filter_num, :, :]
                if filter_num == 0:
                    cmap_col = 'Reds'
                elif filter_num == 1:
                    cmap_col = 'Greens'
                elif filter_num == 2:
                    cmap_col = 'Blues'

                # plt.imshow(matrix)  # showing 2nd channel (example of a channel)

                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                cax_00 = plt.imshow(img, cmap=cmap_col)
                plt.show()
                # plt.savefig(
                #     "/home/kamil/Dropbox/Current_research/python_tests/results_networktest/vis/feature_maps/cifar/94.34/input_batch%d_filternum%d" % (
                #         i, filter_num), bbox_inches='tight', pad_inches=0)

        output = f.relu(self.bn1(self.c1(x)))

        if vis:
            for filter_num in range(64):
                mm = output.cpu().detach().numpy()

                matrix = mm[1, filter_num, :, :]
                print(filter_num)
                # print(matrix[0:20, 0])
                # ave=0
                ave = np.average(matrix[0:20, 0])
                matrix = matrix - ave

                plt.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)  # showing 2nd channel (example of a channel)

                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())

                # plt.savefig(
                #     "/home/kamil/Dropbox/Current_research/python_tests/results_networktest/vis/feature_maps/cifar/94.34/conv1_batch%d_filternum%d" % (
                #     i, filter_num), bbox_inches='tight', pad_inches=0)

        # out = self.activation1(output)
        self.act[1] = self.activation1(output)
        output = f.relu(self.bn2(self.c2(output)))
        self.act[2] = self.activation2(output)
        output = self.mp1(output)

        output = f.relu(self.bn3(self.c3(output)))
        self.act[3] = self.activation3(output)
        output = f.relu(self.bn4(self.c4(output)))
        self.act[4] = self.activation4(output)
        output = self.mp2(output)

        output = f.relu(self.bn5(self.c5(output)))
        self.act[5] = self.activation5(output)
        output = f.relu(self.bn6(self.c6(output)))
        self.act[6] = self.activation6(output)
        output = f.relu(self.bn7(self.c7(output)))
        self.act[7] = self.activation7(output)
        output = self.mp3(output)

        output = f.relu(self.bn8(self.c8(output)))
        self.act[8] = self.activation8(output)
        output = f.relu(self.bn9(self.c9(output)))
        self.act[9] = self.activation9(output)
        output = f.relu(self.bn10(self.c10(output)))
        self.act[10] = self.activation10(output)
        output = self.mp4(output)

        output = f.relu(self.bn11(self.c11(output)))
        self.act[11] = self.activation11(output)
        output = f.relu(self.bn12(self.c12(output)))
        self.act[12] = self.activation12(output)
        output = f.relu(self.bn13(self.c13(output)))
        self.act[13] = self.activation13(output)
        output = self.mp5(output)

        output = output.view(-1, cfg['VGG15_comp'][13])
        output = self.l1(output)
        self.act[14] = self.activation14(output)
        output = self.l3(output)

        return output

    def _fisher1(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 1)

    def _fisher2(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 2)

    def _fisher3(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 3)

    def _fisher4(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 4)

    def _fisher5(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 5)

    def _fisher6(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 6)

    def _fisher7(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 7)

    def _fisher8(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 8)

    def _fisher9(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 9)

    def _fisher10(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 10)

    def _fisher11(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 11)

    def _fisher12(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 12)

    def _fisher13(self, notused1, notused2, grad_output):
        self._fisher(grad_output, 13)

    def _fisher14(self, notused1, notused2, grad_output):
        self._fisher_fc(grad_output, 14)

    def _fisher(self, grad_output, i):
        act = self.act[i].detach()
        grad = grad_output[0].detach()
        #
        # print("Grad: ",grad_output[0].shape)
        # print("Act: ", act.shape, '\n')

        g_nk = (act * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        # print(del_k.shape)
        # print(i)
        self.running_fisher[i] += del_k

    def _fisher_fc(self, grad_output, i):
        act = self.act[i].detach()
        grad = grad_output[0].detach()
        #
        # print("Grad: ",grad_output[0].shape)
        # print("Act: ", act.shape, '\n')

        g_nk = (act * grad)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        # print(del_k.shape)
        # print(i)
        self.running_fisher[i] += del_k

    def reset_fisher(self):
        for i in range(len(self.running_fisher)):
            self.running_fisher[i] = torch.Tensor(len(self.running_fisher[i])).to(device)

    # def forward(self, x):
    #     out = self.features(x)
    #     out = out.view(out.size(0), -1)
    #     out = self.classifier(out)
    #     return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


#####################################
# DATA

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

trainval_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
# with more workers there may be an error in debug mode: RuntimeError: DataLoader worker (pid 29274) is killed by signal: Terminated.


trainval_perc = 0.8
train_size = int(trainval_perc * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
torch.manual_seed(0)
train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
# torch.save(val_dataset, "val_dataset")
# torch.save(train_dataset, "train_dataset")


trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

###################################################
# MAKE AN INSTANCE OF A NETWORK AND (POSSIBLY) LOAD THE MODEL


print('==> Building model..')
net = VGG('VGG16')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)

# for name, param in net.named_parameters():
#     print (name, param.shape)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(device)


#########################
#
#  def func(name):
#     def hook(module, grad_input, grad_output):
#
#     # hook implementation
#     return hook
#
# for name, layer in net.named_modules():
#     layer.register_backward_hook(func(name))




a=0

#######################################################
#TRAIN


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # outputs = net(inputs, batch_idx) #VISU
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # if (batch_idx % 1000 ==0):
    print('Training Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
    train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100. * correct / total, best_acc


###############################################################

def finetune():
    # switch to train mode
    net.train()

    dataiter = iter(trainloader)

    for i in range(0, 100):

        try:
            input, target = dataiter.next()
        except StopIteration:
            dataiter = iter(trainloader)
            input, target = dataiter.next()

        input, target = input.to(device), target.to(device)

        # compute output
        output = net(input)

        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#################################################################
# TEST



def test(dataset):
    # for name, param in net.named_parameters():
    #     print (name)
    #     print (param)
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # outputs = net(inputs, batch_idx) #VISU
            outputs = net(inputs, batch_idx)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Test Lossds: %.3f | Acc: %.3f%% (%d/%d)' % (
    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return 100.0 * float(correct) / total


def testval():
    # for name, param in net.named_parameters():
    #     print (name)
    #     print (param)
    global best_acc
    net.eval()
    # net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # outputs = net(inputs, batch_idx) #VISU
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # if (predicted.eq(targets).sum().item()) != 128:
            #     print(predicted.eq(targets).sum().item())
            #     print(predicted.eq(targets))
            #     print(predicted)
            #     print(targets)
            # else:
            #     print(predicted)
            # print("----------------------------------------------")

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Test Lossds: %.3f | Acc: %.3f%% (%d/%d)' % (
    test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return 100.0 * float(correct) / total


########################################
## just LOAD MODEL AND SAVE




def load_model(test_bool=True):
    # Load checkpoint.
    # print('==> Resuming from checkpoint..')
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(model2load)
    # checkpoint = torch.load('./checkpoint/ckpt_vgg16_prunedto[39, 39, 63, 48, 55, 98, 97, 52, 62, 22, 42, 47, 47, 42, 62]_64.55.t7')
    net.load_state_dict(checkpoint['net'], strict=False)
    # print(net.module.c1.weight)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    if test_bool:
        print("Accuracy of the tested model: ")

        test(-1)
    print("----")
    # testval(-1)
    # test_val(-1, net)
    if comp_combinations:
        compute_combinations(True, net, testval, args.layer)



######################################################
# SAVE experiment

def save_checkpoint(epoch, acc, best_acc, remaining=0):
    # Save checkpoint.
    # acc = test(epoch)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        print("acc: ", acc)
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if acc > save_accuracy:
            if remaining == 0:  # regular training
                torch.save(state, path_compression+'/checkpoint/ckpt_vgg16_{}.t7'.format(acc))
            else:
                torch.save(state, path_compression+'/checkpoint/ckpt_vgg16_prunedto{}_{}.t7'.format(remaining, acc))
        best_acc = acc

    return best_acc


##############################################
# PRUNEand RETRAIN

def prune_and_retrain(thresh):

    #LOAD
    load_model(False)
    # PRINT
    # for name, param in net.named_parameters():
    #    print (name)
    #    print(param.shape)

    dummy=0

    # PRUNE

    if prune_bool:
        ############################3
        # READ THE RANKS

        #it seems that unlike in lenet here the ranks are from worse to best
        if method == 'switch':
            ranks_method=args.ranks_method
            switches_epoch = 10

            if ranks_method == 'shapley':
                combinationss = []
                shapley_file = open(
                    "/home/kamil/Dropbox/Current_research/python_tests/results_shapley/combinations/94.34/zeroing_0.2val/shapley.txt")
                for line in shapley_file:
                    line = line.strip()[1:-2]
                    nums = line.split(",")
                    nums_int = [int(i) for i in nums]
                    combinationss.append(nums_int)

                # combinationss=torch.load('results/ranks/ranks_93.92_shapley.pt')
                # for i in range(1,14):
                #     name="../results_shapley/results/vgg_"+str(orig_accuracy)+"/shapley_"+str(orig_accuracy)+"_vgg16_"+str(i)+".npy"
                #     shapley_rank=np.load(name)
                #     combinationss.append(shapley_rank)
                # name = "../results_shapley/results/vgg_" + str(orig_accuracy) + "/shapley_" + str(orig_accuracy) + "_vgg16_l1.npy"
                # shapley_rank = np.load(name)
                # combinationss.append(shapley_rank)
            # elif ranks_method=='switches':
            #     #combinationss=torch.load('results/ranks/ranks_93.92_switches.pt')
            #     combinationss=[0]*15
            #     ranks_path = '../Dir_switch/results/cifar/vgg_93.92/switch_init_0.05, alpha_0.05, annealing_6000000/'
            #     #ranks_path = '../Dir_switch/results/cifar/vgg_93.92/switch_init_-1, alpha_2/'
            #     for i in range(len(combinationss)):
            #         ranks_filepath = ranks_path + "93.92_alpha0.05_switchinit0.05_conv" + str(i + 1) + "_ep"+str(switches_epoch)+".pt"

            elif ranks_method == 'integral':
                # combinationss=torch.load('results/ranks/ranks_93.92_switches.pt')
                #combinationss = [0] * len(cfg['VGGBC'])  # 15
                ranks_path = path_switch+'/results/switch_data_cifar_integral_samps_%i_epochs_7.npy' % args.switch_samps

                # for i in range(len(combinationss)):
                #     ranks_filepath = ranks_path + "93.92_conv" + str(i + 1) + "_ep49.pt"
                #
                #     switch_values = torch.load(ranks_filepath)
                #     # print(switch_values)
                #     # combinationss[i]=torch.argsort(switch_values)
                #     # combinationss[i]=torch.LongTensor(np.argsort(switch_values.cpu().detach().numpy()).copy())#argsort is increasing order, we want decreasing hence [::-1]
                #
                #     combinationss[i] = torch.LongTensor(np.argsort(switch_values.cpu().detach().numpy())[
                #                                         ::-1].copy())  # argsort is increasing order, we want decreasing hence [::-1]
                #     # print(combinationss[i])
                #     # print("new")
                #
                # file = open("switches.txt", "a")
                # for comb in combinationss:
                #     comb_det = comb.detach().cpu().numpy()
                #     comb_str = ",".join([str(a) for a in comb_det])
                #     file.write(comb_str)
                #     file.write("\n")

                combinationss=list(np.load(ranks_path,  allow_pickle=True).item()['combinationss'])

            elif ranks_method == 'point':
                print(ranks_method)
                # combinationss=torch.load('results/ranks/ranks_93.92_switches.pt')
                #combinationss = [0] * len(cfg['VGGBC'])  # 15
                ranks_path = path_switch+'/results/switch_data_cifar_point_epochs_7.npy'


                combinationss=list(np.load(ranks_path,  allow_pickle=True).item()['combinationss'])


            # these numbers from the beginning will be cut off, meaning the worse will be cut off
            #for i in range(len(combinationss)):
            #    combinationss[i] = torch.LongTensor(combinationss[i][:thresh[i]])
            #print(combinationss[1])
            #we change to the other way around
            for i in range(len(combinationss)):
                combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:].copy())


        elif method == 'l1' or method == 'l2':
            magnitude_rank.setup()
            combinationss = magnitude_rank.get_ranks(method, net)
            # for i in range(4):
            #     combinationss.append(torch.LongTensor(combinat[i]))

            # file = open("l1.txt", "a")
            # for comb in combinationss:
            #     comb_det = comb
            #     comb_str = ",".join([str(a) for a in comb_det])
            #     file.write(comb_str)
            #     file.write("\n")

            # these numbers from the beginning will be cut off, meaning the worse will be cut off
            for i in range(len(combinationss)):
                combinationss[i] = torch.LongTensor(combinationss[i][:thresh[i]].copy())
            print(combinationss[1])


        elif method == 'fisher':
            # in the process of finetuning we accumulate the gradient information that w eadd for each batch. We use this gradient info for constructing a ranking.
            net.module.reset_fisher()
            finetune()

            combinationss = []
            for i in range(14):
                fisher_rank = torch.argsort(net.module.running_fisher[i], descending=True)
                combinationss.append(fisher_rank.detach().cpu())

            # print(combinationss)
            # file=open("fisher.txt", "a")
            # for comb in combinationss:
            #     comb_det=comb.detach().cpu().numpy()
            #     comb_str = ",".join([str(a) for a in comb_det])
            #     file.write(comb_str)
            #     file.write("\n")

            # these numbers from the beginning will be cut off, meaning the worse will be cut off
            for i in range(len(combinationss)):
                combinationss[i] = torch.LongTensor(combinationss[i][:thresh[i]])
            print(combinationss[1])

        # PRINT THE PRUNED ARCHITECTURE
        remaining = []
        for i in range(len(combinationss)):
            print(cfg['VGGBC'][i], len(combinationss[i]))
            remaining.append(int(cfg['VGGBC'][i]) - len(combinationss[i]))
        print(remaining)

        # PRUNE

        it = 0
        for name, param in net.named_parameters():
            # print(name, param.shape)
            if "module.c" in name and "weight" in name:
                it += 1
                param.data[combinationss[it - 1]] = 0
                # print(param.data)
            if "module.c" in name and "bias" in name:
                param.data[combinationss[it - 1]] = 0
                # print(param.data)
            if ("bn" in name) and ("weight" in name):
                param.data[combinationss[it - 1]] = 0
            if ("bn" in name) and ("bias" in name):
                param.data[combinationss[it - 1]] = 0

        # checking pruning
        # for name, param in net.named_parameters():
        #     if "c1.bias" in name:
        #         print(name)
        #         print(param)

        #
        #     combinationss[i]
        #
        # net.c1.weight.data[combination]=0; net.c1.bias.data[combination] = 0
        # net.c3.weight.data[combination2] = 0; net.c3.bias.data[combination2] = 0
        # net.c5.weight.data[combination3] = 0;net.c5.bias.data[combination3] = 0
        # net.f6.weight.data[combination4] = 0;net.f6.bias.data[combination4] = 0

        print("After pruning")

        test(-1)

        ######## RETRAINING

        def gradi1(module):
            module[combinationss[0]] = 0
            # print(module[21])

        net.module.c1.weight.register_hook(gradi1)
        net.module.c1.bias.register_hook(gradi1)
        net.module.bn1.weight.register_hook(gradi1)
        net.module.bn1.bias.register_hook(gradi1)

        def gradi2(module):
            module[combinationss[1]] = 0
            # print(module[21])

        net.module.c2.weight.register_hook(gradi2)
        net.module.c2.bias.register_hook(gradi2)
        net.module.bn2.weight.register_hook(gradi2)
        net.module.bn2.bias.register_hook(gradi2)

        def gradi3(module):
            module[combinationss[2]] = 0
            # print(module[21])

        net.module.c3.weight.register_hook(gradi3)
        net.module.c3.bias.register_hook(gradi3)
        net.module.bn3.weight.register_hook(gradi3)
        net.module.bn3.bias.register_hook(gradi3)

        def gradi4(module):
            module[combinationss[3]] = 0
            # print(module[21])

        net.module.c4.weight.register_hook(gradi4)
        net.module.c4.bias.register_hook(gradi4)
        net.module.bn4.weight.register_hook(gradi4)
        net.module.bn4.bias.register_hook(gradi4)

        def gradi5(module):
            module[combinationss[4]] = 0
            # print(module[21])

        h1 = net.module.c5.weight.register_hook(gradi5)
        h1 = net.module.c5.bias.register_hook(gradi5)
        h12 = net.module.bn5.weight.register_hook(gradi5)
        h13 = net.module.bn5.bias.register_hook(gradi5)

        def gradi6(module):
            module[combinationss[5]] = 0
            # print(module[21])

        h1 = net.module.c6.weight.register_hook(gradi6)
        h1 = net.module.c6.bias.register_hook(gradi6)
        h12 = net.module.bn6.weight.register_hook(gradi6)
        h13 = net.module.bn6.bias.register_hook(gradi6)

        def gradi7(module):
            module[combinationss[6]] = 0
            # print(module[21])

        h1 = net.module.c7.weight.register_hook(gradi7)
        h1 = net.module.c7.bias.register_hook(gradi7)
        h12 = net.module.bn7.weight.register_hook(gradi7)
        h13 = net.module.bn7.bias.register_hook(gradi7)

        def gradi8(module):
            module[combinationss[7]] = 0
            # print(module[21])

        h1 = net.module.c8.weight.register_hook(gradi8)
        h1 = net.module.c8.bias.register_hook(gradi8)
        h12 = net.module.bn8.weight.register_hook(gradi8)
        h13 = net.module.bn8.bias.register_hook(gradi8)

        def gradi9(module):
            module[combinationss[8]] = 0
            # print(module[21])

        h1 = net.module.c9.weight.register_hook(gradi9)
        h1 = net.module.c9.bias.register_hook(gradi9)
        h12 = net.module.bn9.weight.register_hook(gradi9)
        h13 = net.module.bn9.bias.register_hook(gradi9)

        def gradi10(module):
            module[combinationss[9]] = 0
            # print(module[21])

        h1 = net.module.c10.weight.register_hook(gradi10)
        h1 = net.module.c10.bias.register_hook(gradi10)
        h12 = net.module.bn10.weight.register_hook(gradi10)
        h13 = net.module.bn10.bias.register_hook(gradi10)

        def gradi11(module):
            module[combinationss[10]] = 0
            # print(module[21])

        h1 = net.module.c11.weight.register_hook(gradi11)
        h1 = net.module.c11.bias.register_hook(gradi11)
        h12 = net.module.bn11.weight.register_hook(gradi11)
        h13 = net.module.bn11.bias.register_hook(gradi11)

        def gradi12(module):
            module[combinationss[11]] = 0
            # print(module[21])

        h1 = net.module.c12.weight.register_hook(gradi12)
        h1 = net.module.c12.bias.register_hook(gradi12)
        h12 = net.module.bn12.weight.register_hook(gradi12)
        h13 = net.module.bn12.bias.register_hook(gradi12)

        def gradi13(module):
            module[combinationss[12]] = 0
            # print(module[21])

        h1 = net.module.c13.weight.register_hook(gradi13)
        h1 = net.module.c13.bias.register_hook(gradi13)
        h12 = net.module.bn13.weight.register_hook(gradi13)
        h13 = net.module.bn13.bias.register_hook(gradi13)

        def gradi14(module):
            module[combinationss[13]] = 0

        h1 = net.module.l1.weight.register_hook(gradi14)
        h1 = net.module.l1.bias.register_hook(gradi14)
        h12 = net.module.l1.weight.register_hook(gradi14)
        h13 = net.module.l1.bias.register_hook(gradi14)

        # def gradi15(module):
        #     module[combinationss[14]] = 0
        #     # print(module[21])
        # h1 = net.module.c15.weight.register_hook(gradi15)
        # h1 = net.module.c15.bias.register_hook(gradi15)
        # h12 = net.module.bn15.weight.register_hook(gradi15)
        # h13 = net.module.bn15.bias.register_hook(gradi15)



        # it = -1
        # for name, param in net.named_parameters():
        #     print(name)
        #     if "module.c" in name and "weight" in name:
        #         it += 1
        #
        #         def gradi(module):
        #             module[combinationss[it]] = 0
        #
        #         h1 = param.register_hook(gradi)
        #
        #     if "module.c" in name and "bias" in name:
        #         #param.data[combinationss[it - 1]] = 0
        #         #print(param.data)
        #
        #         def gradi(module):
        #             module[combinationss[it]] = 0
        #
        #         h1 = param.register_hook(gradi)

    #######################################################

    #RETRAIN



    if retrain_bool:
        print("Retraining")

        filename = "retrained_paramsearch1_vgg.txt"
        # with open(filename, "a+") as file:
        # file.write("---NEW EXPERIMENT-----")
        # if prune_bool:
        #     file.write("\n\nprunedto:%s\n\n" % (" ".join(str(e) for e in remaining)))

        path = path_compression+"/checkpoint/"

        # here retraining works
        net.train()
        stop = 0;
        epoch = 0;
        best_accuracy = 0;
        entry = np.zeros(3);
        best_model = -1;
        early_stopping = 100
        optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
        while (stop < early_stopping):
            epoch = epoch + 1
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                net.module.c2.weight.grad  # the hook is automatically applied, here we just check the gradient
                optimizer.step()
                # net.c1.weight.data[1] = 0  # instead of hook
                # net.c1.bias.data[1] = 0  # instead of hook
                # if i % 100==0:
                #    print (i)
                #   print (loss.item())

            print(loss.item())
            accuracy = test(-1)
            # print(net.module.c2.weight.data)
            print("Epoch " + str(epoch) + " ended.")

            # check if pruned weights are pruned
            # for name, param in net.named_parameters():
            #     if "1.weight" in name:
            #         print(name)
            #         print(param)

            if (accuracy <= best_accuracy):
                stop = stop + 1
                entry[2] = 0
            else:
                if accuracy > 90.5:
                    best_accuracy = save_checkpoint(epoch, accuracy, best_accuracy,
                                                    remaining)  # compares accuracy and best_accuracy by itself again
                    # if prune_bool:
                    #     torch.save(best_model, "{}_retrained_epo-{}_prunedto-{}_acc-{:.2f}".format(path, epoch, remaining, best_accuracy))
                    # else:
                    #     torch.save(best_model, "%s_retrained_epo-%d_only_acc-%.2f" % (path, epoch, best_accuracy))
                print("Best updated")
                stop = 0
                entry[2] = 1
                best_model = net.state_dict()

                entry[0] = accuracy;
                entry[1] = loss
                # with open(filename, "a+") as file:
                #     file.write("\n Epoch: %d\n" % epoch)
                #     file.write(",".join(map(str, entry)) + "\n")
                #     if (accuracy>98.9):
                #         file.write("Yes\n")
                #     elif (accuracy>98.8):
                #         file.write("Ok\n")

        print(loss.item())
        accuracy = test(-1)





#################################################################

model2load = path_compression+'/checkpoint/ckpt_vgg16_94.34.t7'
orig_accuracy = 94.34
# if all False just train thenetwork
resume = False
prune_bool = False
retrain_bool = False  # whether we retrain the model or just evaluate

comp_combinations = False  # must be with resume #with retrain if we want to retrain combinations
vis = False
# file_write=True
# compute_combinations(file_write)

if resume:
    load_model()

# loading a pretrained model
if prune_bool:
    # thresh=[15,15,10,10,10,110,210,490,490,497,505,505,504,503,495]
    # thresh=[15,15,10,10,10,110,21,49,490,497,505,505,504,503,495]
    # thresh=[30,30,70,80,201,170,175,420,430,440,440,445,445,450,450]
    # thresh=[30,30,60,60,181,150,155,420,410,420,420,445,445,450,450]
    # thresh=[20,20,30,90,181,150,155,320,310,320,320,445,445,450,50]
    # thresh=[15,15,24,10,141,150,195,220,210,220,220,345,345,350,350]
    # thresh=args['arch']

    # thresh=[20, 20, 40, 40, 80, 80, 80, 160, 160, 160, 160, 160, 160, 160, 80]
    # thresh = [20, 20, 40, 40, 80, 80, 80, 160, 160, 160, 160, 80, 80, 80, 80]
    # thresh = [5, 5, 40, 40, 20, 40, 120, 230, 250, 300, 300, 160, 250, 250, 160]  # 10 #0.3
    # thresh=[5, 5, 40, 40, 20, 40, 80, 130, 190, 260, 260, 160, 250, 250, 160] #11 #0.4
    # thresh=[5, 5, 40, 40, 20, 40, 80, 80, 160, 40, 40, 160, 80, 160, 160] #12 #0.5 %17.81
    # thresh=[5, 5, 10, 10, 40, 20, 20, 40, 40, 160, 160, 40, 160, 80, 80] # 13
    # thresh=[5, 5, 20, 10, 20, 80, 40, 40, 40, 80, 160, 80, 80, 40, 80] #14 #0.6 #10.74 (94.34)
    # thresh = [5, 5, 10, 10, 20, 20, 20, 40, 40, 40, 40, 40, 80, 160, 80] #15
    thresh = [5, 5, 10, 10, 20, 20, 20, 40, 40, 40, 40, 40, 40, 40,
              160]  # 16 55.86. 69.81, 58.49, 57.11 (fish, filt, l1, l2) (94.34
    # thresh=[5, 5, 10, 10, 20, 10, 20, 20, 40, 20, 20, 40, 40, 20, 80] #17 # 87.86. 71.44, 58.49, 57.21 (94.34)
    # thresh=[5, 5, 10, 10, 10, 10, 10, 20, 20, 20, 10, 10, 10, 10, 10] #~18 #0.95

    # for i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15 in product([5,10,15], [5,10,15], [10,20,40], [10,20,40], [20,40,80], [20,40,80], [20,40,80], [40,80,160], [40,80,160], [40,80,160], [40,80,160], [40,80,160], [40,80,160], [40,80,160], [40,80,160]):
    # for i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15 in product([5, 10], [5, 10],
    #                                                                                 [10, 20], [10, 20],
    #                                                                                 [20, 40], [20, 40],
    #                                                                                 [20, 40], [40, 80],
    #                                                                                 [40, 80], [40, 80],
    #                                                                                 [40, 80], [40, 80],
    #                                                                                 [40, 80], [40, 80],
    #                                                                                 [40, 80]):
    # for i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15 in product([25], [25],
    #                                                                                 [60, 40], [60, 40],
    #                                                                                 [175], [140, 170],
    #                                                                                 [140], [430],
    #                                                                                 [380, 430], [380, 430],
    #                                                                                 [440], [440, 400],
    #                                                                                 [440], [450],
    #                                                                                 [450]):
    if 1:
        print('\n****************\n')
        for method in [args.method]:
            # for method in ['fisher']:
            print('\n\n' + method + "\n")
            thresh = [int(n) for n in args.arch.split(",")]
            #[i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15]
            print(thresh)
            prune_and_retrain(thresh)

    # prune_and_retrain(thresh) #first argument is whether to trune, False only retraining

# training from scratchhttps://www.onet.pl/
if resume == False:
    best_accuracy = 0
    session1end = start_epoch + 10;
    session2end = start_epoch + 250;
    session3end = start_epoch + 3250;  # was til 550
    for epoch in range(start_epoch, session1end):
        train_acc = train(epoch)
        test_acc = test(epoch)
        print(test_acc)
        best_accuracy = save_checkpoint(epoch, test_acc, best_accuracy)

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    for epoch in range(session1end, session2end):
        train_acc = train(epoch)
        test_acc = test(epoch)
        print(test_acc)
        best_accuracy = save_checkpoint(epoch, test_acc, best_accuracy)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    for epoch in range(session2end, session3end):
        train_acc = train(epoch)
        test_acc = test(epoch)
        best_accuracy = save_checkpoint(epoch, test_acc, best_accuracy)