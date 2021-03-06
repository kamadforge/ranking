# 1. It trains the network from scratch (if "resume" is off)
# 2. if resume is on, it loads the pretrained model,
#     if prune_ bool , it prunes the network
#     if retrain_ bool is on , retrains it
    #e.g. it can retrain, but not prune
# 3. it can be used for visualizing, if uncomment the comments #VISU

# other features:
# it loads the ranks from shapley or switches (ranks_path = '../Dir_switch/results/cifar/vgg_93.92/switch_init_-1, alpha_2/')


#
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

'''Train CIFAR10 with PyTorch.'''
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
print (sys.path)
print("newh2")
sys.path.append("/home/kamil/Dropbox/Current_research/python_tests/results_networktest/external_codes/pytorch-cifar-master/models")
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as f
import logging
import matplotlib.pyplot as plt
import magnitude_rank


#file_dir = os.path.dirname("utlis.p")
#sys.path.append(file_dir)

#from models import *

#from utils import progress_bar

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


############################################################
# NETWORK


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGGBC': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()

        #self.features = self._make_layers(cfg[vgg_name])
        #self.classifier = nn.Linear(512, 10)

        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp1 = nn.MaxPool2d(2)

        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp2 = nn.MaxPool2d(2)

        self.c5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp3 = nn.MaxPool2d(2)

        self.c8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp4 = nn.MaxPool2d(2)

        self.c12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c14 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn14 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.c15 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn15 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mp5 = nn.MaxPool2d(2, stride=2, dilation=1, ceil_mode=False)
        self.ap = nn.AvgPool2d(1, stride=1)

        # self.l1 = nn.Linear(512, 512)
        # self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 10)
        self.d1 = nn.Dropout()
        self.d2 = nn.Dropout()

        self.parameter = Parameter(-1 * torch.ones(64), requires_grad=True)  # this parameter lies #S

    #def forward(self, x, i):  # VISU
    def forward(self, x):
        phi = f.softplus(self.parameter)
        S = phi / torch.sum(phi)
        # Smax = torch.max(S)
        # Sprime = S/Smax
        Sprime = S

        output = f.relu(self.bn1(self.c1(x)))
        output = f.relu(self.bn2(self.c2(output)))
        output = self.mp1(output)

        output = f.relu(self.bn3(self.c3(output)))
        output = f.relu(self.bn4(self.c4(output)))
        output = self.mp2(output)

        output = f.relu(self.bn5(self.c5(output)))
        output = f.relu(self.bn6(self.c6(output)))
        output = f.relu(self.bn7(self.c7(output)))
        output = self.mp3(output)

        output = f.relu(self.bn8(self.c8(output)))
        output = f.relu(self.bn9(self.c9(output)))
        output = f.relu(self.bn10(self.c10(output)))
        output = f.relu(self.bn11(self.c11(output)))
        output = self.mp4(output)
        output = f.relu(self.bn12(self.c12(output)))
        output = f.relu(self.bn13(self.c13(output)))
        output = f.relu(self.bn14(self.c14(output)))
        output = f.relu(self.bn15(self.c15(output)))
        output = self.mp5(output)
        output = self.ap(output)

        output = output.view(-1, 512)
        output = self.l3(output)

        # output = f.relu(self.l1(output))
        # output = self.d2(output)
        # output = f.relu(self.l2(output))

        # out = self.features(x)
        # out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        return output

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


# ############################################################
# # NETWORK
#
#
# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGGKAM': [0, 39, 39, 63, 48, 55, 98, 97, 52, 62, 22, 42, 47, 47, 42, 62],
#     'VGGBC': [0, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
# model_structure = cfg['VGGBC']
#
#
# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#
#         #self.features = self._make_layers(cfg[vgg_name])
#         #self.classifier = nn.Linear(512, 10)
#         #model_structure={'c1_num':39, 'c2_num'=39, 'c3_num'=63; 'c4_num'=48, 'c5_num'=55, 'c6_num'=98, 'c7_num'=97, 'c8_num'=52, 'c9_num'=62,
#         #'c10_num'=22, 'c11_num'=42, 'c12_num'=47 ; 'c13_num'=47 ; 'c14_num'=42 ; 'c15_num'=62}
#
#         self.c1 = nn.Conv2d(3, model_structure[1], 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(model_structure[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.c2 = nn.Conv2d(model_structure[1], model_structure[2], 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(model_structure[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.mp1 = nn.MaxPool2d(2)
#
#         self.c3 = nn.Conv2d(model_structure[2], model_structure[3], 3, padding=1)
#         self.bn3 = nn.BatchNorm2d(model_structure[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.c4 = nn.Conv2d(model_structure[3], model_structure[4], 3, padding=1)
#         self.bn4 = nn.BatchNorm2d(model_structure[4], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.mp2 = nn.MaxPool2d(2)
#
#         self.c5 = nn.Conv2d(model_structure[4], model_structure[5], 3, padding=1)
#         self.bn5 = nn.BatchNorm2d(model_structure[5], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.c6 = nn.Conv2d(model_structure[5], model_structure[6], 3, padding=1)
#         self.bn6 = nn.BatchNorm2d(model_structure[6], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.c7 = nn.Conv2d(model_structure[6], model_structure[7], 3, padding=1)
#         self.bn7 = nn.BatchNorm2d(model_structure[7], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.mp3 = nn.MaxPool2d(2)
#
#         self.c8 = nn.Conv2d(model_structure[7], model_structure[8], 3, padding=1)
#         self.bn8 = nn.BatchNorm2d(model_structure[8], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.c9 = nn.Conv2d(model_structure[8], model_structure[9], 3, padding=1)
#         self.bn9 = nn.BatchNorm2d(model_structure[9], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.c10 = nn.Conv2d(model_structure[9], model_structure[10], 3, padding=1)
#         self.bn10 = nn.BatchNorm2d(model_structure[10], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.c11 = nn.Conv2d(model_structure[10], model_structure[11], 3, padding=1)
#         self.bn11 = nn.BatchNorm2d(model_structure[11], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.mp4 = nn.MaxPool2d(2)
#
#         self.c12 = nn.Conv2d(model_structure[11], model_structure[12], 3, padding=1)
#         self.bn12 = nn.BatchNorm2d(model_structure[12], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.c13 = nn.Conv2d(model_structure[12], model_structure[13], 3, padding=1)
#         self.bn13 = nn.BatchNorm2d(model_structure[13], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.c14 = nn.Conv2d(model_structure[13], model_structure[14], 3, padding=1)
#         self.bn14 = nn.BatchNorm2d(model_structure[14], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.c15 = nn.Conv2d(model_structure[14], model_structure[15], 3, padding=1)
#         self.bn15 = nn.BatchNorm2d(model_structure[15], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         self.mp5 = nn.MaxPool2d(2, stride=2, dilation=1, ceil_mode=False)
#         self.ap = nn.AvgPool2d(1, stride=1)
#
#         self.l3 = nn.Linear(model_structure[15], 10)
#         self.d1 = nn.Dropout()
#         self.d2 = nn.Dropout()
#
#         self.parameter = Parameter(-1 * torch.ones(64), requires_grad=True)  # this parameter lies #S
#
#     def forward(self, x, i): #VISU
#     #def forward(self, x):
#         # phi = f.softplus(self.parameter)
#         # S = phi / torch.sum(phi)
#         # # Smax = torch.max(S)
#         # # Sprime = S/Smax
#         # Sprime = S
#         #
#         output = f.relu(self.bn1(self.c1(x)))
#
#         #
#         #we visualize (and prune) output channels, that is we take as input of the three input channels (in case of RGB)
#         #and see how each of the 64 channels transform this feature map into a new feature map
#         # hence we visualize 64 feature maps
#
#         #VISU
#         #if vis:
#         # for filter_num in range(64):
#         #     mm=output.cpu().detach().numpy()
#         #     #fig,ax = plt.subplots(1)
#         #     matrix=mm[1,filter_num,:,:]
#         #     ave = np.average(matrix[0:20, 0])
#         #     matrix = matrix - ave
#         #
#         #     #ax.imshow(mm[1,filter_num,:,:], cmap="gray", aspect='normal')
#         #     plt.imshow(matrix, cmap="coolwarm") #showing 2nd channel (example of a channel)
#         #
#         #
#         #     plt.gca().set_axis_off()
#         #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
#         #                     hspace=0, wspace=0)
#         #     plt.margins(0, 0)
#         #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#         #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#         #     epoch=-1
#         #     plt.savefig("/home/kamil/Dropbox/Current_research/python_tests/results_networktest/vis/feature_maps/cifar/trial_coolwarm/conv1_batch%d_filternum%d_epoch%d" % (i, filter_num, epoch), bbox_inches='tight', pad_inches=0)
#         #
#
#
#         output = f.relu(self.bn1(output))
#
#
#         output = f.relu(self.bn2(self.c2(output)))
#         output = self.mp1(output)
#         output = f.relu(self.bn3(self.c3(output)))
#         output = f.relu(self.bn4(self.c4(output)))
#         output = self.mp2(output)
#         output = f.relu(self.bn5(self.c5(output)))
#         output = f.relu(self.bn6(self.c6(output)))
#         output = f.relu(self.bn7(self.c7(output)))
#         output = self.mp3(output)
#         output = f.relu(self.bn8(self.c8(output)))
#         output = f.relu(self.bn9(self.c9(output)))
#         output = f.relu(self.bn10(self.c10(output)))
#         output = f.relu(self.bn11(self.c11(output)))
#         output = self.mp4(output)
#         output = f.relu(self.bn12(self.c12(output)))
#         output = f.relu(self.bn13(self.c13(output)))
#         output = f.relu(self.bn14(self.c14(output)))
#         output = f.relu(self.bn15(self.c15(output)))
#         output = self.mp5(output)
#         output = self.ap(output)
#         output = output.view(-1, model_structure[15])
#         output = self.l3(output)
#         return output
#
#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)


#####################################
# DATA

# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
#with more workers there may be an error in debug mode: RuntimeError: DataLoader worker (pid 29274) is killed by signal: Terminated.

#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
#testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

###################################################
# MAKE AN INSTANCE OF A NETWORK AND (POSSIBLY) LOAD THE MODEL

# Model
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
#net = ShuffleNetV2(1)
net = net.to(device)

# for name, param in net.named_parameters():
#     print (name, param.shape)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    #print(device)

########################################################
# TRAIN


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        #outputs = net(inputs, batch_idx) #VISU
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #if (batch_idx % 1000 ==0):
    print('Training Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100.*correct/total, best_acc




#################################################################
# TEST

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            #outputs = net(inputs, batch_idx) #VISU
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Test Lossds: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return 100*correct/total



    # global best_acc
    # net.eval()
    # test_loss = 0
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(testloader):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = net(inputs, batch_idx) #VISU
    #         #outputs = net(inputs)
    #         loss = criterion(outputs, targets)
    #
    #         test_loss += loss.item()
    #         _, predicted = outputs.max(1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()
    #         #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #         #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    #
    # return 100*correct/total




########################################
# just RESUME

#if args.resume:
def load_model(test_bool=True):
    # Load checkpoint.
    #print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_93.92.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    if test_bool:
        print("Accuracy of the tested model: ")

        test(-1)
    print("----")

###############################################
######################################################
# RUN EXPERIMENT

def save_checkpoint(epoch, acc, best_acc):

#Save checkpoint.
    #acc = test(epoch)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_%.2f.t7' % acc)
        best_acc = acc

    return best_acc




##############################################
# PRUNEand RETRAIN

def prune_and_retrain(thresh):

    load_model(False)

    #PRINT
    # for name, param in net.named_parameters():
    #    print (name)
    #    print(param.shape)

    #from worse to best
#
#     ]

    if prune_bool:
        ############################3
        # READ THE RANKS

        if method=='filter':
            ranks_method='switches'
            switches_epoch=10



            if ranks_method=='shapley':
                combinationss=torch.load('results/ranks/ranks_93.92_shapley.pt')
            # elif ranks_method=='switches':
            #     #combinationss=torch.load('results/ranks/ranks_93.92_switches.pt')
            #     combinationss=[0]*15
            #     ranks_path = '../Dir_switch/results/cifar/vgg_93.92/switch_init_0.05, alpha_0.05, annealing_6000000/'
            #     #ranks_path = '../Dir_switch/results/cifar/vgg_93.92/switch_init_-1, alpha_2/'
            #     for i in range(len(combinationss)):
            #         ranks_filepath = ranks_path + "93.92_alpha0.05_switchinit0.05_conv" + str(i + 1) + "_ep"+str(switches_epoch)+".pt"

            elif ranks_method == 'switches':
                # combinationss=torch.load('results/ranks/ranks_93.92_switches.pt')
                combinationss = [0] * len(cfg['VGGBC']) #15
                ranks_path = '../Dir_switch/results/cifar/vgg_93.92/switch_init_-1, alpha_2/'
                for i in range(len(combinationss)):
                    ranks_filepath = ranks_path + "93.92_conv" + str(i + 1) + "_ep49.pt"

                    switch_values = torch.load(ranks_filepath)
                    #print(switch_values)
                    #combinationss[i]=torch.argsort(switch_values)
                    combinationss[i]=torch.LongTensor(np.argsort(switch_values.cpu().detach().numpy())[::-1].copy())#argsort is increasing order, we want decreasing hence [::-1]
                    #print(combinationss[i])
                    #print("new")


            #these numbers from the beginning will be cut off, meaning the worse will be cut off
            for i in range(len(combinationss)):
                combinationss[i] = torch.LongTensor(combinationss[i][:thresh[i]])
            print(combinationss[1])


        elif method=='l1' or method=='l2':
            combinationss = magnitude_rank.get_ranks(method)
            # for i in range(4):
            #     combinationss.append(torch.LongTensor(combinat[i]))

            # these numbers from the end will be cut off, meaning the worse will be cut off
            for i in range(len(combinationss)):
                combinationss[i] = torch.LongTensor(combinationss[i][:thresh[i]].copy())
            print(combinationss[1])




        # PRINT THE PRUNED ARCHITECTURE
        remaining=[]
        for i in range(len(combinationss)):
            print(cfg['VGGBC'][i], len(combinationss[i]))
            remaining.append(int(cfg['VGGBC'][i])-len(combinationss[i]))
        print(remaining)


        # PRUNE

        it=0
        for name, param in net.named_parameters():
            print(name)
            if "module.c" in name and "weight" in name:
                it+=1
                param.data[combinationss[it-1]]=0
                #print(param.data)
            if "module.c" in name and "bias" in name:
                param.data[combinationss[it - 1]] = 0
                #print(param.data)
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
        h1 = net.module.c1.weight.register_hook(gradi1)
        h1 = net.module.c1.bias.register_hook(gradi1)
        h12 = net.module.bn1.weight.register_hook(gradi1)
        h13 = net.module.bn1.bias.register_hook(gradi1)


        def gradi2(module):
            module[combinationss[1]] = 0
            # print(module[21])
        h1 = net.module.c2.weight.register_hook(gradi2)
        h1 = net.module.c2.bias.register_hook(gradi2)
        h12 = net.module.bn2.weight.register_hook(gradi2)
        h13 = net.module.bn2.bias.register_hook(gradi2)


        def gradi3(module):
            module[combinationss[2]] = 0
            # print(module[21])
        h1 = net.module.c3.weight.register_hook(gradi3)
        h1 = net.module.c3.bias.register_hook(gradi3)
        h12 = net.module.bn3.weight.register_hook(gradi3)
        h13 = net.module.bn3.bias.register_hook(gradi3)


        def gradi4(module):
            module[combinationss[3]] = 0
            # print(module[21])
        h1 = net.module.c4.weight.register_hook(gradi4)
        h1 = net.module.c4.bias.register_hook(gradi4)
        h12 = net.module.bn4.weight.register_hook(gradi4)
        h13 = net.module.bn4.bias.register_hook(gradi4)


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
            # print(module[21])
        h1 = net.module.c14.weight.register_hook(gradi14)
        h1 = net.module.c14.bias.register_hook(gradi14)
        h12 = net.module.bn14.weight.register_hook(gradi14)
        h13 = net.module.bn14.bias.register_hook(gradi14)

        def gradi15(module):
            module[combinationss[14]] = 0
            # print(module[21])
        h1 = net.module.c15.weight.register_hook(gradi15)
        h1 = net.module.c15.bias.register_hook(gradi15)
        h12 = net.module.bn15.weight.register_hook(gradi15)
        h13 = net.module.bn15.bias.register_hook(gradi15)



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


    if retrain_bool:
        print("Retraining")

        filename = "retrained_paramsearch1_vgg.txt"
        with open(filename, "a+") as file:
            file.write("---NEW EXPERIMENT-----")
            if prune_bool:
                file.write("\n\nprunedto:%s\n\n" % (" ".join(str(e) for e in remaining)))

        path="./checkpoint/"

        #here retraining works
        net.train()
        stop = 0; epoch = 0; best_accuracy = 0; entry = np.zeros(3); best_model = -1; early_stopping=350
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
                net.module.c2.weight.grad #the hook is automatically applied, here we just check the gradient
                optimizer.step()
                #net.c1.weight.data[1] = 0  # instead of hook
                #net.c1.bias.data[1] = 0  # instead of hook
                # if i % 100==0:
                #    print (i)
                #   print (loss.item())

            print(loss.item())
            accuracy = test(-1)
            #print(net.module.c2.weight.data)
            print("Epoch " + str(epoch) + " ended.")

            #check if pruned weights are pruned
            # for name, param in net.named_parameters():
            #     if "1.weight" in name:
            #         print(name)
            #         print(param)

            if (accuracy <= best_accuracy):
                stop = stop + 1
                entry[2] = 0
            else:
                best_accuracy = accuracy
                print("Best updated")
                stop = 0
                entry[2] = 1
                best_model = net.state_dict()
                if best_accuracy>13.5:
                    if prune_bool:
                        torch.save(best_model, "{}_retrained_epo-{}_prunedto-{}_acc-{:.2f}".format(path, epoch, remaining, best_accuracy))
                    else:
                        torch.save(best_model, "%s_retrained_epo-%d_only_acc-%.2f" % (path, epoch, best_accuracy))

                entry[0] = accuracy;
                entry[1] = loss
                with open(filename, "a+") as file:
                    file.write("\n Epoch: %d\n" % epoch)
                    file.write(",".join(map(str, entry)) + "\n")
                    if (accuracy>98.9):
                        file.write("Yes\n")
                    elif (accuracy>98.8):
                        file.write("Ok\n")

        print(loss.item())
        accuracy = test(-1)

#################################################################

#if all False just train thenetwork
resume = True
prune_bool = True
retrain_bool = True  # whether we retrain the model or just evaluate


#file_write=True
#compute_combinations(file_write)

if resume:
    load_model()

#loading a pretrained model
if prune_bool:
    #thresh=[15,15,10,10,10,110,210,490,490,497,505,505,504,503,495]
    #thresh=[15,15,10,10,10,110,21,49,490,497,505,505,504,503,495]
    #thresh=[30,30,70,80,201,170,175,420,430,440,440,445,445,450,450]
    #thresh=[30,30,60,60,181,150,155,420,410,420,420,445,445,450,450]
    #thresh=[20,20,30,90,181,150,155,320,310,320,320,445,445,450,50]
    #thresh=[15,15,24,10,141,150,195,220,210,220,220,345,345,350,350]
    thresh=[25,25,65,80,201,158,159,460,450,490,470,465,465,470,450]

    thresh=[20, 20, 40, 40, 80, 80, 80, 160, 160, 160, 160, 160, 160, 160, 80]
    thresh = [20, 20, 40, 40, 80, 80, 80, 160, 160, 160, 160, 80, 80, 80, 80]
    #thresh=[5, 5, 40, 40, 20, 40, 80, 80, 160, 40, 40, 160, 80, 160, 160] #12
    #thresh=[5, 5, 10, 10, 40, 20, 20, 40, 40, 160, 160, 40, 160, 80, 80] # 13
    #thresh=[5, 5, 20, 10, 20, 80, 40, 40, 40, 80, 160, 80, 80, 40, 80] #14
    #thresh = [5, 5, 10, 10, 20, 20, 20, 40, 40, 40, 40, 40, 80, 160, 80] #15
    #thresh=[5, 5, 10, 10, 20, 20, 20, 40, 40, 40, 40, 40, 40, 40, 160] #16
    #thresh=[5, 5, 10, 10, 20, 10, 20, 20, 40, 20, 20, 40, 40, 20, 80] #17
    #thresh=[5, 5, 10, 10, 10, 10, 10, 20, 20, 20, 10, 10, 10, 10, 10] #~18




    print('\n****************\n')
    for method in ['filter', 'l1', 'l2']:
        print('\n\n'+method+"\n")
        #thresh=[i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15]
        print(thresh)
        prune_and_retrain(thresh)



    prune_and_retrain(thresh) #first argument is whether to trune, False only retraining

# training from scratch
if resume==False:
    best_accuracy=0
    session1end=start_epoch+10; session2end=start_epoch+250; session3end=start_epoch+1950; #was til 550
    for epoch in range(start_epoch, session1end):
        train_acc=train(epoch)
        test_acc=test(epoch)
        best_accuracy=save_checkpoint(epoch, test_acc, best_accuracy)

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    for epoch in range(session1end, session2end):
        train_acc = train(epoch)
        test_acc = test(epoch)
        best_accuracy=save_checkpoint(epoch, test_acc, best_accuracy)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    for epoch in range(session2end, session3end):
        train_acc = train(epoch)
        test_acc = test(epoch)
        best_accuracy=save_checkpoint(epoch, test_acc, best_accuracy)

