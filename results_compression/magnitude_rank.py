import torch
from torch import nn, optim

import torch
import torch.optim as optim
from torch import nn, optim
import torch.nn.functional as f

import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import csv
import pdb

from torch.nn.parameter import Parameter

#######################
# takes the network parameters from the conv layer and clusters them (with the purpose of removing some of them)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

trainval_perc=1
BATCH_SIZE = 100



def setup(network_arg='vgg', dataset_arg='cifar'):
    global dataset;
    dataset= dataset_arg
    global network;
    network= network_arg

setup()


if network =='vgg':

    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGGKAM': [0, 39, 39, 63, 48, 55, 98, 97, 52, 62, 22, 42, 47, 47, 42, 62],
        'VGGBC': [0, 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                  'M'],
    }
    model_structure = cfg['VGGBC']


    class VGG(nn.Module):
        def __init__(self, vgg_name):
            super(VGG, self).__init__()

            # self.features = self._make_layers(cfg[vgg_name])
            # self.classifier = nn.Linear(512, 10)
            # model_structure={'c1_num':39, 'c2_num'=39, 'c3_num'=63; 'c4_num'=48, 'c5_num'=55, 'c6_num'=98, 'c7_num'=97, 'c8_num'=52, 'c9_num'=62,
            # 'c10_num'=22, 'c11_num'=42, 'c12_num'=47 ; 'c13_num'=47 ; 'c14_num'=42 ; 'c15_num'=62}

            self.c1 = nn.Conv2d(3, model_structure[1], 3, padding=1)
            self.bn1 = nn.BatchNorm2d(model_structure[1], eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)
            self.c2 = nn.Conv2d(model_structure[1], model_structure[2], 3, padding=1)
            self.bn2 = nn.BatchNorm2d(model_structure[2], eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)
            self.mp1 = nn.MaxPool2d(2)

            self.c3 = nn.Conv2d(model_structure[2], model_structure[3], 3, padding=1)
            self.bn3 = nn.BatchNorm2d(model_structure[3], eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)
            self.c4 = nn.Conv2d(model_structure[3], model_structure[4], 3, padding=1)
            self.bn4 = nn.BatchNorm2d(model_structure[4], eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)
            self.mp2 = nn.MaxPool2d(2)

            self.c5 = nn.Conv2d(model_structure[4], model_structure[5], 3, padding=1)
            self.bn5 = nn.BatchNorm2d(model_structure[5], eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)
            self.c6 = nn.Conv2d(model_structure[5], model_structure[6], 3, padding=1)
            self.bn6 = nn.BatchNorm2d(model_structure[6], eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)
            self.c7 = nn.Conv2d(model_structure[6], model_structure[7], 3, padding=1)
            self.bn7 = nn.BatchNorm2d(model_structure[7], eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)
            self.mp3 = nn.MaxPool2d(2)

            self.c8 = nn.Conv2d(model_structure[7], model_structure[8], 3, padding=1)
            self.bn8 = nn.BatchNorm2d(model_structure[8], eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)
            self.c9 = nn.Conv2d(model_structure[8], model_structure[9], 3, padding=1)
            self.bn9 = nn.BatchNorm2d(model_structure[9], eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)
            self.c10 = nn.Conv2d(model_structure[9], model_structure[10], 3, padding=1)
            self.bn10 = nn.BatchNorm2d(model_structure[10], eps=1e-05, momentum=0.1, affine=True,
                                       track_running_stats=True)
            self.c11 = nn.Conv2d(model_structure[10], model_structure[11], 3, padding=1)
            self.bn11 = nn.BatchNorm2d(model_structure[11], eps=1e-05, momentum=0.1, affine=True,
                                       track_running_stats=True)
            self.mp4 = nn.MaxPool2d(2)

            self.c12 = nn.Conv2d(model_structure[11], model_structure[12], 3, padding=1)
            self.bn12 = nn.BatchNorm2d(model_structure[12], eps=1e-05, momentum=0.1, affine=True,
                                       track_running_stats=True)
            self.c13 = nn.Conv2d(model_structure[12], model_structure[13], 3, padding=1)
            self.bn13 = nn.BatchNorm2d(model_structure[13], eps=1e-05, momentum=0.1, affine=True,
                                       track_running_stats=True)
            self.c14 = nn.Conv2d(model_structure[13], model_structure[14], 3, padding=1)
            self.bn14 = nn.BatchNorm2d(model_structure[14], eps=1e-05, momentum=0.1, affine=True,
                                       track_running_stats=True)
            self.c15 = nn.Conv2d(model_structure[14], model_structure[15], 3, padding=1)
            self.bn15 = nn.BatchNorm2d(model_structure[15], eps=1e-05, momentum=0.1, affine=True,
                                       track_running_stats=True)
            self.mp5 = nn.MaxPool2d(2, stride=2, dilation=1, ceil_mode=False)
            self.ap = nn.AvgPool2d(1, stride=1)

            self.l3 = nn.Linear(model_structure[15], 10)
            self.d1 = nn.Dropout()
            self.d2 = nn.Dropout()

            self.parameter = Parameter(-1 * torch.ones(64), requires_grad=True)  # this parameter lies #S


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


elif network=='lenet':

    ####################################
    # NETWORK (conv-conv-fc-fc)


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
            self.output = nn.Linear(nodesFc2, 10)

            self.parameter = Parameter(-1e-10*torch.ones(nodesNum1),requires_grad=True) # this parameter lies #S



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
            output = self.bn1(output)
            output = self.c3(output)
            output = f.relu(self.s4(output))
            output = self.bn2(output)
            output = output.view(-1, self.nodesNum2 * 4 * 4)
            output = self.c5(output)
            output = self.f6(output)
            return output

if network=='vgg':
    net = VGG('VGG16')

    path="checkpoint/ckpt_93.92.t7"

    net=net.to(device)

    net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage), strict=False)

elif network=='lenet':
    nodesNum1, nodesNum2, nodesFc1, nodesFc2 = 10, 20, 100, 25
    net = Lenet(nodesNum1, nodesNum2, nodesFc1, nodesFc2)

    # path="models/mnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo540_acc99.27"
    path = "models/fashionmnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo62_acc90.04"


    net = net.to(device)


    net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'], strict=False)
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"










#####################################3


def get_ranks(method):
    combinationss=[]
    for name, param in net.named_parameters():

        if (("c" in name) or ("f" in name)) and ("weight" in name):
            #print(name)
            #print (name, param.shape)
            m = torch.flatten(param, start_dim=1)
            l2 = torch.norm(m, p=2, dim=1)
            l1= torch.norm(m, p=1, dim=1)
            #print(l1)
            #print(l2)

            # m = torch.flatten(param, start_dim=1)
            m = torch.abs(m)
            sum = m.sum(1)#
            #print(sum) #SAME AS L1

            # l1rank=torch.argsort(l1, descending=True)
            # l2rank=torch.argsort(l2, descending=True)

            l1 = l1.detach().cpu().numpy()
            l2 = l2.detach().cpu().numpy()

            #for lenet change the irder
            # l1rank = np.argsort(l1)[::-1]
            # l2rank = np.argsort(l2)[::-1]

            l1rank = np.argsort(l1)
            l2rank = np.argsort(l2)


            #print(l1rank)

            if method=='l1':
                #print(l1rank)
                combinationss.append(l1rank)
            elif method=='l2':
                combinationss.append(l2rank)

            # if method == 'l1':
            #     combinationss.append(torch.LongTensor(l1rank))
            # elif method == 'l2':
            #     combinationss.append(torch.LongTensor(l2rank))

    return combinationss
            #sum=np.sum(param[0, :].sum()
            #print(sum)

if __name__=='__main__':
    print(get_ranks('l1'))