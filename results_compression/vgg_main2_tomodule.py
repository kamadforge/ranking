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
sys.path.append("../results_switch")
sys.path.append("results_switch")
from script_vgg_vggswitch import switch_script


network_structure_dummy=0

#######
# PATH

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

print("Current working directory:", cwd)

sys.path.append(os.path.join(path_networktest, "external_codes/pytorch-cifar-master/models"))
sys.path.append(path_compression)

#TODO, specify path to the benchmark, download from here https://github.com/hendrycks/robustness


##############################
# PARAMETERS

parser = argparse.ArgumentParser()
# parser.add_argument("--arch", default='25,25,65,80,201,158,159,460,450,490,470,465,465,450')
# 0, parser.add_argument("--arch", default='5,5,5,5,6,6,6,6,7,7,7,7,8,8')
# 1, parser.add_argument("--arch", default='5,5,10,10,20,20,20,20,40,40,40,40,80,80')
# 2, parser.add_argument("--arch", default='16,32,32,64,96,128,128,192,256,256,256,256,256,256')
# 3, parser.add_argument("--arch", default='16,32,32,64,96,128,128,192,256,256,256,256,256,256')
# 4parser.add_argument("--arch", default='16,32,64,64,96,128,128,192,256,256,320,320,384,448')
parser.add_argument("--arch", default='5,5,5,5,6,6,6,6,7,7,7,7,8,8')
parser.add_argument('--layer', help="layer to prune", default="c1")
parser.add_argument("--method", default='l1')
parser.add_argument("--switch_samps", default=10, type=int)
parser.add_argument("--switch_epochs", default=1, type=int)
parser.add_argument("--ranks_method", default='point')
parser.add_argument("--switch_train", default=False)
parser.add_argument("--resume", default=False)
parser.add_argument("--prune_bool", default=False)
parser.add_argument("--retrain_bool", default=False)
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument("--run_name", default="")
parser.add_argument("--dataset", default="CIFAR-10")
parser.add_argument("--model", default="None")
parser.add_argument("--save_accuracy", default=50.0, type=float)
parser.add_argument("--scratch_training_numepochs", default=3250, type=int, help="at least 250")
parser.add_argument("--early_stopping", default=1000, type=int)
args = parser.parse_args()
print(args)

CORRUPTION_PATH = '/home/sebek/raid/datasets/cifar-c/{}-C/'.format(args.dataset)

if args.resume==False and args.prune_bool==False and args.retrain_bool==False:
    print("Training from scratch")
elif args.prune_bool==True:
    print("Pruning:", args.arch)
    #print(args.layer)

save_accuracy=args.save_accuracy



#######################################################################################################################################3
############################################################
# NETWORK


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG15': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512],
    # 'VGG15_comp': [39, 39, 63, 48, 55, 88, 87, 52, 62, 22, 42, 47, 47, 47],
    'VGG15_comp': [34, 34, 68, 68, 75, 106, 101, 92, 102, 92, 67, 67, 62, 62],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super(VGG, self).__init__()

        cfg_arch=cfg['VGG15']

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
        self.l3 = nn.Linear(cfg_arch[13], num_classes)
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

        output = output.view(-1, cfg['VGG15'][13])
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

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

if args.dataset =="CIFAR-10":
    trainval_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes)
    #not used anyway
else:
    trainval_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    num_classes = 100
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

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)



###################################################
# MAKE AN INSTANCE OF A NETWORK AND (POSSIBLY) LOAD THE MODEL


print('==> Building model..')
net = VGG('VGG16', 100)
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

def test_net_inner(net, test_loader):
    """Evaluate network on given dataset."""
    net.eval()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = net(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()
    return 100. * total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset), total_correct, len(test_loader.dataset)


def test_corruptions(net, test_data, base_path):
    """Evaluate network on given corrupted dataset."""
    corruption_accs = []
    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression']


    for corruption in CORRUPTIONS:
    # Reference to original data is mutated
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=512,
            shuffle=False,
            num_workers=8,
            pin_memory=True)

        test_loss, test_acc, _, _ = test_net_inner(net, test_loader)
        corruption_accs.append(test_acc)
        print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_loss, 100 - 100. * test_acc))
    return np.mean(corruption_accs)


def test(dataset, test_corr=False):
    # for name, param in net.named_parameters():
    #     print (name)
    #     print (param)
    global best_acc
    test_loss, acc, correct, total = test_net_inner(net, testloader)
    print('Test Lossds: %.3f | Acc: %.3f%% (%d/%d)' % (
        test_loss, 100. * acc, correct, total))
    
    if test_corr:
        corr_res = test_corruptions(net, testset, CORRUPTION_PATH)
        with open("results.txt", "a") as myfile:
            myfile.write("{} {}\n".format(acc, corr_res))
        print("acc, corr = ", acc, 100. * corr_res)
        
    return 100.0 * acc

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


def load_model(test_bool=True, test_corr=False):
    # Load checkpoint.
    # print('==> Resuming from checkpoint..')
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    print("LOADING model ", model2load)
    checkpoint = torch.load(model2load)
    # checkpoint = torch.load('./checkpoint/ckpt_vgg16_prunedto[39, 39, 63, 48, 55, 98, 97, 52, 62, 22, 42, 47, 47, 42, 62]_64.55.t7')
    net.load_state_dict(checkpoint['net'], strict=False)
    # print(net.module.c1.weight)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    if test_bool:
        print("Accuracy of the tested model: ")

        test(-1, test_corr)
    print("----")
    # testval(-1)
    # test_val(-1, net)
    if comp_combinations:
        compute_combinations(True, net, testval, args.layer)



######################################################
# SAVE experiment

def save_checkpoint(dest_dir, epoch, acc, best_acc, remaining=0):
    # Save checkpoint.
    # acc = test(epoch)
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        print("acc: ", acc)
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        if acc > save_accuracy:
            print('Saving..')
            if remaining == 0:  # regular training
                torch.save(state, path_compression+'/{}/ckpt_vgg16_{}.t7'.format(dest_dir, acc))
            else:
                torch.save(state, path_compression+'/{}/ckpt_vgg16_prunedto{}_{}.t7'.format(dest_dir, remaining, acc))
            torch.save(state, path_compression+'/{}/ckpt_vgg16_last.t7'.format(dest_dir))
        best_acc = acc

    return best_acc


##############################################
# PRUNEand RETRAIN

def prune_and_retrain(dest_dir, thresh):

    load_model(False)
    # PRUNE

    if prune_bool:
        ############################3
        # READ THE RANKS

        if method == 'switch':
            ranks_method=args.ranks_method

            if ranks_method == 'shapley':
                combinationss = []
                shapley_file = open(
                    "/home/kamil/Dropbox/Current_research/python_tests/results_shapley/combinations/94.34/zeroing_0.2val/shapley.txt")
                for line in shapley_file:
                    line = line.strip()[1:-2]
                    nums = line.split(",")
                    nums_int = [int(i) for i in nums]
                    combinationss.append(nums_int)

            elif ranks_method == 'integral':
                print("\nSwitch integral\n")
                if args.switch_train:
                    print("\nTraining switches\n")
                    switch_data = switch_script("switch_integral", args.switch_samps, args.switch_epochs, args.model)
                    combinationss=switch_data['combinations']
                else:
                    print("\nLoading switches\n")
                    ranks_path = path_switch+'/results/switch_data_cifar_integral_samps_%i_epochs_%i.npy' % (args.switch_samps, args.switch_epochs)
                    combinationss=list(np.load(ranks_path,  allow_pickle=True).item()['combinationss'])

            elif ranks_method == 'point':
                print("\nSwitch point\n")

                if args.switch_train:
                    print("\nTraining switches\n")
                    switch_data = switch_script("switch_point", args.switch_samps, args.switch_epochs, args.model)
                    combinationss=switch_data['combinationss']
                else:
                    print("\nLoading switches\n")
                    print(ranks_method)
                    ranks_path = path_switch+'/results/switch_data_cifar_point_epochs_%i.npy' % (args.switch_epochs)
                    combinationss=list(np.load(ranks_path,  allow_pickle=True).item()['combinationss'])

            # these numbers from the beginning will be cut off, meaning the worse will be cut off
            #we change to the other way around
            for i in range(len(combinationss)):
                combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:].copy())

        elif method == 'l1' or method == 'l2':
            #magnitude_rank.setup(model2load)
            #magnitude_rank.setup()
            combinationss = magnitude_rank.get_ranks(method, net)
            # the numbers from the beginning will be cut off, meaning the worse will be cut off
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

            # these numbers from the beginning will be cut off, meaning the worse will be cut off
            for i in range(len(combinationss)):
                combinationss[i] = torch.LongTensor(combinationss[i][:thresh[i]])
            print(combinationss[1])

        # PRINT THE PRUNED ARCHITECTURE
        remaining = []
        for i in range(len(combinationss)):
            print(cfg['VGG15'][i], len(combinationss[i]))
            remaining.append(int(cfg['VGG15'][i]) - len(combinationss[i]))
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

        ######## GRAD




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

        # here retraining works
        net.train()
        stop = 0;
        epoch = 0;
        best_accuracy = 0;
        entry = np.zeros(3);
        best_model = -1;
        early_stopping = args.early_stopping
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
                if accuracy > args.save_accuracy:
                    best_accuracy = save_checkpoint(dest_dir, epoch, accuracy, best_accuracy,
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
run_name = args.run_name
dest_dir = os.path.join('checkpoint', run_name)

os.makedirs(dest_dir, exist_ok=True)

if args.model=="None":
    model_name = 'ckpt_vgg16_last.t7'
    model2load = path_compression+'/checkpoint/{}/{}'.format(run_name, model_name)
    print("loading model: ", model2load)
else:
    model2load = args.model


orig_accuracy = 94.34
# if all False just train thenetwork
resume = args.resume
prune_bool = args.prune_bool
retrain_bool = args.retrain_bool  # whether we retrain the model or just evaluate

test_corr = resume and not prune_bool and not retrain_bool

comp_combinations = False  # must be with resume #with retrain if we want to retrain combinations
vis = False
# file_write=True
# compute_combinations(file_write)

if resume:
    load_model(True, test_corr)

# loading a pretrained model
if prune_bool:
    if 1:
        print('\n****************\n')
        for method in [args.method]:
            # for method in ['fisher']:
            print('\n\n' + method + "\n")
            thresh = [int(n) for n in args.arch.split(",")]
            print(thresh)
            prune_and_retrain(dest_dir, thresh)

    # prune_and_retrain(thresh) #first argument is whether to trune, False only retraining

# training from scratch
if resume == False and prune_bool == False and retrain_bool==False:
    best_accuracy = 0
    session1end = start_epoch + 10;
    session2end = start_epoch + 250;
#    session3end = start_epoch + 500; #SEBASTIAN, using 500 for experiments
    session3end = start_epoch + args.scratch_training_numepochs;  # was til 550
    for epoch in range(start_epoch, session1end):
        train_acc = train(epoch)
        test_acc = test(epoch)
        print(test_acc)
        best_accuracy = save_checkpoint(dest_dir, epoch, test_acc, best_accuracy)

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    for epoch in range(session1end, session2end):
        train_acc = train(epoch)
        test_acc = test(epoch)
        print(test_acc)
        best_accuracy = save_checkpoint(dest_dir, epoch, test_acc, best_accuracy)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    for epoch in range(session2end, session3end):
        train_acc = train(epoch)
        test_acc = test(epoch)
        best_accuracy = save_checkpoint(dest_dir, epoch, test_acc, best_accuracy)
