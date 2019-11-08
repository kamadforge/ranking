


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
#sys.path.append("/home/kamil/Dropbox/Current_research/python_tests/results_networktest/external_codes/pytorch-cifar-master")
sys.path.append("/home/kamil/Dropbox/Current_research/python_tests/results_networktest/external_codes/pytorch-cifar-master/models")
#sys.path.append("/home/kamil/Dropbox/Current_research/python_tests/results_networktest/external_codes/pytorch-cifar-master/utils.py")
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as f




#file_dir = os.path.dirname("utlis.p")
#sys.path.append(file_dir)

#from models import *

#from utils import progress_bar


##########################################
resume=True


'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

############################################################
# NETWORK


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
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


        self.parameter_switch = Parameter(-1*torch.ones(128),requires_grad=True) # this parameter lies #S


    def forward(self, x):
        phi = f.softplus(self.parameter_switch)
        S = phi / torch.sum(phi)
        # Smax = torch.max(S)
        # Sprime = S/Smax
        Sprime = S

        output = self.c1(x)
        output = f.relu(self.bn1(output))
        output = self.c2(output)



        output = f.relu(self.bn2(output))
        output = self.mp1(output)

        output = self.c3(output)
        for i in range(len(Sprime)):
            output[:, i] *= Sprime[i].expand_as(output[:, i])

        output = f.relu(self.bn3(output))


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

        #output = f.relu(self.l1(output))
        #output = self.d2(output)
        #output = f.relu(self.l2(output))

        #out = self.features(x)
        #out = out.view(out.size(0), -1)
        #out = self.classifier(out)
        return output, Sprime

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




# def test():
#     net = VGG('VGG11')
#     x = torch.randn(2,3,32,32)
#     y = net(x)
#     print(y.size())

# test()

#####################################
# DATA

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#############################
# PARAMS

how_many_epochs=200
annealing_steps = float(8000. * how_many_epochs)
beta_func = lambda s: min(s, annealing_steps) / annealing_steps
alpha_0 = 0.01  # below 1 so that we encourage sparsity
hidden_dim = 64 #it's a number of parameters we want to estimate, e.g. # conv1 filters
BATCH_SIZE=100

###################################################
# MAKE AN INSTANCE OD NETWORK AND (POSSIBLY) LOAD THE MODEL

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

for name, param in net.named_parameters():
    print (name, param.shape)


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    print(device)

#if args.resume:
if (resume):
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_93.92.t7', map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net'], strict=False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


#######################s
# LOSS



def loss_function(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps):
    # BCE = f.binary_cross_entropy(prediction, true_y, reduction='sum')
    BCE = criterion(prediction, true_y)

    return BCE


def loss_functionKL(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps, annealing_rate):
    # BCE = F.binary_cross_entropy(prediction, true_y, reduction='mean')
    BCE = criterion(prediction, true_y)

    # KLD term
    alpha_0 = torch.Tensor([alpha_0]).to(device)
    hidden_dim = torch.Tensor([hidden_dim]).to(device)
    trm1 = torch.lgamma(torch.sum(S)) - torch.lgamma(hidden_dim * alpha_0)
    trm2 = - torch.sum(torch.lgamma(S)) + hidden_dim * torch.lgamma(alpha_0)
    trm3 = torch.sum((S - alpha_0) * (torch.digamma(S) - torch.digamma(torch.sum(S))))
    KLD = trm1 + trm2 + trm3
    # annealing kl-divergence term is better

    return BCE + annealing_rate * KLD / how_many_samps

########################################################
# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    annealing_rate = beta_func(epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, S = net(inputs)
        #loss = criterion(outputs, targets)
        loss = loss_functionKL(outputs, targets, S, alpha_0, hidden_dim, BATCH_SIZE, annealing_rate)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_idx % 1000 ==0):
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print("S")
    print(S)

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
            outputs = net(inputs)[0] #[0] added because of the tuple output,S
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return 100.*correct/total


    # Save checkpoint.
    acc = 100.*correct/total
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

###########################################################
#copied from network pruning

def compute_combinations_random(file_write):
    for name, param in net.named_parameters():
        print(name)
        print(param.shape)
        layer = "module.features.1.weight"
        if layer in name:
            layerbias = layer[:17] + ".bias"
            params_bias = net.state_dict()[layerbias]
            while (True):

                all_results = {}
                # s=torch.range(0,49) #list from 0 to 19 as these are the indices of the data tensor
                # for r in range(1,50): #produces the combinations of the elements in s
                #    results=[]
                randperm = np.random.permutation(param.shape[0])
                randint = 0
                while (randint == 0):
                    randint = np.random.randint(param.shape[0])
                randint_indextoremove = np.random.randint(randint)
                combination = randperm[:randint]
                combination2 = np.delete(combination, randint_indextoremove)
                print(combination[randint_indextoremove])

                #if file_write:
                print("in")
                with open("results_running/combinations_pruning_cifar_vgg_%s.txt" % (layer),
                          "a+") as textfile:
                    textfile.write("%d\n" % randint_indextoremove)

                for combination in [combination, combination2]:
                    # for combination in list(combinations(s, r)):

                    combination = torch.LongTensor(combination)

                    print(combination)
                    params_saved = param[combination].clone()
                    param_bias_saved = params_bias[combination].clone()

                    # param[torch.LongTensor([1, 4])] = 0
                    # workaround, first using multiple indices does not work, but if one of the change first then it works to use  param[combinations]
                    if len(combination) != 0:
                        param[combination[0]] = 0
                        # param[combination]=0
                        params_bias[combination] = 0

                    accuracy = test(-1)
                    param[combination] = params_saved
                    params_bias[combination] = param_bias_saved

                    #if file_write:
                    print("out")
                    with open("results_running/combinations_pruning_cifar_vgg_%s.txt" % (layer),
                              "a+") as textfile:
                        textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))


##########################################################################3
# RUN EXPERIMENT

file_write=True
compute_combinations_random(file_write)

training=True


for name, param in net.named_parameters():
    #print(name, param.shape)
    if (name != "module.parameter"):
        h = param.register_hook(lambda grad: grad * 0)  # double the gradient


if training:
    session1end=start_epoch+1; session2end=start_epoch+250; session3end=start_epoch+350;
    for epoch in range(start_epoch, session1end):
        train(epoch)
        test(epoch)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    for epoch in range(session1end+1, session2end):
        train(epoch)
        test(epoch)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    for epoch in range(session2end+1, session3end):
        train(epoch)
        test(epoch)

