from copy import deepcopy
import torch
import torch.nn as nn
from models.layer import ArmConv2d
from models.layer import ARMDense
import numpy as np
from config import opt
import torch.nn.functional as f


def get_flat_fts(self, in_size, fts):
    dummy_input = torch.ones(1, *in_size)
    #if torch.cuda.is_available():
    if opt.use_gpu and torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    #f = fts(torch.autograd.Variable(dummy_input))
    x=torch.autograd.Variable(dummy_input)
    o = self.conv1(x, 0, True)
    o = f.relu(self.mp1(o))
    o = self.conv2(o, 0, False)
    fout = f.relu(self.mp2(o))

    return int(np.prod(fout.size()[1:]))


class ARMLeNet5(nn.Module):
    #def __init__(self, num_classes=10, input_size=(1, 28, 28), conv_dims=(20, 50), fc_dims=500,
    def __init__(self, num_classes=10, input_size=(1, 28, 28), conv_dims=(10, 20), fc_dims=100,

                 N=60000, beta_ema=0.999, weight_decay=0.0005, lambas=(.1, .1, .1, .1)):
        super(ARMLeNet5, self).__init__()
        self.N = N
        assert (len(conv_dims) == 2)
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.beta_ema = beta_ema
        self.weight_decay = weight_decay

        self.conv1=ArmConv2d(input_size[0], conv_dims[0], 5, droprate_init=opt.lenet_dr, lamba=lambas[0], local_rep=opt.local_rep,
                           weight_decay=self.weight_decay)
        self.mp1=nn.MaxPool2d(2)
        self.conv2=ArmConv2d(conv_dims[0], conv_dims[1], 5, droprate_init=opt.lenet_dr, lamba=lambas[1],
                           local_rep=opt.local_rep,
                           weight_decay=self.weight_decay)
        self.mp2=nn.MaxPool2d(2)

        convs = [ArmConv2d(input_size[0], conv_dims[0], 5, droprate_init=opt.lenet_dr, lamba=lambas[0],
                           local_rep=opt.local_rep,
                           weight_decay=self.weight_decay),
                 nn.ReLU(), nn.MaxPool2d(2),
                 ArmConv2d(conv_dims[0], conv_dims[1], 5, droprate_init=opt.lenet_dr, lamba=lambas[1],
                           local_rep=opt.local_rep,
                           weight_decay=self.weight_decay),
                 nn.ReLU(), nn.MaxPool2d(2)]

        #self.convs = nn.Sequential(*convs)

        if opt.use_gpu and torch.cuda.is_available():
            #self.convs = self.convs.cuda()
            self.conv1=self.conv1.cuda()
            self.mp1 = self.mp1.cuda()
            self.conv2 = self.conv2.cuda()
            self.mp2 = self.mp2.cuda()


        flat_fts = get_flat_fts(self, input_size, 0) #0 is dummy
        fcs = [ARMDense(flat_fts, self.fc_dims, droprate_init=opt.lenet_dr, lamba=lambas[2], local_rep=opt.local_rep,
                        weight_decay=self.weight_decay), nn.ReLU(),
               #ARMDense(self.fc_dims, num_classes, droprate_init=opt.lenet_dr, lamba=lambas[3], local_rep=opt.local_rep,
               ARMDense(self.fc_dims, 25, droprate_init=opt.lenet_dr, lamba=lambas[3],
                                 local_rep=opt.local_rep,

                                 weight_decay=self.weight_decay)]
        self.fcs = nn.Sequential(*fcs)
        self.layers = []
        for m in self.modules():
            if isinstance(m, ARMDense) or isinstance(m, ArmConv2d):
                self.layers.append(m)

        if beta_ema > 0.:
            print('Using temporal averaging with beta: {}'.format(beta_ema))
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if opt.use_gpu and torch.cuda.is_available():
                self.avg_param = [a.cuda() for a in self.avg_param]
            self.steps_ema = 0.

    def update_phi_gradient(self, f1, f2):
        for layer in self.layers:
            layer.update_phi_gradient(f1, f2)
            layer.zero_gradients() #myfunc

    def forward_mode(self, mode):
        for layer in self.layers:
            layer.forward_mode = mode

    def score(self, x, neuron_bool):
        o= self.conv1(x, 0, neuron_bool)
        o=f.relu(self.mp1(o))
        o = self.conv2(o, 0, False)
        o = f.relu(self.mp2(o))
        #o = self.convs(x)
        o = o.view(o.size(0), -1)
        o = self.fcs(o)
        return o

    def forward(self, x, y=None, node=0):
        if self.training:
            self.forward_mode(True)

            self.conv2.z_phi[node] = torch.tensor(10000)
            score = self.score(x, True) #with neuron #should be smaller loss now
            f2 = nn.CrossEntropyLoss()(score, y).data #with neuron #should be smaller loss now


            self.eval() if opt.gpus <= 1 else self.module.eval()
            if opt.ar is not True:
                self.forward_mode(False)
                score2 = self.score(x).data
                f1 = nn.CrossEntropyLoss()(score2, y).data #with neuron #should be smaller loss now
            else:
                f1 = 0

            #self.conv1.z_phi[0].detach()
            self.conv2.z_phi[node]=torch.tensor(-10000)
            score3 = self.score(x, False).data
            f3 = nn.CrossEntropyLoss()(score3, y).data #removed neuron #should be bigger loss now
            #self.conv1.z_phi[0]=temp


            diff = f3 - f2  # should be positive because f3 loss is bigger since there is no one neuron
            #print("f3 ", f3)
            #print("f2 ", f2)
            #print(diff)  # minimize that (goes to negative large nums)

            self.update_phi_gradient(f1, f2)

            #print(self.conv1.z_phi.grad)
            self.conv2.update_phi_gradient_sh(f1, f2, f3, node)
            #print(self.conv1.z_phi.grad)

            self.train() if opt.gpus <= 1 else self.module.train()

            return score, diff
        else:
            self.forward_mode(True)
            score = self.score(x, None)

            return score


    def regularization(self):
        regularization = 0.
        for layer in self.layers:
            regularization += (1. / self.N) * layer.regularization()
        return regularization

    def get_exp_flops_l0(self):
        expected_flops, expected_l0 = 0., 0.
        for layer in self.layers:
            e_fl, e_l0 = layer.count_expected_flops_and_l0()
            expected_flops += e_fl
            expected_l0 += e_l0
        return expected_flops, expected_l0

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params

    def get_activated_neurons(self):
        return [layer.activated_neurons() for layer in self.layers]

    def get_expected_activated_neurons(self):
        return [layer.expected_activated_neurons() for layer in self.layers]

    def z_phis(self):
        return [layer.z_phi for layer in self.layers]

    def prune_rate(self):
        '''
        the number of parameters being pruned / the number of parameters
        '''
        l = [layer.activated_neurons().cpu().numpy() for layer in self.layers]
        print(l)
        return 100 - 100.0 * (l[0] * 25.0 + l[1] * l[0] * 25.0 + l[2] * l[3] + l[3] * 10.0) / (
                    20.0 * 25 + 50 * 20 * 25 + 800 * 500 + 5000)
