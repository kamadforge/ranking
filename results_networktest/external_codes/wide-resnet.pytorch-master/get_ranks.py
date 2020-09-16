import torch
import numpy as np

def l1_ranks(net, method):

    for name, param in net.named_parameters():
        print(name, param.shape)

    layers = {"layer1.0", "layer1.1", "layer1.2", "layer1.3",
                   "layer2.0", "layer2.1", "layer2.2", "layer2.3",
                   "layer3.0", "layer3.1", "layer3.2", "layer3.3"}

    magnitude_ranks = {}
    magnitude_weightranks = {}
    for layer in layers:
        magnitude_dic={}
        for name, param in net.named_parameters():
            if layer in name and (("bn" not in name) and ("parameter" not in name) and ("shortcut" not in name)) and ("conv1.weight" in name):

                print(name, param.shape)
                m = torch.flatten(param, start_dim=1)

                l2 = torch.norm(m, p=2, dim=1)
                l1 = torch.norm(m, p=1, dim=1)

                l1 = l1.detach().cpu().numpy()
                l2 = l2.detach().cpu().numpy()

                # for lenet change the order
                # l1rank = np.argsort(l1)[::-1]
                # l2rank = np.argsort(l2)[::-1]

                l1rank = np.argsort(l1)[::-1]
                l2rank = np.argsort(l2)[::-1]
                l1weightrank = np.sort(l1)[::-1]
                l2weightrank = np.sort(l2)[::-1]



                # print(l1rank)

                if method == 'l1':
                    magnitude_ranks[layer]=l1rank
                    magnitude_weightranks[layer] = l1weightrank
                elif method == 'l2':
                    magnitude_ranks[layer]=l2rank
                    magnitude_weightranks[layer] = l2weightrank

    return magnitude_ranks, magnitude_weightranks




