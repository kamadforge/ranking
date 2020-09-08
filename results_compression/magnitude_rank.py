
import torch
import numpy as np


def get_ranks(method, net):
    combinationss = [];
    combinationss_weights = []

    # print(net.named_parameters()['c1.weight'])
    for name, param in net.named_parameters():

        if (("c" in name) or ("f" in name) or ("l1" in name)) and ("weight" in name): #i think f and l are both fullt connected
            print (name, param.shape)
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
                combinationss.append(l1rank)
                combinationss_weights.append(l1weightrank)
            elif method == 'l2':
                combinationss.append(l2rank)
                combinationss_weights.append(l2weightrank)

            # if method == 'l1':
            #     combinationss.append(torch.LongTensor(l1rank))
            # elif method == 'l2':
            #     combinationss.append(torch.LongTensor(l2rank))

    return combinationss
    # sum=np.sum(param[0, :].sum()
    # print(sum)

