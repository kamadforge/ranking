# for four layer lenet:
# -loads the models
# -evaluates the model
#- prunes the model
# -retrains the model
# - computes the combinations with the pruned model


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
import magnitude_rank



#######################
# takes the network parameters from the conv layer and clusters them (with the purpose of removing some of them)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

trainval_perc=1
BATCH_SIZE = 100

#dataset="mnist"

global train_loader
global test_loader
###################################################
# DATA

# def get_data(dataset):
#
#     if dataset=="fashionmnist":
#
#         trainval_dataset=datasets.FashionMNIST('data', train=True, download=True,
#                             #transform=transforms.Compose([transforms.ToTensor(),
#                             #transforms.Normalize((0.1307,), (0.3081,))]),
#                             transform=transforms.ToTensor())
#
#         train_size = int(trainval_perc * len(trainval_dataset))
#         val_size = len(trainval_dataset) - train_size
#         train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
#
#         test_dataset=datasets.FashionMNIST('data', train=False, transform=transforms.ToTensor())
#
#     elif dataset=="mnist":
#
#         trainval_dataset = datasets.MNIST('data', train=True, download=True,
#                                                  # transform=transforms.Compose([transforms.ToTensor(),
#                                                  # transforms.Normalize((0.1307,), (0.3081,))]),
#                                                  transform=transforms.ToTensor())
#
#         train_size = int(trainval_perc * len(trainval_dataset))
#         val_size = len(trainval_dataset) - train_size
#         train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
#
#         test_dataset = datasets.MNIST('data', train=False, transform=transforms.ToTensor())
#
#
#     # Load datasets
#     global test_loader
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(
#         #datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
#         test_dataset,
#         batch_size=BATCH_SIZE, shuffle=False)
#
#
#
# nodesNum1, nodesNum2, nodesFc1, nodesFc2=10,20,100,25
# # net=Lenet(nodesNum1,nodesNum2,nodesFc1,nodesFc2).to(device)
# criterion = nn.CrossEntropyLoss()

#optimizer=optim.Adam(net.parameters(), lr=0.001)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

#path="models/fashionmnist_conv:20_conv:50_fc:800_fc:500_rel_bn_trainval1.0_epo:11_acc:90.01"
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
# path="models/mnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo540_acc99.27"
#path="models/fashionmnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:62_acc:90.04"

# net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'], strict=False)


########################################################
# EVALUATE

# def evaluate(net):
#     net.eval()
#     correct = 0
#     total = 0
#     for j, data in enumerate(test_loader):
#         images, labels = data
#         images = images.to(device)
#         predicted_prob = net.forward(images)  # images.view(-1,28*28)
#         predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
#         total += labels.size(0)
#         correct += (predicted == labels.numpy()).sum().item()
#     accuracy = 100 * float(correct) / total
#     print("accuracy: %.2f %%" % (accuracy))
#     return accuracy

#print("Loaded model:")
#zzevaluate()







    #remaining1, remaining2, remaining3, remaining4=nodesNum1-len(combinationss[0]), nodesNum2-len(combinationss[1]), nodesFc1-len(combinationss[2]), nodesFc2-len(combinationss[3])
    #remaining1, remaining2, remaining3, remaining4 = combinationss[0], combinationss[1],
    #filename = "%s_retrained_paramsearch1.txt" % path




    #print("After pruning")
    #evaluate()



#########################################################################
######################################################################
# COMPUTE PRUNED (NOT RETRAINED) ACCURACIES

from itertools import chain, combinations


def compute_combinations_lenet(file_write, net, evaluate, dataset, perturbation_method):
    print("before:")
    #evaluate(net)
    print("from other file")
    evaluate()
    print("from other")
    for name, param in net.named_parameters():
        #print (name)
        #print (param.shape)
        layer="c3.weight"
        print(layer)
        if layer in name:
            layerbias=layer[:3]+"bias"
            params_bias = net.state_dict()[layerbias]

            all_results={}
            s=torch.arange(0,param.shape[0]) #list from 0 to 19 as these are the indices of the data tensor
            for r in range(1,param.shape[0]): #produces the combinations of the elements in s
                results=[]
                for combination in list(combinations(s, r)):
                    combination=torch.LongTensor(combination)


                    print(combination)
                    params_saved = param[combination].clone(); param_bias_saved=params_bias[combination].clone()

                    ###################################33333
                    if perturbation_method=="zeroing":
                        param[combination[0]] = 0
                        params_bias[combination]=0

                        accuracy = evaluate()

                    ######################################### add noise to weights

                    #adding noise
                    elif perturbation_method=="additive_noise":
                        #norm_dist=torch.distributions.Normal(0,0.1)
                        #param[combination[0]] += norm_dist.sample(param[combination[0]].shape).to(device)

                    #multiplying by noise
                        #norm_dist = torch.distributions.Normal(1, 0.1)
                        #param[combination[0]] *= norm_dist.sample(param[combination[0]].shape)

                        # adding noise
                        accuracies=[]
                        for i in range(5):
                            norm_dist=torch.distributions.Normal(0,0.1)
                            param[combination[0]] += norm_dist.sample(param[combination[0]].shape)
                            accuracies.append(evaluate())

                        accuracy=np.mean(accuracies)
                        print("Averaged accuracy: ", accuracy)



                    ########################################333

                    #accuracy = evaluate(net)


                    param[combination] = params_saved
                    params_bias[combination]=param_bias_saved

                    results.append((combination, accuracy))




                    if file_write:
                        with open("results_running/combinations_pruning_mnist_all_%s_%s_%s.txt" % (dataset,layer, perturbation_method), "a+") as textfile:
                            textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))

                all_results[r]=results

# def compute_combinations_lenet(file_write, net, evaluate):
#     print("before:")
#     #evaluate(net)
#     print("from other file")
#     evaluate()
#     print("from other")
#     for name, param in net.named_parameters():
#         #print (name)
#         #print (param.shape)
#         layer="c1.weight"
#         print(layer)
#         if layer in name:
#             layerbias=layer[:3]+"bias"
#             params_bias = net.state_dict()[layerbias]
#
#             all_results={}
#             s=torch.range(0,param.shape[0]-1) #list from 0 to 19 as these are the indices of the data tensor
#             for r in range(1,param.shape[0]): #produces the combinations of the elements in s
#                 results=[]
#                 for combination in list(combinations(s, r)):
#                     combination=torch.LongTensor(combination)
#
#
#                     print(combination)
#                     params_saved = param[combination].clone(); param_bias_saved=params_bias[combination].clone()
#
#                     ###################################33333
#                     #param[combination[0]] = 0
#                     #params_bias[combination]=0
#
#                     ######################################### add noise to weights
#
#                     #adding noise
#                     norm_dist=torch.distributions.Normal(0,0.1)
#                     param[combination[0]] += norm_dist.sample(param[combination[0]].shape)
#
#                     #multiplying by noise
#                     #norm_dist = torch.distributions.Normal(1, 0.1)
#                     #param[combination[0]] *= norm_dist.sample(param[combination[0]].shape)
#
#                     ########################################333
#
#                     #accuracy = evaluate(net)
#                     accuracy = evaluate()
#                     param[combination] = params_saved; params_bias[combination]=param_bias_saved
#
#                     results.append((combination, accuracy))
#
#
#
#
#                     # if file_write:
#                     #     with open("results_running/combinations_pruning_mnist_all_%s_%s.txt" % (path[7:],layer), "a+") as textfile:
#                     #         textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))
#
#                 all_results[r]=results
#
#                 # import pickle
#                 # filename='combinations_all_results_rel_bn_%d.pkl' % r
#                 # file=open(filename, 'wb')
#                 # pickle.dump(all_results, file)
#                 # file.close()
#


############################################################################################################################
# CHOOSES RANDOM COMBINATION and then removed one of the random nodes and computes accuracy for that node



############# LEAVE it (but comment not to get confused
# def compute_combinations_random(file_write):
#     for name, param in net.named_parameters():
#         print (name)
#         print (param.shape)
#         layer="c1.weight"
#         if layer in name:
#             layerbias=layer[:3]+"bias"
#             params_bias = net.state_dict()[layerbias]
#             while (True):
#
#
#                 all_results={}
#                 #s=torch.range(0,49) #list from 0 to 19 as these are the indices of the data tensor
#                 #for r in range(1,50): #produces the combinations of the elements in s
#                 #    results=[]
#                 randperm=np.random.permutation(param.shape[0])
#                 randint=0
#                 while (randint==0):
#                     randint=np.random.randint(param.shape[0])
#                 randint_indextoremove=np.random.randint(randint)
#                 combination=randperm[:randint]
#                 combination2=np.delete(combination, randint_indextoremove)
#                 print(combination[randint_indextoremove])
#
#                 if file_write:
#                         with open("results_running/combinations_pruning_mnist_%s_%s.txt" % (path[7:],layer), "a+") as textfile:
#                             textfile.write("%d\n" % randint_indextoremove)
#
#
#                 for combination in [combination, combination2]:
#                     #for combination in list(combinations(s, r)):
#
#                     combination=torch.LongTensor(combination)
#
#                     print(combination)
#                     params_saved=param[combination].clone()
#                     param_bias_saved=params_bias[combination].clone()
#
#                     #param[torch.LongTensor([1, 4])] = 0
#                     #workaround, first using multiple indices does not work, but if one of the change first then it works to use  param[combinations]
#                     if len(combination) !=0:
#                         param[combination[0]]=0
#                         #param[combination]=0
#                         params_bias[combination]=0
#
#                     accuracy=evaluate(net)
#                     param[combination]=params_saved
#                     params_bias[combination]=param_bias_saved
#
#
#                     if file_write:
#                         with open("results_running/combinations_pruning_fashionmnist_%s_%s.txt" % (path[7:],layer), "a+") as textfile:
#                             textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))
#
#                 #all_results[r]=results
#
#                     # import pickle
#                     # filename='combinations_all_results_rel_bn_%d.pkl' % r
#                     # file=open(filename, 'wb')
#                     # pickle.dump(all_results, file)
#                     # file.close()
#
#
# #################




#######################
# PRUNING

#SAVING MODEL
#the models are saved in the savedirectory as the original model
# if dataset=="mnist":
#     save_accuracy=99.00
# if dataset=="fashionmnist":
#     save_accuracy=89.50

save=False

#WRITING
# the output text file will be saved also in the same directory as the original model
write=False

# methods=['filter_ranking', 'l1', 'l2']

# for method in methods:
    #numbers indicate how many are pruned

    # for i in range(1,10):
    #     print("\n\n******method: %s, percentage: %d******" % (method, i*10))
    #     prune(True, 1*i, 2*i, 10*i, 2*i, write, save)

    # print("\n\n *******method: *******%s \n" % method)
    # for i1 in [3, 5, 7]:
    #      for i2 in [4, 8, 12]:
    #          for i3 in [50]:
    #              for i4 in [20]:
    #                  prune(False, i1, i2, i3, i4, write, save)
    # 
    #  for i1 in [2, 3, 4, 5, 6, 7, 10]:
    #      for i2 in [4, 6, 8, 10, 12, 20]:
    #          for i3 in [20, 30, 40, 50, 60, 100]:
    #              for i4 in [5, 10, 15, 20, 25]:
    #                  for method in methods:
    #                      print("\n\n %s \n" % method)
    #                      prune(False, i1, i2, i3, i4, write, save)
    #                  print("\n*********************************\n\n")

###########################3
# COMPUTING COMBINATIONS

file_write=True #oly one file_write here (and one read fie)
#compute_combinations(file_write)
#dummy


print("\n\nEND")
