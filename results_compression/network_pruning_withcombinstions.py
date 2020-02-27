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

dataset="fashionmnist"


###################################################
# DATA


if dataset=="fashionmnist":

    trainval_dataset=datasets.FashionMNIST('data', train=True, download=True,
                        #transform=transforms.Compose([transforms.ToTensor(),
                        #transforms.Normalize((0.1307,), (0.3081,))]),
                        transform=transforms.ToTensor())

    train_size = int(trainval_perc * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    test_dataset=datasets.FashionMNIST('data', train=False, transform=transforms.ToTensor())

elif dataset=="mnist":

    trainval_dataset = datasets.MNIST('data', train=True, download=True,
                                             # transform=transforms.Compose([transforms.ToTensor(),
                                             # transforms.Normalize((0.1307,), (0.3081,))]),
                                             transform=transforms.ToTensor())

    train_size = int(trainval_perc * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    test_dataset = datasets.MNIST('data', train=False, transform=transforms.ToTensor())


# Load datasets

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
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


nodesNum1, nodesNum2, nodesFc1, nodesFc2=10,20,100,25
net=Lenet(nodesNum1,nodesNum2,nodesFc1,nodesFc2).to(device)
criterion = nn.CrossEntropyLoss()

#optimizer=optim.Adam(net.parameters(), lr=0.001)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)


#path="models/fashionmnist_conv:20_conv:50_fc:800_fc:500_rel_bn_trainval1.0_epo:11_acc:90.01"
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
path="models/mnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo540_acc99.27"
#path="models/fashionmnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:62_acc:90.04"
#path="models/conv:10_conv:50_fc:800_fc:500_rel_bn_epo:103_acc:99.37""
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:11_switch_acc:99.15"
#path="/home/kamil/Dropbox/Current_research/python_tests/Dir_switch/models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:2_acc:98.75"

net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'], strict=False)
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"


########################################################
# EVALUATE

def evaluate():
    # print('Prediction when network is forced to predict')
    net.eval()
    correct = 0
    total = 0
    for j, data in enumerate(test_loader):
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
    print("accuracy: %.2f %%" % (accuracy))
    return accuracy

print("Loaded model:")
evaluate()

########################################################
# RETRAIN

def prune(retrain, thresh1, thresh2, thresh3, thresh4, write, save):

    #reload the model
    net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'], strict=False)

    #PRINT NAMED_PARAMETERS
    # for name, param in net.named_parameters():
    #    print (name)
    #    print(param.shape)

    ##### THRESHOLD
    # thresh is what we keep, combinationss is what we discard

    thresh = [thresh1, thresh2, thresh3, thresh4]


    #### GET RANKS

    if method=='filter_ranking':
        if dataset=="mnist":

        #from best to worst
            combinationss=[
                [1,8,7,4,6,3,9,2,0,5],
            [2,8,9,19,4,12,5,14,11,6,3,7,18,1,17,15,16,13,0,10],
                [56,86,25,64,33,17,23,96,46,52,43,22,81,15,1,44,39,85,19,8,58,63,29,70,14,95,27,73,72,45,68,4,13,99,75,47,34,89,97,88,61,53,
                 21,50,93,57,94,51,82,60,98,40,76,62,30,9,84,
                 37,0,42,80,71,92,79,32,38,78,11,90,5,2,87,66,65,16,55,48,36,18,67,35,91,83,10,24,31,7,28,20,74,54,6,12,3,49,69,59,77,26,41],
                [1,7,2,3,0,4,6,9,5,8,10,24,13,19,12,21,23,16,22,18,17,15,14,11,20]
            ]
        #from worse to best
        elif dataset=="fashionmnist":
            combinationss=[
                [8, 4, 3, 9, 2, 1, 5, 6, 7, 0],
                [6, 16, 2, 19, 11, 3, 1, 18, 7, 15, 17, 8, 12, 14, 4, 9, 13, 0, 10, 5],
                [0,30,87,66,17,28,62,54,27,11,1,86,96,93,69,48,44,84,8,41,67,47,5,53,16,97,56,23,61,18,24,29,43,38,59,12,72,14,85,74,21,10,83,51,26,31,35,65,6,46,99,78,42,90,49,19,55,82,22,91,34,36,52,81,73,98,7,2,9,58,57,95,15,71,4,75],
                [16, 10, 22, 21, 23, 11, 15, 18, 17, 20, 12, 24, 14, 13, 19, 6, 0, 4, 2, 3, 7, 9, 8, 1, 5]

            ]

        # these numbers from the end will be cut off, meaning the worse will be cut off
        # these numbers from the end will be cut off, meaning the worse will be cut off
        for i in range(len(combinationss)):
            combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:])


    else:
        combinationss=[]
        combinationss=magnitude_rank.get_ranks(method)
        # for i in range(4):
        #     combinationss.append(torch.LongTensor(combinat[i]))

        # these numbers from the end will be cut off, meaning the worse will be cut off
        # these numbers from the end will be cut off, meaning the worse will be cut off
        for i in range(len(combinationss)):
            combinationss[i] = torch.LongTensor(combinationss[i][thresh[i]:].copy())











    #remaining1, remaining2, remaining3, remaining4=nodesNum1-len(combinationss[0]), nodesNum2-len(combinationss[1]), nodesFc1-len(combinationss[2]), nodesFc2-len(combinationss[3])
    #remaining1, remaining2, remaining3, remaining4 = combinationss[0], combinationss[1],
    filename = "%s_retrained_paramsearch1.txt" % path

    if write:
        with open(filename, "a+") as file:
            file.write("\n\nprunedto:%d_%d_%d_%d\n\n" % (thresh[0], thresh[1], thresh[2], thresh[3]))
    print("\n\n\nprunedto:%d_%d_%d_%d\n" % (thresh[0], thresh[1], thresh[2], thresh[3]))


    ########## PRUNE/ ZERO OUT THE WEIGHTS

    it = 0
    for name, param in net.named_parameters():
        #print(name)
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
    evaluate()

    ############## RETRAIN

    if retrain:
        def gradi(module):
            module[combinationss[0]]=0
            #print(module[21])
        h1 = net.c1.weight.register_hook(gradi)
        h11 = net.c1.bias.register_hook(gradi)
        h12 = net.bn1.weight.register_hook(gradi)
        h13 = net.bn1.bias.register_hook(gradi)
        #h = net.c1.weight.register_hook(lambda gradi: gradi[1] * 0)

        def gradi2(module):
            module[combinationss[1]]=0
            #print(module[1])

        h2 = net.c3.weight.register_hook(gradi)
        h21 = net.c3.bias.register_hook(gradi)
        h22 = net.bn2.weight.register_hook(gradi)
        h23 = net.bn2.bias.register_hook(gradi)

        #h2 = net.c3.weight.register_hook(gradi2)

        def gradi3(module):
            module[combinationss[2]] = 0
            # print(module[1])

        h3 = net.c5.weight.register_hook(gradi3)
        h31 = net.c5.bias.register_hook(gradi3)

        def gradi4(module):
            module[combinationss[3]] = 0
            # print(module[1])

        h4 = net.f6.weight.register_hook(gradi4)
        h41 = net.f6.bias.register_hook(gradi4)


        print("Retraining")

        #here retraining works
        net.train()
        stop = 0; epoch = 0; best_accuracy = 0; entry = np.zeros(3); best_model = -1; early_stopping=20
        while (stop < early_stopping):
            epoch = epoch + 1
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                net.c1.weight.grad #the hook is automatically applied, here we just check the gradient
                optimizer.step()
                #net.c1.weight.data[1] = 0  # instead of hook
                #net.c1.bias.data[1] = 0  # instead of hook
                # if i % 100==0:
                #    print (i)
                #   print (loss.item())

            print(loss.item())
            accuracy = evaluate()
            print("Epoch " + str(epoch) + " ended.")

            if (accuracy <= best_accuracy):
                stop = stop + 1
                entry[2] = 0
            else:
                best_accuracy = accuracy
                print("Best updated")
                stop = 0
                entry[2] = 1
                best_model = net.state_dict()
                if save:
                    if best_accuracy>save_accuracy:
                        torch.save(best_model, "%s_retrained_epo:%d_prunedto:%d_%d_%d_%d_acc:%.2f" % (path, epoch, thresh[0], thresh[1], thresh[2], thresh[3], best_accuracy))

                entry[0] = accuracy;
                entry[1] = loss
                if write:
                    with open(filename, "a+") as file:
                        file.write("\n Epoch: %d\n" % epoch)
                        file.write(",".join(map(str, entry)) + "\n")
                        if (accuracy>98.9):
                            file.write("Yes\n")
                        elif (accuracy>98.8):
                            file.write("Ok\n")

        print(loss.item())
        print("Final: "+ best_accuracy)
        accuracy = evaluate()


#########################################################################
######################################################################
# COMPUTE PRUNED (NOT RETRAINED) ACCURACIES

from itertools import chain, combinations

def compute_combinations(file_write):
    for name, param in net.named_parameters():
        #print (name)
        #print (param.shape)
        layer="c5.weight"
        print(layer)
        if layer in name:
            layerbias=layer[:3]+"bias"
            params_bias = net.state_dict()[layerbias]

            all_results={}
            s=torch.range(0,param.shape[0]-1) #list from 0 to 19 as these are the indices of the data tensor
            for r in range(1,param.shape[0]): #produces the combinations of the elements in s
                results=[]
                for combination in list(combinations(s, r)):
                    combination=torch.LongTensor(combination)


                    print(combination)
                    params_saved = param[combination].clone(); param_bias_saved=params_bias[combination].clone()
                    param[combination[0]] = 0
                    param[combination] = 0; params_bias[combination]=0
                    accuracy = evaluate()
                    param[combination] = params_saved; params_bias[combination]=param_bias_saved

                    results.append((combination, accuracy))




                    if file_write:
                        with open("results_running/combinations_pruning_mnist_all_%s_%s.txt" % (path[7:],layer), "a+") as textfile:
                            textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))

                all_results[r]=results

                # import pickle
                # filename='combinations_all_results_rel_bn_%d.pkl' % r
                # file=open(filename, 'wb')
                # pickle.dump(all_results, file)
                # file.close()
############################################################################################################################
# CHOOSES RANDOM COMBINATION and then removed one of the random nodes and computes accuracy for that node

def compute_combinations_random(file_write):
    for name, param in net.named_parameters():
        print (name)
        print (param.shape)
        layer="c5.weight"
        if layer in name:
            layerbias=layer[:3]+"bias"
            params_bias = net.state_dict()[layerbias]
            while (True):


                all_results={}
                #s=torch.range(0,49) #list from 0 to 19 as these are the indices of the data tensor
                #for r in range(1,50): #produces the combinations of the elements in s
                #    results=[]
                randperm=np.random.permutation(param.shape[0])
                randint=0
                while (randint==0):
                    randint=np.random.randint(param.shape[0])
                randint_indextoremove=np.random.randint(randint)
                combination=randperm[:randint]
                combination2=np.delete(combination, randint_indextoremove)
                print(combination[randint_indextoremove])

                if file_write:
                        with open("results_running/combinations_pruning_mnist_%s_%s.txt" % (path[7:],layer), "a+") as textfile:
                            textfile.write("%d\n" % randint_indextoremove)


                for combination in [combination, combination2]:
                    #for combination in list(combinations(s, r)):
                    
                    combination=torch.LongTensor(combination)

                    print(combination)
                    params_saved=param[combination].clone()
                    param_bias_saved=params_bias[combination].clone()

                    #param[torch.LongTensor([1, 4])] = 0
                    #workaround, first using multiple indices does not work, but if one of the change first then it works to use  param[combinations]
                    if len(combination) !=0:
                        param[combination[0]]=0
                        #param[combination]=0
                        params_bias[combination]=0
                    
                    accuracy=evaluate()
                    param[combination]=params_saved
                    params_bias[combination]=param_bias_saved


                    if file_write:
                        with open("results_running/combinations_pruning_fashionmnist_%s_%s.txt" % (path[7:],layer), "a+") as textfile:
                            textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))

                #all_results[r]=results

                    # import pickle
                    # filename='combinations_all_results_rel_bn_%d.pkl' % r
                    # file=open(filename, 'wb')
                    # pickle.dump(all_results, file)
                    # file.close()


#################
enddummy=1
#prune(True, 5, 11, 31, 10)
#prune(True, 5, 10, 21, 10)
#prune(True, 5, 9, 41, 10)



#######################
# PRUNING

#SAVING MODEL
#the models are saved in the savedirectory as the original model
if dataset=="mnist":
    save_accuracy=99.00
if dataset=="fashionmnist":
    save_accuracy=89.50

save=False

#WRITING
# the output text file will be saved also in the same directory as the original model
write=False

methods=['filter_ranking', 'l1', 'l2']

for method in methods:
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

     for i1 in [2, 3, 4, 5, 6, 7, 10]:
         for i2 in [4, 6, 8, 10, 12, 20]:
             for i3 in [20, 30, 40, 50, 60, 100]:
                 for i4 in [5, 10, 15, 20, 25]:
                     for method in methods:
                         print("\n\n %s \n" % method)
                         prune(False, i1, i2, i3, i4, write, save)
                     print("\n*********************************\n\n")

###########################3
# COMPUTING COMBINATIONS

file_write=True #oly one file_write here (and one read fie)
#compute_combinations(file_write)
#dummy


print("\n\nEND")
