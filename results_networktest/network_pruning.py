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


#######################
# takes the network parameters from the conv layer and clusters them (with the purpose of removing some of them)

device=torch.cuda.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

###################################################
# DATA
dataset="mnist"


BATCH_SIZE = 100
# Download or load downloaded MNIST dataset
# shuffle data at every epoch
trainval_dataset=datasets.MNIST('data', train=True, download=True,
                    #transform=transforms.Compose([transforms.ToTensor(),
                    #transforms.Normalize((0.1307,), (0.3081,))]),
                    transform=transforms.ToTensor())

train_size = int(0.8 * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
# Same for test data
test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.ToTensor()), batch_size=BATCH_SIZE, shuffle=False)

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

optimizer=optim.Adam(net.parameters(), lr=0.001)

path="models/fashionmnist_conv:20_conv:50_fc:800_fc:500_rel_bn_trainval1.0_epo:11_acc:90.01"
path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"
#path="models/conv:10_conv:50_fc:800_fc:500_rel_bn_epo:103_acc:99.37"

net.load_state_dict(torch.load(path)['model_state_dict'])
path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"


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

        # _,prediction = torch.max(predicted_prob.data, 1)
        # #prediction = prediction.cpu().numpy()

        predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)

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

def prune(retrain, thresh1, thresh2, thresh3, thresh4):

    #PRINT
    # for name, param in net.named_parameters():
    #    print (name)
    #    print(param.shape)

    combination=[4,2,8,6,7,5,1,9,3,0]
    combination2=[16,12,17,11,6,14,7,18,10,1,0,3,15,19,13,9,8,2,5,4]
    combination3=[0,2,28,3,19,9,40,41,52,15,18,11,50,31,64,39,38,63,10,21,7,34,30,22,56,25,70,35,26,51,4,47,55,58,16,59,73,12,27,29,33,32,43,1,46, 23,45,20,74,48,91,13,42,14,61,53,65,67,6,76,68,79,36,78,44,49,5,85,84,54,81,95,82,93,92,83,72,87,71,90,60,24,77,57,75,96,99,98,88,89,97,80,94,17,69,8,86,66,62,37]
    combination4=[10,22,24,19,12,23,17,14,16,20,21,11,18,13,15,8,5,6,9,4,0,3,7,2,1]

    #these numbers from the beginning will be cut off
    combination=combination[:thresh1]
    combination2=combination2[:thresh2]
    combination3=combination3[:thresh3]
    combination4=combination4[:thresh4]


    combination = torch.LongTensor(combination)
    combination2 = torch.LongTensor(combination2)
    combination3 = torch.LongTensor(combination3)
    combination4 = torch.LongTensor(combination4)
    remaining1, remaining2, remaining3, remaining4=nodesNum1-len(combination), nodesNum2-len(combination2), nodesFc1-len(combination3), nodesFc2-len(combination4)

    filename = "%s_retrained2.txt" % path
    with open(filename, "a+") as file:
        file.write("\n\nprunedto:%d_%d_%d_%d\n\n" % (remaining1, remaining2, remaining3, remaining4))
        print("\n\nprunedto:%d_%d_%d_%d\n\n" % (remaining1, remaining2, remaining3, remaining4))



    placeholder_weight=net.c1.weight.data[combination].clone()
    placeholder_bias=net.c1.bias.data[combination].clone()
    net.c1.weight.data[combination]=0; net.c1.bias.data[combination] = 0
    net.c3.weight.data[combination2] = 0; net.c3.bias.data[combination2] = 0
    net.c5.weight.data[combination3] = 0;net.c5.bias.data[combination3] = 0
    net.f6.weight.data[combination4] = 0;net.f6.bias.data[combination4] = 0


    print("After pruning")
    evaluate()

    if retrain:
        def gradi(module):
            module[combination]=0
            #print(module[21])
        h1 = net.c1.weight.register_hook(gradi)
        #h = net.c1.weight.register_hook(lambda gradi: gradi[1] * 0)

        def gradi2(module):
            module[combination2]=0
            #print(module[1])
        h2 = net.c3.weight.register_hook(gradi2)

        def gradi3(module):
            module[combination3] = 0
            # print(module[1])

        h3 = net.c5.weight.register_hook(gradi3)

        def gradi4(module):
            module[combination4] = 0
            # print(module[1])

        h4 = net.f6.weight.register_hook(gradi4)


        print("Retraining")

        #here retraining works
        net.train()
        stop = 0; epoch = 0; best_accuracy = 0; entry = np.zeros(3); best_model = -1; early_stopping=150
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
                torch.save(best_model, "%s_retrained_epo:%d_prunedto:%d_%d_%d_%d_acc:%.2f" % (path, epoch, remaining1, remaining2, remaining3, remaining4, best_accuracy))

                entry[0] = accuracy;
                entry[1] = loss
                with open(filename, "a+") as file:
                    file.write("\n Epoch: %d\n" % epoch)
                    file.write(",".join(map(str, entry)) + "\n")


        print(loss.item())
        accuracy = evaluate()


######################################################################
# COMPUTE PRUNED (NOT RETRAINED) ACCURACIES

from itertools import chain, combinations

def compute_combinations(file_write):
    for name, param in net.named_parameters():
        print (name)
        print (param.shape)
        layer="f6.weight"
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
                        with open("results_running/combinations_pruning_mnist_%s_%s.txt" % (path[7:],layer), "a+") as textfile:
                            textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))

                all_results[r]=results

                # import pickle
                # filename='combinations_all_results_rel_bn_%d.pkl' % r
                # file=open(filename, 'wb')
                # pickle.dump(all_results, file)
                # file.close()
############################################################################################################################
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
                        with open("results_running/combinations_pruning_mnist_%s_%s.txt" % (path[7:],layer), "a+") as textfile:
                            textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))

                #all_results[r]=results

                    # import pickle
                    # filename='combinations_all_results_rel_bn_%d.pkl' % r
                    # file=open(filename, 'wb')
                    # pickle.dump(all_results, file)
                    # file.close()


#################
enddummy=1
for i1 in [5,4,3, 6]:
    for i2 in [13, 12, 11, 10, 9, 8, 7, 6]:
        for i3 in [60, 55, 50, 45, 40, 35, 30, 25, 20, 15]:
            for i4 in [15, 13, 11, 9, 7, 5]:
                prune(True, i1, i2, i3, i4)

#file_write=True #only one file_write here (and one read fie)
#compute_combinations_random(file_write)
#dummy

