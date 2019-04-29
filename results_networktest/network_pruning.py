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

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

###################################################
# DATA



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


net=Lenet(20,50,800,500).to(device)
criterion = nn.CrossEntropyLoss()

optimizer=optim.Adam(net.parameters(), lr=0.001)

path="models/fashionmnist_conv:20_conv:50_fc:800_fc:500_rel_bn_trainval1.0_epo:11_acc:90.01"
path="models/conv:20_conv:50_fc:800_fc:500_rel_bn_epo:366_acc:99.34"
net.load_state_dict(torch.load(path))

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

def retrain():

    #PRINT
    # for name, param in net.named_parameters():
    #    print (name)
    #    print(param.shape)

    combination=[11,  4,  8,  7, 12,  6, 16,  9,  2, 10,  0, 19, 17, 15, 5]
    combination2=[8,49,0,26,15,19,10,22,32,29,13,4,5,23,11,24,17,35,39,36,9,3,43,2,47,20,1,31,12,14,18,48,30,38,44,40,42,45,28,7]
    combination = torch.LongTensor(combination)
    combination2 = torch.LongTensor(combination2)

    placeholder_weight=net.c1.weight.data[combination].clone()
    placeholder_bias=net.c1.bias.data[combination].clone()
    net.c1.weight.data[combination]=0; net.c1.bias.data[combination] = 0
    net.c3.weight.data[combination2] = 0; net.c3.bias.data[combination2] = 0

    print("After pruning")
    evaluate()


    def gradi(module):
        module[combination]=0
        #print(module[1])
    h1 = net.c1.weight.register_hook(gradi)
    #h = net.c1.weight.register_hook(lambda gradi: gradi[1] * 0)

    def gradi(module):
        module[combination2]=0
        #print(module[1])
    h2 = net.c3.weight.register_hook(gradi)


    print("Training")
    #here retraining works
    net.train()
    stop = 0; epoch = 0; best_accuracy = 0; entry = np.zeros(3); best_model = -1; early_stopping=100
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
            torch.save(best_model, "%s_retrained_epo:%d_acc:%.2f" % (path, epoch, best_accuracy))



    print(loss.item())
    accuracy = evaluate()


retrain()

######################################################################
# COMPUTE PRUNED (NOT RETRAINED) ACCURACIES

from itertools import chain, combinations


# s=list(torch.range(1,20))
# for r in range(len(s)+1):
#     print(list(combinations(s, r)))

# def powerset(iterable):
#     "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
#     s = list(iterable)
#     return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
#
# for result in powerset([1, 2, 3]):
#     print(result)
#
# results = list(powerset([1, 2, 3]))
# print(results)

def compute_combinations(file_write):
    for name, param in net.named_parameters():
        print (name)
        print (param.shape)
        layer="c1.weight"
        if layer in name:


            all_results={}
            s=torch.range(0,param.shape[0]-1) #list from 0 to 19 as these are the indices of the data tensor
            for r in range(1,param.shape[0]): #produces the combinations of the elements in s
                results=[]
                for combination in list(combinations(s, r)):
                    combination=torch.LongTensor(combination)


                    print(combination)
                    params_saved = param[combination].clone()
                    param[combination[0]] = 0
                    param[combination] = 0
                    accuracy = evaluate()
                    param[combination] = params_saved

                    results.append((combination, accuracy))




                    if file_write:
                        with open("combinations_pruning_fashionmnist_rel_bn_%s_serw.txt" % layer, "a+") as textfile:
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
        layer="c3.weight"
        if layer in name:

            for i in range(100):


                all_results={}
                #s=torch.range(0,49) #list from 0 to 19 as these are the indices of the data tensor
                #for r in range(1,50): #produces the combinations of the elements in s
                #    results=[]
                randperm=np.random.permutation(50)
                randint=np.random.randint(50)
                combination=randperm[randint:]
                #for combination in list(combinations(s, r)):
                combination=torch.LongTensor(combination)

                print(combination)
                params_saved=param[combination].clone()
                #param[torch.LongTensor([1, 4])] = 0
                #workaround, first using multiple indices does not work, but if one of the change first then it works to use  param[combinations]
                param[combination[0]]=0
                param[combination]=0
                #zeroed[:]=0
                #print(param)
                accuracy=evaluate()
                param[combination]=params_saved

                # for i1 in range(param.shape[0]):
                #     for i2 in range(param.shape[1]):
                #         print("%d, %d" % (i1, i2))
                #         params_saved=param[i1, i2].clone()
                #         #print(param[i1, i2])
                #         param[i1, i2]=0
                #         #print(param[i1, i2])
                #         evaluate()
                #         param[i1, i2]=params_saved
                #         #print(param[i1, i2])
                #results.append((combination, accuracy))
                if file_write:
                    with open("combinations_random_pruning_rel_bn_%s.txt" % layer, "a+") as textfile:
                        textfile.write("%s: %.2f\n" % (",".join(str(x) for x in combination.numpy()), accuracy))

                #all_results[r]=results

                    # import pickle
                    # filename='combinations_all_results_rel_bn_%d.pkl' % r
                    # file=open(filename, 'wb')
                    # pickle.dump(all_results, file)
                    # file.close()


#################

file_write=False #only one file_write here (and one read fie)
#compute_combinations(file_write)


