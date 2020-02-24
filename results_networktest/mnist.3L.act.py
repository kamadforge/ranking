# transforms the input data

import torch
import torch.optim as optim
from torch import nn, optim
import torch.nn.functional as f

import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import csv

# import matplotlib.pyplot as plt

############################################3
# DATASET

dataset = "mnist"
filename = "results_running/networktest_mnist_3L_drop_batchnorm_reluafter.txt"
BATCH_SIZE = 100
# Download or load downloaded MNIST dataset
# shuffle data at every epoch
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   # transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    # datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)


# train_loader = torch.utils.data.DataLoader(
#                 dataset=train_set,
#                 batch_size=batch_size,
#                 shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#                dataset=test_set,
#                batch_size=batch_size,
#                shuffle=False)

#######################################################
# NETWORK

class Lenet(nn.Module):
    def __init__(self, nodesNum1, nodesNum2):
        super(Lenet, self).__init__()

        self.fc1 = nn.Linear(784, nodesNum1)
        self.bn1 = nn.BatchNorm1d(nodesNum1)
        self.fc2 = nn.Linear(nodesNum1, nodesNum2)
        self.bn2 = nn.BatchNorm1d(nodesNum2)
        self.fc3 = nn.Linear(nodesNum2, 10)

        self.drop_layer1 = nn.Dropout(p=0.5)
        self.drop_layer2 = nn.Dropout(p=0.5)

    def forward(self, x, activations=0):

        # x=x.view(-1,784)
        # output=f.relu(self.fc1(x))
        # output=self.bn1(output)
        # output=f.relu(self.fc2(output))
        # output=self.bn2(output)
        # output=self.fc3(output)
        # return output

        x = x.view(-1, 784)
        output = self.fc1(x)
        output = f.relu(self.bn1(output))
        # output=self.drop_layer1(output)
        if SAVE_ACTIVS and self.training == False:
            # activations.append(output)
            # activations_dummy=torch.cat((activations,output))
            # activations=activations_dummy
            activations_dummy = torch.cat((activations['l1'], output))
            activations['l1'] = activations_dummy

        output = self.fc2(output)
        output = f.relu(self.bn2(output))
        # output=self.drop_layer1(output)
        if SAVE_ACTIVS and self.training == False:
            activations_dummy = torch.cat((activations['l2'], output))
            activations['l2'] = activations_dummy

        output = self.fc3(output)
        return output, activations


###############################################


def run_experiment(nodesNum1, nodesNum2):
    net = Lenet(nodesNum1, nodesNum2)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer=optim.Adam(net.parameters(), lr=0.01)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    # activations_all=[]

    def evaluate():
        # print('Prediction when network is forced to predict')
        net.eval()
        activations = {'l1': torch.tensor(()), 'l2': torch.tensor(())}
        correct = 0
        total = 0
        for j, data in enumerate(test_loader):
            images, labels = data
            predicted_prob, activations = net.forward(images.view(-1, 28 * 28), activations)
            predicted = np.argmax(predicted_prob.detach().numpy(), axis=1)
            total += labels.size(0)
            correct += (predicted == labels.numpy()).sum().item()
        # print(len(activations))
        # print(str(correct) +" "+ str(total))
        accuracy = 100 * float(correct) / total
        print("accuracy: %.2f %%" % (accuracy))
        return accuracy, activations

    stop = 0;
    epoch = 0;
    best_accuracy = 0;
    entry = np.zeros(3)
    while (stop < 350):
        epoch = epoch + 1
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()
            outputs, activations = net(inputs, )
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # if i % 100==0:
            #    print (i)
            #   print (loss.item())
        # print (i)
        print(loss.item())
        accuracy, activations = evaluate()
        torch.save(activations, 'activations/mnist.3L/activations_epoch-%d' % epoch)
        print("Epoch " + str(epoch) + " ended.")
        if (accuracy <= best_accuracy):
            stop = stop + 1
            entry[2] = 0
        else:
            best_accuracy = accuracy
            print("Best updated!")
            stop = 0
            entry[2] = 1
            best_model = net.state_dict()
            best_optim = optimizer.state_dict()
            # if best_accuracy>desired_accuracy:
            torch.save({'model_state_dict': best_model, 'optimizer_state_dict': best_optim},
                       "models/%s_nodes:%d_nodes:%d_rel_bn_drop_trainval_modelopt_epo:%d_acc:%.2f" % (
                           dataset, nodesNum1, nodesNum2, epoch, best_accuracy))

        print("\n")
        # write
        entry[0] = accuracy;
        entry[1] = loss
        if WRITE:
            with open(filename, "a+") as file:
                file.write(",".join(map(str, entry)) + "\n")
    return best_accuracy, epoch


###########################################################
desired_accuracy = 98
WRITE = False
SAVE_ACTIVS = True
activations = torch.Tensor(())

# best_accuracy, num_epochs=run_experiment(300, 100)

for i in [90]:
    if WRITE:
        with open(filename, "a+") as file:
            file.write("\n\nnumber of hidden nodes: " + str(i) + "\n")
    best_accuracy, num_epochs = run_experiment(152, i)
    if WRITE:
        with open(filename, "a+") as file:
            file.write("Best accuracy: %.2f in %d epochs" % (best_accuracy, num_epochs))