#it's the same test as for mnist.#L.py but with conv layers (con lenet)
#it's also a gpu version which add extra gpu support to the previous version of mnist.3L.conv.py (which wa deleted and this version was named after this)

#transforms the input data

import torch
from torch import nn, optim
import torch.nn.functional as f
import scipy.io


import torch.utils.data
from torchvision import datasets, transforms


import numpy as np
import csv
import pdb

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"
print(device)

write_to_file=False
desired_save_accuracy=99.1


###################################################
# DATA

#MNIST, FashionMNIST,housenum

dataset="MNIST"
trainval_perc=1
BATCH_SIZE = 100
# Download or load downloaded MNIST dataset
# shuffle data at every epoch

if dataset=="housenums":
    train_data = scipy.io.loadmat('/datadisk1/data/house_ numbers/train_32x32.mat')
    test_data = scipy.io.loadmat('/datadisk1/data/house_ numbers/train_32x32.mat')

    train_data_x=train_data['X'].swapaxes(2,3).swapaxes(1,2).swapaxes(0,1)
    train_data_y=train_data['y']


    tensor_x = torch.stack([torch.Tensor(i) for i in train_data_x]) # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor(i) for i in train_data_y])

    my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
    train_loader = torch.utils.data.DataLoader(my_dataset) # create your dataloader

#train_data['X'].shape (32, 32, 3, 73257)
#train_data['y'].shape (73257, 1)


else:

    trainval_dataset=datasets.__dict__[dataset]('data', train=True, download=True,
                        #transform=transforms.Compose([transforms.ToTensor(),
                        #transforms.Normalize((0.1307,), (0.3081,))]),
                        transform=transforms.ToTensor())

    train_size = int(trainval_perc * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Same for test data
    test_loader = torch.utils.data.DataLoader(
        #datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
        datasets.__dict__[dataset]('data', train=False, transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE, shuffle=True)



 ##########################################################
# NETWORK


class Lenet(nn.Module):
    def __init__(self, nodesNum1, nodesNum2, nodesFc1, nodesFc2):
        super(Lenet, self).__init__()
        self.nodesNum2=nodesNum2

        self.c1=nn.Conv2d(1, nodesNum1, 5)
        self.s2=nn.MaxPool2d(2)
        self.bn1=nn.BatchNorm2d(nodesNum1)
        self.c3=nn.Conv2d(nodesNum1,nodesNum2,5)
        self.s4=nn.MaxPool2d(2)
        self.bn2=nn.BatchNorm2d(nodesNum2)
        self.c5=nn.Linear(nodesNum2*4*4, nodesFc1)
        self.f6=nn.Linear(nodesFc1,nodesFc2)
        self.f7=nn.Linear(nodesFc2,10)

        self.drop_layer = nn.Dropout(p=0.5)
    def forward(self, x):

        # x=x.view(-1,784)
        # output=f.relu(self.fc1(x))
        # output=self.bn1(output)
        # output=f.relu(self.fc2(output))
        # output=self.bn2(output)
        # output=self.fc3(output)
        # return output

        #x=x.view(-1,784)
        output=self.c1(x)
        output=f.relu(self.s2(output))
        output=self.bn1(output)
        output=self.drop_layer(output)
        output=self.c3(output)

        output=f.relu(self.s4(output))
        output=self.bn2(output)
        output=self.drop_layer(output)
        output=output.view(-1, self.nodesNum2*4*4)

        output=self.c5(output)
        output=self.f6(output)
        output = self.f7(output)
        return output




###################################################
# RUN

def run_experiment(early_stopping, nodesNum1, nodesNum2, nodesFc1, nodesFc2):

    resume=False
    training=True

    net=Lenet(nodesNum1, nodesNum2, nodesFc1, nodesFc2).to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer=optim.Adam(net.parameters(), lr=0.001)
    optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9, weight_decay=5e-4)

    ######################################
    # LOADING MODEL/RESUME

    def load_model():
        # path="models/fashionmnist_conv:20_conv:50_fc:800_fc:500_rel_bn_trainval1.0_epo:11_acc:90.01"
        # path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
        if dataset == "MNIST":
            path = "models/mnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo540_acc99.27"
        elif dataset == "FashionMNIST":
            path = "models/fashionmnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo62_acc90.04"
        # path="models/conv:10_conv:50_fc:800_fc:500_rel_bn_epo:103_acc:99.37""
        # path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:11_switch_acc:99.15"
        # path="/home/kamil/Dropbox/Current_research/python_tests/Dir_switch/models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:2_acc:98.75"

        net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'],
                            strict=False)
        # path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
        # path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"

    ##########################################
    # EVALUATE

    # def evaluate():
    #     net.eval()
    #     correct = 0
    #     total = 0
    #     for j, data in enumerate(test_loader):
    #         images, labels = data
    #         images =images.to(device)
    #         predicted_prob = net.forward(images) #images.view(-1,28*28)
    #
    #         predicted=np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
    #         total += labels.size(0)
    #         correct += (predicted == labels.numpy()).sum().item()
    #
    #     accuracy=100 * float(correct) / total
    #     print("accuracy: %.2f %%" % (accuracy))
    #     return accuracy
    # looks same as below
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
            # print(predicted_prob)
            predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels.numpy()).sum().item()
        # print(str(correct) +" "+ str(total))
        # pdb.set_trace()
        accuracy = 100 * float(correct) / total
        print("test accuracy: %.2f %%" % (accuracy))
        return accuracy

    if resume:
        load_model()
        evaluate()

    def train():

        stop=0; epoch=0; best_accuracy=0; entry=np.zeros(3); best_model=-1
        while (stop<early_stopping):
            epoch=epoch+1
            for i, data in enumerate(train_loader):

                inputs, labels=data
                inputs, labels=inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs=net(inputs)
                loss=criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                #if i % 100==0:
                #    print (i)
                #   print (loss.item())
            #print (i)
            print (loss.item())
            accuracy=evaluate()
            print ("Epoch " +str(epoch)+ " ended.")

            if (accuracy<=best_accuracy):
                stop=stop+1
                entry[2]=0
            else:
                best_accuracy=accuracy
                print("Best updated")
                stop=0
                entry[2]=1
                best_model=net.state_dict()
                best_optim=optimizer.state_dict()
                if best_accuracy>desired_save_accuracy:
                    torch.save({'model_state_dict' : best_model, 'optimizer_state_dict': best_optim}, "models/%s_conv_%d_conv_%d_fc_%d_fc_%d_rel_bn_drop_trainval_modelopt%.1f_epo_%d_acc_%.2f" % (dataset, conv1, conv2, fc1, fc2, trainval_perc, epoch, best_accuracy))

            print("\n")
            #write
            entry[0]=accuracy; entry[1]=loss
            if write_to_file:
                with open(filename, "a+") as file:
                    file.write(",".join(map(str, entry))+"\n")
        return best_accuracy, epoch, best_model

    if training:
        best_accuracy, epoch, best_model=train()
        return best_accuracy, epoch, best_model
    else:
        return -1,-1,-1


        

print("\n\n NEW EXPERIMENT:\n")




########################################################
# PARAMS
early_stopping=600
sum_average=0; conv1=10; conv2=20; fc1=100; fc2=25
filename="%s_test_conv_relu_bn_drop_trainval%.1f_conv:%d_conv:%d_fc:%d_fc:%d.txt" % (dataset, trainval_perc, conv1, conv2, fc1, fc2)
filename="%s_test_conv_relu_bn_drop_trainval%.1f_conv:%d_conv:%d_fc:%d_fc:%d.txt" % (dataset, trainval_perc, conv1, conv2, fc1, fc2)

######################################################
#single run  avergaed pver n iterations  

for i in range(1):
    if write_to_file:
        with open(filename, "a+") as file:
            file.write("\nInteration: "+ str(i)+"\n")
    print("\nIteration: "+str(i))
    best_accuracy, num_epochs, best_model=run_experiment(early_stopping, conv1, conv2, fc1, fc2)
    sum_average+=best_accuracy
    average_accuracy=sum_average/(i+1)

    if write_to_file:
        with open(filename, "a+") as file:
            file.write("\nAv accuracy: %.2f, best accuracy: %.2f\n" % (average_accuracy, best_accuracy))
    print("\nAv accuracy: %.2f, best accuracy: %.2f\n" % (average_accuracy, best_accuracy))
    #torch.save(best_model, filename_model)

#multiple runs

# for i1 in range(1,20):        
#     for i2 in range(1,20):
#         with open(filename, "a+") as file:
#             file.write("\n\nnumber of hidden nodes 1: "+str(i1)+", hidden nodes 2: " +str(i2)+"\n")
#             print("\n\nnumber of hidden nodes 1: "+str(i1)+", hidden nodes 2: " +str(i2)+"\n")
    
#         best_accuracy, num_epochs=run_experiment(i1, i2)
#         with open(filename, "a+") as file:
#             file.write("Best accuracy: %.2f in %d epochs" % (best_accuracy, num_epochs-early_stopping))
#             print("Best accuracy: %.2f in %d epochs" % (best_accuracy, num_epochs-early_stopping))