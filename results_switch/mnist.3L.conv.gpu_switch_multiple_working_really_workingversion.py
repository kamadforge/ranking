#it's the same test as for mnist.#L.py but with conv layers (con lenet)
#it's also a gpu version which add extra gpu support to the previous version of mnist.3L.conv.py (which wa deleted and this version was named after this)

#transforms the input data

# the difference between this file nad mnist.#L.conv.gpu (without switch is
#1. changing the loss function to cross entropy plus KL
#2. addding loading the weights (could be added there too)
#3. adding require_grad = False option for network layers

import torch.utils.data

import torch
from torch import nn, optim
import torch.nn.functional as f

import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import csv
import pdb

from torch.nn.parameter import Parameter

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"
print(device)
print("Drop")

##############################3
# CHANGE
# PARAMS hidden_dim
# filename
# parameter size (in the architecture part)


#############################
# PARAMS

early_stopping=350
sum_average=0; conv1=10; conv2=20; fc1=100; fc2=25

dataset="fashionmnist" #to load the proper fashionmnist model

how_many_epochs=200
annealing_steps = float(8000. * how_many_epochs)
beta_func = lambda s: min(s, annealing_steps) / annealing_steps
alpha_0 = 2  # below 1 so that we encourage sparsity
switch_init=-1
layer='f6'
hidden_dims={'c1': conv1, 'c3': conv2, 'f5': fc1, 'f6' : fc2}
hidden_dim = hidden_dims[layer] #it's a number of parameters we want to estimate, e.g. # conv1 filters

trainval_perc=1
BATCH_SIZE = 100

# PARAMS

filename="%s_test_conv_relu_bn_drop_trainval%.1f_conv:%d_conv:%d_fc:%d_fc:%d.txt" % (dataset, trainval_perc, conv1, conv2, fc1, fc2)


###################################################
# DATA



# Download or load downloaded MNIST dataset
# shuffle data at every epoch

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


##############################################################################
# NETWORK (conv-conv-fc-fc)

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

        self.parameter = Parameter(switch_init*torch.ones(hidden_dim),requires_grad=True) # this parameter lies #S

    def switch_func(self, output):
        #############S
        phi = f.softplus(self.parameter)
        # """directly use mean of Dir RV."""
        S = phi / torch.sum(phi)

        # Smax = torch.max(S)
        # Sprime = S/Smax
        Sprime = S

        for i in range(len(Sprime)):
            output[:, i] *= Sprime[i].expand_as(output[:, i])

        ##################
        return output, Sprime

    def forward(self, x):

        # FEED FORWARD NETWORK
        # x=x.view(-1,784)
        # output=f.relu(self.fc1(x))
        # output=self.bn1(output)
        # output=f.relu(self.fc2(output))
        # output=self.bn2(output)
        # output=self.fc3(output)
        # return output

        #x=x.view(-1,784)
        output=self.c1(x)
        if layer=='c1':
            output, Sprime = self.switch_func(output)

        #for i in range(len(S)):
        #output[:, i] = output[:, i] * S[i]


        #output = output[1] * S
        ##############

        output=f.relu(self.s2(output))
        output=self.bn1(output)
        output=self.drop_layer(output)
        output=self.c3(output)
        if layer=='c3':
            output, Sprime = self.switch_func(output)

        output=f.relu(self.s4(output))
        output=self.bn2(output)
        output=self.drop_layer(output)
        output=output.view(-1, self.nodesNum2*4*4)

        output=self.c5(output)
        if layer=='f5':
            output, Sprime = self.switch_func(output)

        output=self.f6(output)
        if layer=='f6':
            output, Sprime = self.switch_func(output)

        #output = self.f7(output)



        # for i in range(len(Sprime)):
        #     output[:, i] *= Sprime[i].expand_as(output[:, i])

        return output, Sprime


####################

nodesNum1, nodesNum2, nodesFc1, nodesFc2=10,20,100,25
net=Lenet(nodesNum1,nodesNum2,nodesFc1,nodesFc2).to(device)
criterion = nn.CrossEntropyLoss()

optimizer=optim.Adam(net.parameters(), lr=0.001)

###############################################################################
# LOAD MODEL (optionally)

path="models/fashionmnist_conv:20_conv:50_fc:800_fc:500_rel_bn_trainval1.0_epo:11_acc:90.01"
path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
#path="/home/kamil/Dropbox/Current_research/python_tests/results_networktest/models/fashionmnist_90.04"
if dataset=="mnist":
    path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"
elif dataset=="fashionmnist":
    path="models/fashionmnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:62_acc:90.04"
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"
#path="models/conv:10_conv:50_fc:800_fc:500_rel_bn_epo:103_acc:99.37""
#path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:11_switch_acc:99.15"
#path="/home/kamil/Dropbox/Current_research/python_tests/Dir_switch/models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:2_acc:98.75"

net.load_state_dict(torch.load(path)['model_state_dict'], strict=False)
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
        predicted_prob = net.forward(images)[0]  # if using switches
        #predicted_prob = net.forward(images)
        predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()
        #print(correct)
    # print(str(correct) +" "+ str(total))
    # pdb.set_trace()
    accuracy = 100 * float(correct) / total
    print("accuracy: %.2f %%" % (accuracy))
    return accuracy

print("Loaded model:")
evaluate()


#######################s
# LOSS



def loss_function(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps):
    # BCE = f.binary_cross_entropy(prediction, true_y, reduction='sum')
    BCE = criterion(prediction, true_y)

    return BCE


###########################

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

    return BCE + annealing_rate * KLD / how_many_samps, BCE, KLD, annealing_rate * KLD / how_many_samps





###################################################
# RUN TRAINING

def run_experiment(early_stopping, nodesNum1, nodesNum2, nodesFc1, nodesFc2):
    print("\nRunning experiment\n")


    #net=Lenet(nodesNum1, nodesNum2, nodesFc1, nodesFc2).to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer=optim.Adam(net.parameters(), lr=0.001)

    #path="/home/kamil/Dropbox/Current_research/python_tests/results_networktest/models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"
    #net.load_state_dict(torch.load(path), strict=False)
    print("Evaluate:\n")
    evaluate()
    for name, param in net.named_parameters():
        print(name)
        print(param[1])
    # for name, param in net.named_parameters():
    #     print(name)
    #     #print (name, param.shape)
    #     #print("/n")
    #     if name!="parameter":
    #         param.requires_grad=False
    #     print(param.requires_grad)
    accuracy=evaluate()

    # h = net.c1.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.c3.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.c5.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.f6.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.c1.bias.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.c3.bias.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.c5.bias.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.f6.bias.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.bn1.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.bn1.bias.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.bn2.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.bn2.bias.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.output.weight.register_hook(lambda grad: grad * 0)  # double the gradient
    # h = net.output.bias.register_hook(lambda grad: grad * 0)  # double the gradient

    for name, param in net.named_parameters():
        # print(name, param.shape)
        #print(name)
        if (name != "parameter"):
            h = param.register_hook(lambda grad: grad * 0)  # double the gradient


    accuracy = evaluate()

    print("Retraining\n")
    net.train()
    stop=0; epoch=0; best_accuracy=0; entry=np.zeros(3); best_model=-1
    while (stop<early_stopping):
        epoch=epoch+1
        annealing_rate = beta_func(epoch)
        net.train()
        evaluate()
        for i, data in enumerate(train_loader):
            inputs, labels=data
            inputs, labels=inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, S=net(inputs) #when switc hes
            #outputs=net(inputs)
            #loss=criterion(outputs, labels)
            loss, BCE, KLD, KLD_discounted = loss_functionKL(outputs, labels, S, alpha_0, hidden_dim, BATCH_SIZE, annealing_rate)
            #loss=loss_function(outputs, labels, 1, 1, 1, 1)
            loss.backward()
            #print(net.c1.weight.grad[1, :])
            #print(net.c1.weight[1, :])
            optimizer.step()
            # if i % 100==0:
            #    print (i)
            #    print (loss.item())
            #    evaluate()
        #print (i)
        print (loss.item())
        print('BCE: %.4f, KDE: %.4f, KDE_discounted: %.4f' % (BCE, KLD, KLD_discounted))
        accuracy=evaluate()
        print ("Epoch " +str(epoch)+ " ended.")
        # for name, param in net.named_parameters():
        #     print(name)
        #     print(param[1])


        print("S")
        print(S)
        print("max: %.4f, min: %.4f" % (torch.max(S), torch.min(S)))
        ranks_sorted = np.argsort(S.cpu().detach().numpy())[::-1]
        print(",".join(map(str, ranks_sorted)))
        if (epoch==3):
            torch.save(S, 'results/%s/switch_init_-1,alpha_2_proper/layer-%s_epoch-%s_accuracy-%.2f.pt' % (dataset, layer, epoch, accuracy))
        #print(torch.argsort(S), descending=True)

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
            torch.save({'model_state_dict' : best_model, 'optimizer_state_dict': best_optim}, "models/%s_conv:%d_conv:%d_fc:%d_fc:%d_rel_bn_drop_trainval_modelopt%.1f_epo:%d_acc:%.2f" % (dataset, conv1, conv2, fc1, fc2, trainval_perc, epoch, best_accuracy))

        print("\n")
        #write
        entry[0]=accuracy; entry[1]=loss
        # with open(filename, "a+") as file:
        #     file.write(",".join(map(str, entry))+"\n")
    return best_accuracy, epoch, best_model

print("\n\n NEW EXPERIMENT:\n")







######################################################
#single run  avergaed pver n iterations  

for i in range(3):
    # with open(filename, "a+") as file:
    #     file.write("\nInteration: "+ str(i)+"\n")
    print("\nIteration: "+str(i))
    best_accuracy, num_epochs, best_model=run_experiment(early_stopping, conv1, conv2, fc1, fc2)
    sum_average+=best_accuracy
    average_accuracy=sum_average/(i+1)

    # with open(filename, "a+") as file:
    #     file.write("\nAv accuracy: %.2f, best accuracy: %.2f\n" % (average_accuracy, best_accuracy))
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