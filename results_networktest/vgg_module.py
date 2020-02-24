#################
# architecture of VGG (the implementation of no-filters like in bayesian compression, and linear layers are like in the chengyangfu impl (in results_network test)

# torch.Size([100, 3, 32, 32])
# torch.Size([100, 64, 32, 32])
# torch.Size([100, 64, 32, 32])
# torch.Size([100, 64, 16, 16])
# torch.Size([100, 128, 16, 16])
# torch.Size([100, 128, 8, 8])
# torch.Size([100, 256, 8, 8])
# torch.Size([100, 256, 8, 8])
# torch.Size([100, 256, 8, 8])
# torch.Size([100, 256, 4, 4])
# torch.Size([100, 512, 4, 4])
# torch.Size([100, 512, 4, 4])
# torch.Size([100, 512, 2, 2])
# torch.Size([100, 512, 2, 2])
# torch.Size([100, 512, 1, 1])
# torch.Size([100, 512, 1, 1])
# torch.Size([100, 512])
# torch.Size([100, 512])
###################3

import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as f
import numpy as np
import torch.nn.init as init
import scipy.io
import socket
import matplotlib.pyplot as plt

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"
print(device)


###########################################################
# DATA

dataset = "housenums"
trainval_perc=1
BATCH_SIZE=1

if dataset=="cifar10":

    trainval_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10('data', train=False, download=True, transform = transforms.ToTensor())


    train_size = int(trainval_perc * len(trainval_dataset))
    val_size = len(trainval_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

elif dataset=='housenums':

    print(socket.gethostname())
    if 'g0' not in socket.gethostname():
        train_data = scipy.io.loadmat('/datadisk1/data/house_ numbers/train_32x32.mat')
        test_data = scipy.io.loadmat('/datadisk1/data/house_ numbers/train_32x32.mat')
    else:
        train_data = scipy.io.loadmat('/home/kadamczewski/data/house_ numbers/train_32x32.mat')
        test_data = scipy.io.loadmat('/home/kadamczewski/data/house_ numbers/test_32x32.mat')

    train_data_x = train_data['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1).swapaxes(2,3).swapaxes(1,2)
    train_data_y = train_data['y']
    train_data_y=np.where(train_data_y==10, 0, train_data_y)

    tensor_x = torch.stack([torch.FloatTensor(i) for i in train_data_x])  # transform to torch tensors
    tensor_y = torch.stack([torch.FloatTensor(i) for i in train_data_y])

    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y.squeeze())  # create your datset
    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle=True)  # create your dataloader


    ####

    test_data_x = test_data['X'].swapaxes(2, 3).swapaxes(1, 2).swapaxes(0, 1).swapaxes(2, 3).swapaxes(1, 2)
    test_data_y = test_data['y']
    test_data_y=np.where(test_data_y==10, 0, test_data_y)


    tensor_x_test = torch.stack([torch.FloatTensor(i) for i in test_data_x])  # transform to torch tensors
    tensor_y_test = torch.stack([torch.FloatTensor(i) for i in test_data_y])

    my_dataset = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test.squeeze())  # create your datset
    test_loader = torch.utils.data.DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle=True)

########################################
# NETWORK

architecture="64x2_128x2_256x3_512x8_L512x2"

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.c1=nn.Conv2d(3, 64, 3, padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.c2=nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.mp1=nn.MaxPool2d(2)

        self.c3=nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.c4=nn.Conv2d(128,128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d(2)

        self.c5=nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.c6=nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.c7=nn.Conv2d(256, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.mp3=nn.MaxPool2d(2)

        self.c8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.c9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.c10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.c11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.mp4 = nn.MaxPool2d(2)

        self.c12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.c13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.c14 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.c15 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn15 = nn.BatchNorm2d(512)
        self.mp5 = nn.MaxPool2d(2)


        self.l1 = nn.Linear(512, 512)
        self.l2 =nn.Linear(512, 512)
        self.l3 = nn.Linear(512,10)
        self.d1=nn.Dropout()
        self.d2 = nn.Dropout()


    def forward(self, x):

        output = f.relu(self.bn1(self.c1(x)))
        output = f.relu(self.bn2(self.c2(output)))
        output = self.mp1(output)

        output = f.relu(self.bn3(self.c3(output)))
        output = f.relu(self.bn4(self.c4(output)))
        output = self.mp2(output)

        output = f.relu(self.bn5(self.c5(output)))
        output = f.relu(self.bn6(self.c6(output)))
        output = f.relu(self.bn7(self.c7(output)))
        output = self.mp3(output)


        output = f.relu(self.bn8(self.c8(output)))
        output = f.relu(self.bn9(self.c9(output)))
        output = f.relu(self.bn10(self.c10(output)))
        output = f.relu(self.bn11(self.c11(output)))
        output = self.mp4(output)
        output = f.relu(self.bn12(self.c12(output)))
        output = f.relu(self.bn13(self.c13(output)))
        output = f.relu(self.bn14(self.c14(output)))
        output = f.relu(self.bn15(self.c15(output)))
        output = self.mp5(output)


        output = self.d1(output)
        output=output.view(-1, 512)
        output = f.relu(self.l1(output))
        output = self.d2(output)
        output = f.relu(self.l2(output))

        return output

    ###########################################3

def evaluate():
    #print('Prediction when network is forced to predict')
    vgg.eval()
    correct = 0
    total = 0
    for j, data in enumerate(test_loader):
        #pdb.set_trace()
        images, labels = data
        images =images.to(device)
        predicted_prob = vgg.forward(images) #images.view(-1,28*28)

        # _,prediction = torch.max(predicted_prob.data, 1)
        # #prediction = prediction.cpu().numpy()

        predicted=np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()

    # print(str(correct) +" "+ str(total))
    #pdb.set_trace()
    accuracy=100 * float(correct) / total
    print("accuracy: %.2f %%" % (accuracy))
    return accuracy

    ###########################

vgg=VGG().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(vgg.parameters(), lr=0.001)

def weights_init(m):
    if isinstance(m, nn.Conv2d):

        init.xavier_normal_(m.weight.data)

        if m.bias is not None:
            init.normal_(m.bias.data)

#vgg.apply(weights_init)

stop = 0;epoch = 0;best_accuracy = 0;entry = np.zeros(3);best_model = -1; early_stopping=100
while (stop < early_stopping):
    epoch = epoch + 1
    #print (epoch)
    for i, data in enumerate(train_loader):
        #print (i)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        #inputs, labels = inputs, labels


        # m = np.rollaxis(inputs[0].cpu().detach().numpy(), 0, 3)
        # m=m/255.0
        # plt.imshow(m)
        # plt.show()
        # print(labels)


        optimizer.zero_grad()
        outputs = vgg(inputs)
        labels=labels.long()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # if i % 100==0:
        #    print (i)
        #   print (loss.item())
        # print (i)
    print(loss.item())
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
        best_model=vgg.state_dict()
        best_optim=optimizer.state_dict()
        torch.save({'model_state_dict' : best_model, 'optimizer_state_dict': best_optim}, "models/%s_%s_rel_bn_drop_trainval_modelopt%.1f_epo:%d_acc:%.2f" % (dataset, architecture, trainval_perc, epoch, best_accuracy))

    print("\n")
#write

    filename="%s_test_conv_relu_bn_drop_trainval%.1f_%s.txt" % (dataset, trainval_perc, architecture)
    entry[0]=accuracy; entry[1]=loss
    with open(filename, "a+") as file:
        file.write(",".join(map(str, entry))+"\n")