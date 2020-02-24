from torchvision import datasets, transforms
import torch
from torch import nn, optim
import torch.nn.functional as f
import numpy as np
from torch.nn.parameter import Parameter

device="cpu"
##############
BATCH_SIZE=100
# trainval_perc=1.0
#
# trainval_dataset = datasets.FashionMNIST('data', train=True, download=True,
#                                          # transform=transforms.Compose([transforms.ToTensor(),
#                                          # transforms.Normalize((0.1307,), (0.3081,))]),
#                                          transform=transforms.ToTensor())
#
# train_size = int(trainval_perc * len(trainval_dataset))
# val_size = len(trainval_dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
#
test_dataset = datasets.FashionMNIST('data/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False)


############################


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

    #################

nodesNum1, nodesNum2, nodesFc1, nodesFc2 = 10, 20, 100, 25
net = Lenet(nodesNum1, nodesNum2, nodesFc1, nodesFc2).to(device)

path="models/fashionmnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo62_acc90.04"
#path = "models/mnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo540_acc99.27"

net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'], strict=False)

##############
net.eval()
correct = 0
total = 0
for j, data in enumerate(test_loader):
    images, labels = data
    print(labels)
    images = images.to(device)
    predicted_prob = net.forward(images)  # images.view(-1,28*28)
    predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
    print(predicted)
    total += labels.size(0)
    correct += (predicted == labels.numpy()).sum().item()
accuracy = 100 * float(correct) / total
print("accuracy: %.2f %%" % (accuracy))