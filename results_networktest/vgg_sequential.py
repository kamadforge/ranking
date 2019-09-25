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

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"
print(device)


###########################################################
# DATA

dataset = "cifar10"
trainval_perc=1
BATCH_SIZE=100

trainval_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10('data', train=False, download=True, transform = transforms.ToTensor())


train_size = int(trainval_perc * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


########################################
# NETWORK

architecture="64x2_128x2_256x3_512x8_L512x2"


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()

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

vgg=VGG("VGG16").to(device)
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


        optimizer.zero_grad()
        outputs = vgg(inputs)
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