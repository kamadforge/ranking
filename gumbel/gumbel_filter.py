import torch
import torch.nn. functional as functional
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

input=100
hidden=20
output=10
n_datapoints=4000




class Net(nn.Module):
    def __init__(self, input, hidden, output):
        super(Net, self).__init__()
        self.l1=nn.Linear(input, hidden)
        self.l2=nn.Linear(hidden, output)

        #self.param=nn.Parameter(torch.tensor(hidden).normal_(), requires_grad=True)
        self.gumbel = nn.Parameter(-1e-10*torch.ones(hidden),requires_grad=True) # this parameter lies


    def forward(self, x, mode):
        out=self.l1(x)

        if mode=="train":
            a = functional.gumbel_softmax(self.gumbel, tau=0.03, hard=True)
            #print(a)
        else:
            a = torch.zeros(hidden); a[2:8]=1
        out=out*a
        out=functional.relu(out)
        out=self.l2(out)
        out=torch.sigmoid(out)



        return out

###################

def print_model(model, full):
    if full==False:
        for name, param in model.named_parameters():
            print(name, param.shape)
    else:
        for name, param in model.named_parameters():
            print(name, param)

################


net=Net(input, hidden, output)


# inputs from two Gaussians
x_0 = torch.FloatTensor(np.random.multivariate_normal(np.zeros(input), np.eye(input), int(n_datapoints)))


#############################################3

mode="train"

if mode=='train':

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=0.001)


    #def train():
    #net.train()

    net_state_dict=torch.load("data/net")
    net.load_state_dict(net_state_dict)

    dataloader = torch.load("data/dataloader")

    for name, param in net.named_parameters():
        print(name, param.shape)
        if (name != "gumbel"):
            h = param.register_hook(lambda grad: grad * 0)  # double the gradient

    net.l1.bias.register_hook(lambda grad: grad * 0)

    for epoch in range(300):
        print(epoch)
        for data, label in dataloader:

            optimizer.zero_grad()
            output=net(data, mode)
            loss=criterion(output, label.squeeze(1))
            loss.backward()
            #print(net.l1.bias.grad)
            #print(net.l1.bias)
            #net.l1.bias.grad = torch.zeros_like(net.l1.bias.grad)  # instead of hook
            #print(net.gumbel.grad)

            optimizer.step()
            #print(net.l1.bias)

        print(loss)
        print(net.gumbel)
        print(functional.gumbel_softmax(net.gumbel, tau=0.93, hard=False))
        print(functional.gumbel_softmax(net.gumbel, tau=0.93, hard=True))
        #print(net.gumbel)
        #print((net.l1.weight.grad == 0).sum())  # the hook is automatically applied, here we just check the gradient



####################################

if mode=='test':
    def test():
        net.eval()
        out = net(x_0, mode)
        return out


    out=test()

    y_0 = torch.argmax(out, 1).unsqueeze(1) #already torch


    tensor_x = torch.stack([torch.Tensor(i) for i in x_0])  # transform to torch tensors

    my_dataset = data.TensorDataset(tensor_x, y_0)  # create your datset
    my_dataloader = data.DataLoader(my_dataset, batch_size=3)  # create your dataloader


    torch.save(my_dataloader, "data/dataloader")
    torch.save(net.state_dict(), "data/net")


    print_model(net, True)


