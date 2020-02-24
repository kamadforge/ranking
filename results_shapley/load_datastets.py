from sklearn.datasets import load_boston
boston_dataset = load_boston()
import torch
from torch import nn, optim
from sklearn.datasets import load_boston
import numpy as np



boston=load_boston()

for elem in boston:
    print (elem)

x=boston.data
y=boston.target

tensor_x=torch.stack([torch.Tensor(i) for i in x])
tensor_y=torch.stack([torch.Tensor(i) for i in y])
dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

trainval_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])



print(boston.feature_names)

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()

        self.l=nn.Linear(13,1)

    def forward(self, x):

        out=self.l(x)

perc=1
train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [int(perc * len(trainval_dataset)),
                                                                              len(trainval_dataset) - int(
                                                                                  perc * len(trainval_dataset))])

train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

##################################

net = LinearRegression()
optimizer = optim.Adam(net.parameters(), lr=0.005)
criterion = nn.MSELoss()
#####################################

best_overall = 1000000000000000
for iter in range(10):
    print("-------------------------------Run: %d" % iter)
    best_testmseloss = 100000000000000
    not_improved = 0
    epoch = 0
    while (not_improved < 51):
        # for epoch in range(30):
        epoch += 1
        print("Epoch: %d" % epoch)
        for ind, (x, y) in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            forw = net(x)
            loss = criterion(forw, y)
            loss.backward()
            optimizer.step()

        i = 0
        test_mseloss = 0
        for ind, (x, y) in enumerate(test_dataset_loader):
            i += 1
            forw = net(x)
            mse = criterion(forw, y)
            test_mseloss += mse
        if ((test_mseloss / i) < best_testmseloss):
            best_testmseloss = test_mseloss / i
            print("test mse loss: %.3f" % (test_mseloss / i))
            not_improved = 0
        else:
            not_improved += 1
    best_sofar = float("%.2f" % best_testmseloss.detach().numpy())
    best_overall = np.minimum(best_overall, best_sofar)

#accuracies.append(best_overall)
#print(accuracies)
