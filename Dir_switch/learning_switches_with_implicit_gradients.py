# VI using implicit gradients
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions import Gamma

class Model(nn.Module):
    #I'm going to define my own Model here following how I generated this dataset

    def __init__(self, input_dim, hidden_dim, W1, b_1, W2, b_2):
    # def __init__(self, input_dim, hidden_dim):
        super(Model, self).__init__()

        self.W1 = W1
        self.b1 = b_1
        self.W2 = W2
        self.b2 = b_2
        self.hidden_dim = hidden_dim
        self.parameter = Parameter(-1e-10*torch.ones(hidden_dim),requires_grad=True) # this parameter lies

    def forward(self, x):

        pre_activation = torch.mm(x, self.W1)
        shifted_pre_activation = pre_activation - self.b1
        phi = F.softplus(self.parameter)

        """ draw Gamma RVs using phi and 1 """
        num_samps = 100
        Sstack = torch.zeros((self.hidden_dim, num_samps))
        labelstack = torch.zeros()
        for count in np.range(0,num_samps):
            Gamma_obj = Gamma(phi, 1)
            gamma_samps = Gamma_obj.rsample()
            S = gamma_samps/torch.sum(gamma_samps)
            Sstack[:,count] = S
            # """directly use mean of Dir RV."""
            # S = phi/torch.sum(phi)

            x = shifted_pre_activation * S
            x = F.relu(x)
            x = torch.mm(x, self.W2) + self.b2
            label = torch.sigmoid(x)

        return label,S


def loss_function(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps):
    BCE = F.binary_cross_entropy(prediction, true_y, reduction='sum')

    # KLD term
    # alpha_0 = torch.Tensor([alpha_0])
    # hidden_dim = torch.Tensor([hidden_dim])
    # trm1 = torch.lgamma(torch.sum(S)) - torch.lgamma(hidden_dim*alpha_0)
    # trm2 = - torch.sum(torch.lgamma(S)) + hidden_dim*torch.lgamma(alpha_0)
    # trm3 = torch.sum((S-alpha_0)*(torch.digamma(S)-torch.digamma(torch.sum(S))))
    # KLD = -(trm1 + trm2 + trm3)

    # return BCE + KLD
    return BCE

def data_test_generate(x_0, x_1, y_0, y_1, how_many_samps):

    x = np.concatenate((x_0, x_1))
    y = np.concatenate((y_0, y_1))

    idx = np.random.permutation(how_many_samps)
    shuffled_y = y[idx]
    shuffled_x = x[idx,:]

    return shuffled_y, shuffled_x


def shuffle_data(y,x,how_many_samps):
    idx = np.random.permutation(how_many_samps)
    shuffled_y = y[idx]
    shuffled_x = x[idx,:]
    return shuffled_y, shuffled_x

def main():

    """ load data, parameters, and true Switch """
    x_0 = np.load('x_0.npy')
    x_1 = np.load('x_1.npy')
    y_0 = np.load('y_0.npy')
    y_1 = np.load('y_1.npy')
    # produce test data from this data
    how_many_samps = 2000
    y, X = data_test_generate(x_0, x_1, y_0, y_1, how_many_samps)

    W1 = np.load('W1.npy')
    b_1 = np.load('b_1.npy')
    W2 = np.load('W2.npy')
    b_2 = np.load('b_2.npy')
    trueSwitch = np.load('S.npy')

    # preparing variational inference
    input_dim = 100
    hidden_dim = 20
    alpha_0 = 0.9 # below 1 so that we encourage sparsity
    model = Model(input_dim=input_dim, hidden_dim=hidden_dim, W1=torch.Tensor(W1), b_1=torch.Tensor(b_1), W2=torch.Tensor(W2), b_2=torch.Tensor(b_2))

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mini_batch_size = 50
    how_many_epochs = 200
    how_many_iter = np.int(how_many_samps/mini_batch_size)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        print('epoch number is ', epoch)
        running_loss = 0.0

        yTrain, xTrain= shuffle_data(y, X, how_many_samps)

        for i in range(how_many_iter):

            # print(i)
            # get the inputs
            inputs = xTrain[i*mini_batch_size:(i+1)*mini_batch_size,:]
            labels = yTrain[i*mini_batch_size:(i+1)*mini_batch_size]
            # print(inputs.shape)
            # print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs,S_tmp = model(torch.Tensor(inputs))
            labels = torch.Tensor(labels)
            # loss = F.binary_cross_entropy(outputs, labels)
            # loss = loss_function(outputs, labels)
            loss = loss_function(outputs, labels, S_tmp, alpha_0, hidden_dim, how_many_samps)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        training_loss_per_epoch[epoch] = running_loss/how_many_samps

    print('Finished Training')

    plt.figure(1)
    plt.plot(training_loss_per_epoch)
    plt.title('cross entropy loss as a function of epoch')
    # plt.show()

    estimated_params = list(model.parameters())
    estimated_alphas = (F.softplus(torch.Tensor(estimated_params[0]))).detach().numpy()
    estimated_Switch = estimated_alphas / np.sum(estimated_alphas)
    # samples_from_exponential_distribution = np.random.exponential(1, (1, hidden_dim))
    # pre_normalization_S = estimated_alphas*np.squeeze(samples_from_exponential_distribution)
    #
    # estimated_Switch = pre_normalization_S / np.sum(pre_normalization_S)

    print('true Switch is ', trueSwitch)
    print('estimated posterior mean of Switch is', estimated_Switch)

    plt.figure(2)
    plt.plot(trueSwitch, 'ko')
    plt.plot(estimated_Switch, 'ro')
    plt.title('true Switch (black) vs estimated Switch (red)')
    plt.show()


if __name__ == '__main__':
    main()