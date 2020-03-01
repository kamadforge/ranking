# VI using implicit gradients
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions import Gamma

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

class Model(nn.Module):
    #I'm going to define my own Model here following how I generated this dataset

    def __init__(self, input_dim, hidden_dim, W1, b_1, W2, b_2, num_samps_for_switch):
    # def __init__(self, input_dim, hidden_dim):
        super(Model, self).__init__()

        self.W1 = W1
        self.b1 = b_1
        self.W2 = W2
        self.b2 = b_2
        self.hidden_dim = hidden_dim
        # self.output_dim = W2.size(1) # output dimension
        self.parameter = Parameter(-1e-10*torch.ones(hidden_dim),requires_grad=True) # this parameter lies
        self.num_samps_for_switch = num_samps_for_switch

    def forward(self, x):

        pre_activation = torch.mm(x, self.W1)
        shifted_pre_activation = pre_activation - self.b1
        phi = F.softplus(self.parameter)

        if any(torch.isnan(phi)):
            print("some Phis are NaN")
        # it looks like too large values are making softplus-transformed values very large and returns NaN.
        # this occurs when optimizing with a large step size (or/and with a high momentum value)

        #
        # """directly use mean of Dir RV."""
        # S = phi/torch.sum(phi)
        #
        # x = shifted_pre_activation * S
        # x = F.relu(x)
        # x = torch.mm(x, self.W2) + self.b2
        # label = torch.sigmoid(x)
        #
        # avg_S = S
        # avg_label = label
        # labelstack =[]
        # Sstack =[]

        """ draw Gamma RVs using phi and 1 """
        num_samps = self.num_samps_for_switch
        concentration_param = phi.view(-1,1).repeat(1,num_samps)
        beta_param = torch.ones(concentration_param.size())
        #Gamma has two parameters, concentration and beta, all of them are copied to 200,150 matrix
        Gamma_obj = Gamma(concentration_param, beta_param)
        gamma_samps = Gamma_obj.rsample() #200, 150, hidden_dim x samples_num

        if any(torch.sum(gamma_samps,0)==0):
            print("sum of gamma samps are zero!")
        else:
            Sstack = gamma_samps / torch.sum(gamma_samps, 0) #1dim - number of neurons (200), 2dim - samples (150)

        # Sstack -switch, for output of the network (say 200) we used to have switch 200, now we have samples (150 of them), sowe have switch which is (200, 150)        #

        x_samps = torch.einsum("ij,jk -> ijk",(shifted_pre_activation, Sstack))
        x_samps = F.relu(x_samps)
        x_out = torch.einsum("bjk, j -> bk", (x_samps, torch.squeeze(self.W2))) + self.b2
        labelstack = torch.sigmoid(x_out) #100,200 100- sa

        # avg_label = torch.mean(labelstack,1)
        # avg_S = torch.mean(Sstack,1)

        # return avg_label, avg_S, labelstack, Sstack
        return labelstack, phi #(100, 150) #output 1-dim for neural network (100,10,150) #phi - number of parameters

# def loss_function(prediction, true_y, S, alpha_0, hidden_dim, how_many_samps, annealing_rate):
def loss_function(prediction, true_y, phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate):

    BCE = F.binary_cross_entropy(prediction, true_y, reduction='mean')

    # BCE_mat_avg = BCE*true_y.shape[0] # this is sum of log likeliehood of test data averaged over the posterior samples

    # # change this for-loop to something else.
    # BCE_mat = torch.zeros(prediction.shape[1])
    # for ind in torch.arange(0,prediction.shape[1]):
    #     BCE_mat[ind] = F.binary_cross_entropy(prediction[:,ind], true_y[:,ind], reduction='sum')
    #
    # BCE_mat_avg = torch.mean(BCE_mat) # posterior sample average (but sum across test datapoints)

    # return BCE_mat_avg

    # KLD term
    alpha_0 = torch.Tensor([alpha_0])
    hidden_dim = torch.Tensor([hidden_dim])

    trm1 = torch.lgamma(torch.sum(phi_cand)) - torch.lgamma(hidden_dim*alpha_0)
    trm2 = - torch.sum(torch.lgamma(phi_cand)) + hidden_dim*torch.lgamma(alpha_0)
    trm3 = torch.sum((phi_cand-alpha_0)*(torch.digamma(phi_cand)-torch.digamma(torch.sum(phi_cand))))

    KLD = trm1 + trm2 + trm3
    # annealing kl-divergence term is better

    return BCE + annealing_rate*KLD/how_many_samps


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
    # this is for hidden_dim = 20
    # hidden_dim = 20
    # input_dim = 100
    # how_many_samps = 2000
    # x_0 = np.load('x_0.npy')
    # x_1 = np.load('x_1.npy')
    # y_0 = np.load('y_0.npy')
    # y_1 = np.load('y_1.npy')
    #
    # W1 = np.load('W1.npy')
    # b_1 = np.load('b_1.npy')
    # W2 = np.load('W2.npy')
    # b_2 = np.load('b_2.npy')
    # trueSwitch = np.load('S.npy')

    # # preparing variational inference
    # input_dim = 100
    # alpha_0 = 0.1 # below 1 so that we encourage sparsity
    # num_samps_for_switch = 1000
    # model = Model(input_dim=input_dim, hidden_dim=hidden_dim, W1=torch.Tensor(W1), b_1=torch.Tensor(b_1), W2=torch.Tensor(W2), b_2=torch.Tensor(b_2), num_samps_for_switch=num_samps_for_switch)


    # this is for hidden_dim = 200
    input_dim = 500
    hidden_dim = 200
    how_many_samps = 2000

    # this is for hidden_dim = 500
    # input_dim = 1000
    # hidden_dim = 500
    # how_many_samps = 4000

    file_name = 'x_0' + '_hidden_dim=' + np.str(hidden_dim)
    x_0 = np.load(file_name+'.npy')

    file_name = 'x_1' + '_hidden_dim=' + np.str(hidden_dim)
    x_1 = np.load(file_name+'.npy')

    file_name = 'y_0' + '_hidden_dim=' + np.str(hidden_dim)
    y_0 = np.load(file_name+'.npy')

    file_name = 'y_1' + '_hidden_dim=' + np.str(hidden_dim)
    y_1 = np.load(file_name+'.npy')

    file_name = 'W1' + '_hidden_dim=' + np.str(hidden_dim)
    W1 = np.load(file_name+'.npy')

    file_name = 'W2' + '_hidden_dim=' + np.str(hidden_dim)
    W2 = np.load(file_name+'.npy')

    b_1 = np.load('b1' + '_hidden_dim=' + np.str(hidden_dim)+'.npy')
    b_1 = np.squeeze(b_1)
    b_2 = np.load('b2' + '_hidden_dim=' + np.str(hidden_dim) + '.npy')
    b_2 = np.squeeze(b_2)

    trueSwitch = np.load('S' + '_hidden_dim=' + np.str(hidden_dim) + '.npy')

    y, X = data_test_generate(x_0, x_1, y_0, y_1, how_many_samps)

    tic()

    # preparing variational inference
    alpha_0 = 0.01 # below 1 so that we encourage sparsity.
    # num_samps_for_switch = 1000
    num_samps_for_switch = 150
    model = Model(input_dim=input_dim, hidden_dim=hidden_dim, W1=torch.Tensor(W1), b_1=torch.Tensor(b_1), W2=torch.Tensor(W2), b_2=torch.Tensor(b_2), num_samps_for_switch=num_samps_for_switch)


    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    mini_batch_size = 100
    how_many_epochs = 150
    how_many_iter = np.int(how_many_samps/mini_batch_size)

    training_loss_per_epoch = np.zeros(how_many_epochs)

    annealing_steps = float(8000.*how_many_epochs)
    beta_func = lambda s: min(s, annealing_steps) / annealing_steps

    ############################################################################################3

    print('Starting Training')

    for epoch in range(how_many_epochs):  # loop over the dataset multiple times

        print('epoch number is ', epoch)
        running_loss = 0.0

        yTrain, xTrain= shuffle_data(y, X, how_many_samps)
        annealing_rate = beta_func(epoch)

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
            # outputs,S_tmp, labelstack, Sstack = model(torch.Tensor(inputs))
            outputs, phi_cand = model(torch.Tensor(inputs)) #100,10,150
            labels = torch.squeeze(torch.Tensor(labels))
            # loss = F.binary_cross_entropy(outputs, labels)
            # loss = loss_function(outputs, labels)
            # num_samps = 100
            # loss = loss_function(labelstack, labels.view(-1,1).repeat(1,num_samps_for_switch), S_tmp, alpha_0, hidden_dim, how_many_samps, annealing_rate)
            # loss = loss_function(outputs, labels)
            loss = loss_function(outputs, labels.view(-1, 1).repeat(1, num_samps_for_switch), phi_cand, alpha_0, hidden_dim, how_many_samps, annealing_rate)
            # loss = loss_function(outputs, labels, S_tmp, alpha_0, hidden_dim, how_many_samps, annealing_rate)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        # training_loss_per_epoch[epoch] = running_loss/how_many_samps
        training_loss_per_epoch[epoch] = running_loss

    print('Finished Training')

    # plt.figure(1)
    # plt.plot(training_loss_per_epoch)
    # plt.title('cross entropy loss as a function of epoch')
    # plt.show()

    estimated_params = list(model.parameters())
    # estimated_alphas = (F.softplus(torch.Tensor(estimated_params[0]))).detach().numpy()
    # estimated_Switch = estimated_alphas / np.sum(estimated_alphas)

    """ posterior mean over the switches """
    # num_samps_for_switch
    phi_est = F.softplus(torch.Tensor(estimated_params[0]))
    concentration_param = phi_est.view(-1, 1).repeat(1, 5000)
    # beta_param = torch.ones(self.hidden_dim,1).repeat(1,num_samps)
    beta_param = torch.ones(concentration_param.size())
    Gamma_obj = Gamma(concentration_param, beta_param)
    gamma_samps = Gamma_obj.rsample()
    Sstack = gamma_samps / torch.sum(gamma_samps, 0)
    avg_S = torch.mean(Sstack, 1)
    std_S = torch.std(Sstack, 1)
    posterior_mean_switch = avg_S.detach().numpy()
    posterior_std_switch = std_S.detach().numpy()

    print('true Switch is ', trueSwitch)
    print('estimated posterior mean of Switch is', posterior_mean_switch)

    toc()
    # print('estimated posterior mean of Switch is', estimated_Switch)

    f = plt.figure(2)
    plt.plot(np.arange(0, hidden_dim), trueSwitch, 'ko')
    plt.errorbar(np.arange(0, hidden_dim), posterior_mean_switch, yerr=posterior_std_switch, fmt='ro')
    # plt.plot(estimated_Switch, 'ro')
    # plt.plot(posterior_mean_switch, 'ro')
    plt.title('true Switch (black) vs estimated Switch (red)')
    plt.show()

    # fig_title =
    # f.savefig("posterior_mean_switch_without_sampling_hidden_dim_500_epoch_400.pdf")
    # f.savefig("posterior_mean_switch_with_sampling_hidden_dim_500_epoch_400.pdf")
    # f.savefig("posterior_mean_switch_with_sampling_hidden_dim_20_epoch_400.pdf")


if __name__ == '__main__':
    main()