#this version is different from the basic one such that each row can have a different prior (which here is anyway the same Mixture of gaussians)
#in fact this is though equivalent with the v.1 


#I. create BayesianNetwork 
#at first we just initialize the weights, that is we are running the classes init functions ( so we run the BayesianNetwork with sample true)
#-we get samples for mu and rho and make matrix of Gaussians (e.g. of size [400, 784] for the first layer, each entry is a separate Gaussian)
#-for each layer we also compute two of weight/bias and also weight and bias  mixture of gaussians (single one for each so four in total )

#II run forward and compute loss
#we need to get three things: outputs of size [2, 100, 10], log_prior [2] and
#run BayesianNetwork.sample_elbo
#So for each layer in bayesianBetwork we run that layer as BayesianLinear
#we simply sample weights from given atrix of Gaussians created in I and we simply run the regular forward step with ***F.linear(input, weight, bias)***
#and then we get log_prior and log_variational_posterior


#BayesianNetwork (1 instance)
#    BayesianLinear (num of layers instances)
#        Gaussian (one for the layer and one for biases, so two)
#----------------------

# import torch
# import torch.optim as optim
# from torch import nn, optim

# import torch.utils.data
# from torchvision import datasets, transforms

# import numpy as np
# import csv

#%matplotlib inline
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

writer = SummaryWriter()
sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")

import numpy as np

DEVICE="cpu"


###################################################333
# DATASETS


filename="mnist_test_3L_11.txt"
BATCH_SIZE = 100
# Download or load downloaded MNIST dataset
# shuffle data at every epoch
train_loader = torch.utils.data.DataLoader(
datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)

TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)
NUM_BATCHES = len(train_loader)
NUM_TEST_BATCHES = len(test_loader)

CLASSES = 10
TRAIN_EPOCHS = 20
SAMPLES = 2
TEST_SAMPLES = 10

PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])


#train_loader = torch.utils.data.DataLoader(
#                 dataset=train_set,
#                 batch_size=batch_size,
#                 shuffle=True)
#test_loader = torch.utils.data.DataLoader(
#                dataset=test_set,
#                batch_size=batch_size,
#                shuffle=False)

# class Lenet(nn.Module):
#     def __init__(self, nodesNum1, nodesNum2):
#         super(Lenet, self).__init__()
#
#         self.fc1=nn.Linear(784,nodesNum1)
#         self.fc2=nn.Linear(nodesNum1, nodesNum2)
#         self.fc3=nn.Linear(nodesNum2,10)
#
#
#     def forward(self, x):
#
#         x=x.view(-1,784)
#         output=self.fc1(x)
#         output=self.fc2(output)
#         output=self.fc3(output)
#         return output

##############################################3
# PROBABILISTIC CLASSES

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    #samples mean field variables for each weight
    #samples from standard normal (1,0) and extends it to multiply by sigma and adds mean
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon
    
    #simply log of gaussian distribution, returns one number, [400, 784] matrix of input, same as mus and same as sigmas
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum() 

class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super(ScaleMixtureGaussian, self).__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)

    #here we have constant two mus and two sigmas. in case of gaussians it's the whole matrix of mus and sigmas
    def log_prob(self, weight):
        prob1 = torch.exp(self.gaussian1.log_prob(weight)) #[400, 784]  Returns the log of the probability density/mass function evaluated at`value`.
        prob2 = torch.exp(self.gaussian2.log_prob(weight)) #[400, 784]
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum() #sum of all of [400, 784] points, returns one number weighted between two gaussians by pi

    

#simply creates a matrix of gaussians (input times output), where each entry is its own gaussian

#to be precise, it creates ONE Gaussian class with the parameters which are the matrix of mus and sigmas of the size of the layers)

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters (number of weights number) - initialize vector from uniform distribution for weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2)) #this is just for initialization, will change in the training
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))  #this is just for initialization, will change in the training
        
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters (number of weights number)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2)) #this is just for initialization, will change in the training
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4)) #this is just for initialization, will change in the training
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        # Prior distributions (scalar numbers) 
        #prior means it is fixed and it does not change over the training but we add to the loss the difference between this distribution and what we learn to make sure it is not too far off deom the distribution
        self.row_priors=[ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2) for i in range(0, out_features)] #e.g. size [400]
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):

        # THIS IS OUR POSTERIOR
        # we simply call the function sample from Gaussian which is weight
        if self.training or sample:
            weight = self.weight.sample() #we just sample and multiply it by mu and sigma given
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        #######
        # computing elements that will be later used for computing the loss

        if self.training or calculate_log_probs:
            #it takes weights and biases and does something with them
            modes=[]

<<<<<<< HEAD
            ##############
            # log prior

            #to the current log_prior adds weight_prior and bias prior
            #updates variational prior
            # for i in range(self.out_features):
            #     row_weight=weight[i, :]
            #     self.log_prior = self.log_prior+ self.weight_prior.log_prob(row_weight) # TAKES ROW WEIGHTS AND CHECKS IF IT"S NOT FAR FROM MIX GAUSS WEIGHT_PRIOR
            #
            # #two components of the KL in ELBO qlog(p) and qlog(q)
            # self.log_prior = self.log_prior + self.bias_prior.log_prob(bias) #returns one number

            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias) #returns one number, weight_prior is p(z)

            ##########################
            # variation prior
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias) #returns one number, weight is q(z)
=======
            #to the current log_prior adds weight_prior and bias prior
            #updates variational prior
            for i in range(self.out_features):
                row_weight=weight[i, :]
                self.log_prior = self.log_prior+ self.weight_prior.log_prob(row_weight) # TAKES ROW WEIGHTS AND CHECKS IF IT"S NOT FAR FROM MIX GAUSS WEIGHT_PRIOR

            #two components of the KL in ELBO qlog(p) and qlog(q)
            self.log_prior = self.log_prior + self.bias_prior.log_prob(bias) #returns one number
            #self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias) #returns one number
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias) #returns one number
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


    


class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BayesianLinear(28*28, 400)
        self.l2 = BayesianLinear(400, 400)
        self.l3 = BayesianLinear(400, 10)
    
    def forward(self, x, sample=False):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample)) #since l1 is defined as a BayesianLinear and we are in train mode, we will go to forward
        x = F.relu(self.l2(x, sample))
        x = F.log_softmax(self.l3(x, sample), dim=1)
        return x
    
    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior
    
    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l2.log_variational_posterior
    
    def sample_elbo(self, input, target, samples=SAMPLES, train=True): #the shape 2 is for two number of samples
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(DEVICE) #[2, 100, 10]
        log_priors = torch.zeros(samples).to(DEVICE) #2
        log_variational_posteriors = torch.zeros(samples).to(DEVICE) #2
        for i in range(samples):
            outputs[i] = self(input, sample=True) #it passes the input batch of images further, since we are in train mode, it will go to forward
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        #computes mean of the samples
        if train:
            log_prior = log_priors.mean()
            log_variational_posterior = log_variational_posteriors.mean()
            #nll is good when we have one hot vector so we know which one is a proper label and then we have a vector of probabiities
            #so we take the probabilities for the desired classes (and the should be high) and to make it work easier we take a log and negative so that high number means bad, higher loss 
            #for reduction sum we add the scalaers for each data point (number of points in a batch) and add them
            #the probability of each output class is given by softmax (but inputwe only choose the probability of the true label to compute the loss)
            negative_log_likelihood = F.nll_loss(outputs.mean(0), target, reduction="sum") # mean of [2,100,10]
            #this is equal to elbo that we sample according to the distribution q (paper, equation 2)
            #KL term + entropy term --- the value is for each batch (KL we divided by the number of batch, not bathc size, and nll sums over elements of the batch)
            loss = (log_variational_posterior - log_prior)/NUM_BATCHES + negative_log_likelihood #normal CE - KL(qlog (q/p), here we multiply by (-1), so it's negative ELBO and we minimize
            return loss, log_prior, log_variational_posterior, negative_log_likelihood
        else:
            #print(np.argmax(outputs.mean(0).detach().numpy(), axis=1))
            #print(np.argmax(outputs.mean(0).detach().numpy(), axis=1)==target.numpy())
            good=(np.argmax(outputs.mean(0).detach().numpy(), axis=1)==target.numpy()).sum()
            all=list(target.size())[0]
            return good, all

#at first we just initialize the weights, that is we are running the classes init functions ( so we run the BayesianNetwork with sample true)
net = BayesianNetwork().to(DEVICE)

def train(net, optimizer, epoch):
    net.train() #in the train mode we will move to forward functions for each class
    #if epoch == 0: # write initial distributions
    #    write_weight_histograms(epoch)
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    #for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target) #we pass data x and labels y
        loss.backward(retain_graph=True)
        optimizer.step()
        file=open("modes.txt", "a+")
        if (batch_idx % 50 == 0):
            test(net)
            net.train()
            l1_weights=net.l1.weight_mu
            modes=[]
            for i in range(400):
                modes.append(torch.mode(l1_weights[i, :])[0].detach().numpy())
            plt.hist(modes)
            file.write(str(batch_idx)+"\n"); 
            file.write(",".join(str(mode) for mode in modes)); 
            file.write("\n")
            #plt.show()
            file.close()


        #write_loss_scalars(epoch, batch_idx, loss, log_prior, log_variational_posterior, negative_log_likelihood)
    #write_weight_histograms(epoch+1)


def test(net):
    net.eval() #in the train mode we will move to forward functions for each class
    #if epoch == 0: # write initial distributions
    #    write_weight_histograms(epoch)
    good=0
    all=0
    for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        good_batch, all_batch =net.sample_elbo(data, target, train=False) #we pass data x and labels y
        good=good+good_batch
        all=all+all_batch
        #loss.backward()
        #optimizer.step()
    print("Accuracy: %.3f" % (good/all))

#EXPERIMENT

optimizer = optim.Adam(net.parameters())
for epoch in range(TRAIN_EPOCHS):
    train(net, optimizer, epoch)
    test(net)

        
for i in range(1,2000):
    with open(filename, "a+") as file:
        file.write("\n\nnumber of hidden nodes: "+str(i)+"\n")
    best_accuracy, num_epochs=run_experiment(11, i)
    with open(filename, "a+") as file:
        file.write("Best accuracy: %.2f in %d epochs" % (best_accuracy, num_epochs-15))