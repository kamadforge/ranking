import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch
import torch.nn.functional as F




def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(a):
    return a*(a>0)

def trueModel(x, W1, b1, W2, b2, S):
    shifted_pre_activation = np.dot(x, W1) - b1.transpose() # W1 = input_dim by hidden_dim
    switched_pre_activation = shifted_pre_activation*S.transpose()
    switched_activation = relu(switched_pre_activation)
    y = sigmoid(np.dot(switched_activation,W2) + b2*np.ones((switched_activation.shape[0],1)))
    return y

# true Model for generating data
input_dim = 1000
hidden_dim = 500
n_datapoints = 4000

# inputs from two Gaussians
x_0 = np.random.multivariate_normal(np.zeros(input_dim), np.eye(input_dim), int(n_datapoints/2))
x_1 = np.random.multivariate_normal(2*np.ones(input_dim), 0.2*np.eye(input_dim), int(n_datapoints/2))

W1 = np.random.rand(input_dim, hidden_dim)
W2 = np.random.rand(hidden_dim,1)
b1 = np.random.rand(hidden_dim,1)
b2 = np.random.rand(1)

# true switch
alpha = 0.01
S = np.random.dirichlet(alpha*np.ones(hidden_dim))
print(S.shape)

logits = torch.randn(hidden_dim,)
print(logits)
# Sample soft categorical using reparametrization trick:
G=F.gumbel_softmax(logits, tau=0.005, hard=False)
G=G.numpy()
print(G)
print (G.shape)

# plt.figure(2)
# plt.plot(np.arange(0, hidden_dim), S, 'ko')
# plt.show()

# generate each labels
y_0 = trueModel(x_0, W1, b1, W2, b2, G)
y_1 = trueModel(x_1, W1, b1, W2, b2, G)

print(y_0.shape)
print(y_1.shape)

""" save all the files """
file_name = 'x_0' + '_hidden_dim=' + np.str(hidden_dim)
np.save(file_name, x_0)

file_name = 'x_1' + '_hidden_dim=' + np.str(hidden_dim)
np.save(file_name, x_1)

file_name = 'y_0' + '_hidden_dim=' + np.str(hidden_dim)
np.save(file_name, y_0)

file_name = 'y_1' + '_hidden_dim=' + np.str(hidden_dim)
np.save(file_name, y_1)

file_name = 'W1' + '_hidden_dim=' + np.str(hidden_dim)
np.save(file_name, W1)


file_name = 'W2' + '_hidden_dim=' + np.str(hidden_dim)
np.save(file_name, W2)


file_name = 'b1' + '_hidden_dim=' + np.str(hidden_dim)
np.save(file_name, b1)


file_name = 'b2' + '_hidden_dim=' + np.str(hidden_dim)
np.save(file_name, b2)

file_name = 'S' + '_hidden_dim=' + np.str(hidden_dim)
np.save(file_name, S)


################################################################

# print(x_0.shape) # n_datapoints by input_dim

# pre_activation = torch.mm(x, self.W1)
# shifted_pre_activation = pre_activation - self.b1
# phi = F.softplus(self.parameter)
#
# if any(torch.isnan(phi)):
#     print("some Phis are NaN")
# # it looks like too large values are making softplus-transformed values very large and returns NaN.
# # this occurs when optimizing with a large step size (or/and with a high momentum value)
#
#
# """directly use mean of Dir RV."""
# S = phi / torch.sum(phi)
#
# x = shifted_pre_activation * S
# x = F.relu(x)
# x = torch.mm(x, self.W2) + self.b2
# label = torch.sigmoid(x)
#
# avg_S = S
# avg_label = label
# labelstack = []
# Sstack = []


# """ load data, parameters, and true Switch """
# x_0 = np.load('x_0.npy')
# x_1 = np.load('x_1.npy')
# y_0 = np.load('y_0.npy')
# y_1 = np.load('y_1.npy')
# # produce test data from this data
# how_many_samps = 2000
# y, X = data_test_generate(x_0, x_1, y_0, y_1, how_many_samps)
#
# W1 = np.load('W1.npy')
# b_1 = np.load('b_1.npy')
# W2 = np.load('W2.npy')
# b_2 = np.load('b_2.npy')
# trueSwitch = np.load('S.npy')