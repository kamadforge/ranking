import keras
import keras.backend as K
<<<<<<< HEAD
#import torch as K

import pdb

import numpy as np
from scipy.misc import logsumexp

=======
import pdb

import numpy as np
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78

def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
<<<<<<< HEAD
    # x2 = K.expand_dims(K.sum(K.square(X), axis=1), 1)
    # dists = x2 + K.transpose(x2) - 2*K.dot(X, K.transpose(X))
    # return dists

    X=X[0]
    x2 = np.expand_dims(np.sum(np.square(X), axis=1), 1)
    dists = x2 + np.transpose(x2) - 2*np.dot(X, np.transpose(X))
    return dists

def get_shape(x):
    #dims = K.cast( K.shape(x)[1], K.floatx() )
    #N    = K.cast( K.shape(x)[0], K.floatx() )
    dims=x[0].shape[1]
    N=x[0].shape[0]
=======
    x2 = K.expand_dims(K.sum(K.square(X), axis=1), 1)
    dists = x2 + K.transpose(x2) - 2*K.dot(X, K.transpose(X))
    return dists

def get_shape(x):
    dims = K.cast( K.shape(x)[1], K.floatx() ) 
    N    = K.cast( K.shape(x)[0], K.floatx() )
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78
    return dims, N

#K get upper bound for MI
def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
<<<<<<< HEAD
    #dims, N = get_shape(x)
    dims, N = x[0].shape[1], x[0].shape[0]
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*np.log(2*np.pi*var)
    lprobs = logsumexp(-dists2, axis=1) - np.log(N) - normconst
    h = -np.mean(lprobs)
=======
    dims, N = get_shape(x)
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*K.log(2*np.pi*var)
    lprobs = K.logsumexp(-dists2, axis=1) - K.log(N) - normconst
    h = -K.mean(lprobs)
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78
    return dims/2 + h

#get lower bound for MI
def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    dims, N = get_shape(x)
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)

