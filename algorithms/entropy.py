import numpy as np
import scipy.special

def entropy(list):
    ent=0
    for i in list:
        ent+=i*np.log2(i)
    return -ent


# dist=[1/4]*4
dist=[0.5, 0.5]

print(entropy(dist))


def distortion_rate(distortion, length):
    print("Distortion rate")
    print(distortion, length)
    distortion_rate_sum=0
    for i in range(0,distortion+1):
        distortion_rate_sum+=scipy.special.binom(length, i)
    return distortion_rate_sum/np.power(2,length)



print(distortion_rate(3,7))
