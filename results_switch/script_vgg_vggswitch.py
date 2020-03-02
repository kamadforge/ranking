import subprocess
import argparse
arguments=argparse.ArgumentParser()
arguments.add_argument("--method", default="switch_point")
arguments.add_argument("--switch_samps", default=2, type=int)
arguments.add_argument("--epoch_num", default=3, type=int)
args=arguments.parse_args()

from main2vgg_switch_integral_work import main as main_integral
from main2vgg_switch_point import main as main_point #point estimate

import numpy as np
import os
import socket
import psutil
process = psutil.Process(os.getpid())
print("Memory: ", process.memory_info().rss/1024**2)  # in bytes

#######
# path stuff
cwd = os.getcwd()
if 'g0' in socket.gethostname() or 'p0' in socket.gethostname():
    #the cwd is where the sub file is so ranking/
    path_switch = os.path.join(cwd, "results_switch")
    path_main= cwd
else:
    #the cwd is results_compression
    parent_path = os.path.abspath('..')
    path_switch = cwd
    path_main= parent_path

print("v4")


#for alpha in [0.5, 1, 5, 10, 20, 50, 100, 0.05, 0.1]:
#for alpha in [0.01, 0.05, 0.1, 0.5, -0.5, -1, -2, -5, -10]:
#    for switch_init in [0.05, 0.1, 0.5, 1, 5]:
alpha=0.05; switch_init=0.05
epochs_num=args.epoch_num
dataset='cifar'
num_samps_for_switch=args.switch_samps
method=args.method

if method=="switch_point":
    file_path=path_switch+'/results/switch_data_%s_point_epochs_%i.npy' % (dataset, epochs_num)
elif method=="switch_integral":
    file_path=path_switch+'/results/switch_data_%s_integral_samps_%s_epochs_%i.npy' % (dataset, str(num_samps_for_switch), epochs_num)
    #file_path=os.path.join(path_main, 'results_switch/results/switch_data_%s_9032_integral_samps_%s_epochs_%i.npy' % (dataset, str(num_samps_for_switch), epochs_num))


if 1:
    if 1:
        switch_data={}; switch_data['combinationss'] = []; switch_data['switches']=[]
        for i in range(1,15):
            switch_layer='conv'+str(i)
            #subprocess.call(['/home/kadamczewski/miniconda3/envs/BayesianNeuralNetwork/bin/python', 'main2vgg_switch.py', switch_layer, str(alpha), str(switch_init)])
            #subprocess.call(['python', 'main2vgg_switch.py', switch_layer, str(alpha), str(switch_init)])
            if method=="switch_point":
                ranks, switches=main_point(switch_layer, epochs_num, num_samps_for_switch)
            elif method=="switch_integral":
                ranks, switches=main_integral(switch_layer, epochs_num, num_samps_for_switch)


            print("\n", '*'*30, "\nThe resulting ranks and switches")
            print(ranks, switches)
            switch_data['combinationss'].append(ranks);
            switch_data['switches'].append(switches)
            np.save(file_path, switch_data)
            print("Memory: ", process.memory_info().rss / 1024 ** 2)  # in bytes



