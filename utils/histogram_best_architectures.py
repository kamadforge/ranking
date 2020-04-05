#takes the outputs from cluster
#gathers the accuracy for each rank method and for each architecture
#makes histogram which method is the best for a given architecture

import os
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt



#archs= open('parameters.txt')
archs= open('parameters_vgg_full.txt')

dataset='mnist'

#path='/home/kamil/Dropbox/Current_research/ranking/cluster_output/switch_lenet_%s/competition' % dataset

#files=os.listdir(path)

#configfiles = glob.glob('/home/kamil/Dropbox/Current_research/ranking/cluster_output/switch_lenet_mnist/competition/**/*4,6,40,17*', recursive=True)
#print(configfiles)

# for path in Path(path).rglob("5,7"):
#     print(path.name)


points={}
points['l1']=0; points['l2']=0; points['fisher']=0; points['switch_100']=0; points['switch_150']=0; points['switch_300']=0;  points['switch_500']=0;
points['switch_point_multiple']=0; points['switch_point_single']=0;

for line in archs:
    line=line.strip()
    print('*'*30)
    print(line)
    configfiles = glob.glob('/home/kamil/Dropbox/Current_research/ranking/cluster_output/switch_cifar/competition/**/*%s*' % line, recursive=True)
    #print(configfiles)
    max_method = '';
    max_acc = 0
    for file_path in configfiles:
        print (file_path)
        if 'fisher' in file_path:
            method='fisher'
        if 'l1' in file_path:
            method='l1'
        if 'l2' in file_path:
            method='l2'
        if 'integral_150' in file_path:
            method='switch_150'
        if 'integral_100' in file_path:
            method='switch_100'
        if 'integral_300' in file_path:
            method='switch_300'
        if 'integral_500' in file_path:
            method='switch_500'
        if 'multiple' in file_path:
            method='switch_point_multiple'
        if 'point' in file_path:
            method='switch_point_single'

        file=open(file_path)

        for line in file:
            #if "Final:" in line:
            if "acc: " in line:
                #print(line)
                #acc=float(line[7:12])
                acc=float(line[6:11])
                #print(acc)
                if acc>max_acc:
                    max_acc=acc
                    max_method=method
    print(acc)
    print(max_method)
    points[max_method]+=1

print(points)



# for path in Path(path).rglob('*4,6,45,17*'):
#     print(path)



for method in ['integral_300', 'l1', 'l2', 'fisher']:
    method_path=os.path.join(path, method)


label=[]
val=[]
i=0
for key in points:
    if points[key]!=0:
        label.append(key)
        val.append(points[key])
        i+=1

label=['L1-norm', 'L2-norm', 'Fisher', 'IS (150)', 'IS (300)', 'IS (500)', 'IS (mean)' ]

# val=[1,3,14, 10, 14, 76, 20]
# label=['L1-norm', 'L2-norm', 'Fisher', 'IS (20)', 'IS (50)', 'IS (100)', 'IS (mean)' ]

def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(i)
    plt.bar(index, val)
    #plt.xlabel('Methods', fontsize=11)
    plt.ylabel('Number of architectures', fontsize=17)
    plt.xticks(index, label, fontsize=13, rotation=15)
    #plt.xticks(index, label, fontsize=11, rotation=30)

    #plt.title('VGG (Cifar-10)', fontsize=20)
    #plt.title('LeNet-5 (MNIST)', fontsize=20)
    plt.title('LeNet-5 (MNIST)', fontsize=20)
    plt.tight_layout()
    #plt.show()
    plt.savefig("architectures_test_%s_all.png" % dataset)

plot_bar_x()