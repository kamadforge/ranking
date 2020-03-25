from matplotlib import pyplot  as plt
import numpy as np


def read():
    filepath='/home/kamil/Dropbox/Current_research/python_tests/results_compression/results/architecture_pruning/curves/tes_50_norm.2.out'
    accuracies_20={}

    empty_lines=0
    with open(filepath) as file:
        lines=list(file)

    for i in range(len(lines)):
        line=lines[i]
        #print(line)
        if 'method' in line:
            key, accuracy, best_accuracy = -1, -1, -1
            empty_lines =0
            print('\n\n'+line)
        if 'prunedto' in line:
            key=line[9:]
        if 'accuracy' in line:
            accuracy=line[10:15]
            #print(lines[i+2])
            if 'Best' in lines[i+2]:
                best_accuracy=accuracy
        if len(line)==1:
            empty_lines+=1
        if empty_lines==5:
            accuracies_20[key]=best_accuracy
            print("%s" % (best_accuracy))
            key, accuracy, best_accuracy=-1, -1, -1
            empty_lines=0


    print(accuracies_20)




def plot():

    if dataset == 'cifar':
        # cofar/vgg

        compression = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
        accuracies = {'filter': [41.87, 73.92, 80.58, 87, 87.85, 89.97, 90.45],
                      'l1': [10.02, 25.19, 75.48, 74.44, 78.21, 87.37, 89.54],
                      'l2': [10.21, 14.14, 74.59, 74.44, 76.39, 85.01, 87.12],
                      }

        compression = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1 ]
        accuracies = {'filter': [31.75, 51.14, 57.52, 82.88, 87.49, 86.67],
                      'l1': [10.65, 21.3, 46.48, 78.21, 87.42,  87.30],
                      'l2': [10.4, 21.64, 45.07, 79.18, 87.42, 87.28],
                      'fisher': [10.4, 21.64, 45.07, 79.18, 87.42, 87.28],
                      }

    elif dataset=='mnist':

        # nmnist
        compression = [0.95, 0.90, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        accuracies = {'filter': [33.06, 40.17, 47.61, 50.05, 50.27, 72.32, 90.01, 91.95],
                      'l1': [18.73, 22.40, 41.42, 41.51, 44.47, 50.33, 74.45, 90.64],
                      'l2': [15.90, 27.75, 41.85, 43.64, 46.90, 48.40, 81.75, 89.44]}

    elif dataset == 'fashionmnist':

        #fashionmnist
        #16000
        #[3,4,20,5], [3,4,30,15], [3,8,40,5], [3, 10, 40, 20], [3, 12, 50, 20], [3,12,60,10] [5,12,60,20],  [7_12_60_20]
        #942, .. , .. , 3000 , 4000, 4800, 5600, 6484,
        compression=[0.95, 0.90, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        accuracies={ 'filter':[33.20, 35.98, 41.32, 61.43, 65.27, 67.83, 69.43, 75.27],
                      'l1': [26.17, 29.65, 25.34, 38.07, 37.28, 37.74, 57.81, 72.29],
                      'l2': [24.30, 27.31, 34.36, 54.38, 56.19, 56.59, 62.55, 66.36]}




    plt.rcParams.update({'font.size': 15})
    plt.plot(compression, accuracies['filter'])
    plt.plot(compression, accuracies['l1'])
    plt.plot(compression, accuracies['l2'])
    plt.legend(['Neuron Ranking', 'L1', 'L2'])
    plt.xlabel("Compression")
    plt.ylabel("Accuracies")
    plt.xticks(compression)
    plt.yticks([20,30,40,50,60,70,80,90])

    if dataset=='cifar':
        plt.title("VGG16 / CIFAR")
        plt.savefig('plots/compression_acc_plot_cifar.png')
    elif dataset=='mnist':
        plt.title("Lenet5 / MNIST")
        plt.savefig('plots/compression_acc_plot_mnist.png')
    elif dataset=="fashionmnist":
        plt.title("Lenet5 / FashionMNIST")
        plt.savefig('plots/compression_acc_plot_fashionmnist.png')
      #plt.yscale('log')
    #plt.xscale('log')


    #50
    accuracies_50=[]
    compression_l1_50=[]
    compression_l2_50=[]
    compression_filter_50=[]

    #100
    accuracies_100=[]
    compression_l1_100=[]
    compression_l2_100=[]
    compression_filter_100=[]


dataset='cifar'
#read()
plot()