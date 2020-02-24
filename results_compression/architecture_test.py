import numpy as np
import sys

sys.path.insert(0, "/home/kamil/Desktop/Dropbox/Current_research/python_tests/results_networktest")

from nums_flops_params import get_flops_params

file=open("/home/kamil/Dropbox/Current_research/python_tests/results_comparison/fashionmnist_pruning.txt")

accuracy_data={}
accuracy_data["l1"] = [[],[],[],[],[],[],[],[], [], [], []]
accuracy_data["l2"] = [[],[],[],[],[],[],[],[], [], [], []]
accuracy_data["fisher"] = [[],[],[],[],[],[],[],[], [], [], []]
accuracy_data["filter"] = [[],[],[],[],[],[],[],[], [], [], []]

# for accd in accuracy_data:
#     accd=[[0]*5 for i in range(5)]
    # for i in range(8):
    #     accd.append([[0]])


for line in file:

    if "l1" in line:
        mode="l1"
    if "l2" in line:
        mode="l2"
    if "fisher" in line:
        mode="fisher"
    if "filter" in line:
        mode="filter"

    if "pruned" in line:
        architecture_line=line[9:]
        architecture_str=architecture_line.strip()
        next(file); next(file);
        accuracy_line=next(file)
        accuracy=float(float(accuracy_line[15:21]))

        #accuracy_data[mode][architecture_str]=accuracy

        architecture=[int(a) for a in architecture_line.strip().split("_")]



        filter_nums=architecture

        conv_filters = [[3, filter_nums[0], 5, 5], [2,2],[filter_nums[0], filter_nums[1], 5, 5], [2, 2],
                        [filter_nums[2]], [filter_nums[3]]]

        paddings=[0,0,0,0,0,0]
        strides=[1,2,1,2,1,1]

        input_shape = (1, 28, 28)
        layers_type = ['C', 'P', 'C', 'P', 'FC', 'FC']

        params=get_flops_params(layers_type, conv_filters, paddings, strides, input_shape)

        if params<1200:
            accuracy_data[mode][0].append(accuracy)
        elif params <2400:
            accuracy_data[mode][1].append(accuracy)
        elif params < 3600:
            accuracy_data[mode][2].append(accuracy)
        elif params < 4800:
            accuracy_data[mode][3].append(accuracy)
        elif params < 6000:
            accuracy_data[mode][4].append(accuracy)
        elif params < 7200:
            accuracy_data[mode][5].append(accuracy)
        elif params < 8400:
            accuracy_data[mode][6].append(accuracy)
        elif params < 9600:
            accuracy_data[mode][8].append(accuracy)
        elif params < 10800:
            accuracy_data[mode][9].append(accuracy)
        elif params < 12000:
            accuracy_data[mode][10].append(accuracy)

averages=[0]*10

averages={}
averages["l1"] = [0]*10
averages["l2"] = [0]*10
averages["fisher"] = [0]*10
averages["filter"] = [0]*10

for i in range(1,11):
    for mode in ["l1", "l2", "fisher", "filter"]:
        print("%s : %d" %(mode, i))
        averages[mode][i]=np.average(accuracy_data[mode][i])
        print(averages[mode][i])


######
# variance

# variances={}
# variances["l1"] = [0]*10
# variances["l2"] = [0]*10
# variances["fisher"] = [0]*10
# variances["filter"] = [0]*10
#
# for i in range(1,11):
#     for mode in ["l1", "l2", "fisher", "filter"]:
#         var=0
#         for j in range(len(accuracy_data[mode][i])):
#             var+=np.square(accuracy_data[mode]-averages[mode][i])
#


