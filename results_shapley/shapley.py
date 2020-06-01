import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.special



#for i in range(1000):
#    line=next(f)

#############################################3
# READ ONLY DATA
# not sampled, we take all the combinations of size 1, then all the combinations of size 2, etc.

# reads into dic 0,6 : 98.51
# 6: 98.82
# 7: 98.17
# 8: 98.57
# 9: 99.02
# 0,1: 97.65
# 0,2: 98.83
# 0,3: 98.63
# 0,4: 98.80
# 0,5: 98.81
# 0,6: 98.51
# 0,7: 96.75
# 0,8: 98.17
# 0,9: 98.73

def readdata_notsampled(original_accuracy):
    f = open(file)
    dict = {(): 0}
    for line in f:
        linesplit=line.strip().split(":")
        tup=tuple(int(i) for i in linesplit[0].split(","))
        acc=float(linesplit[1])
        dict[tup]=original_accuracy-acc
        print(tup, acc)
    f.close()
    return dict

#same as above, just the output from the MPI cluster is a little bit different
def readdata_notsampled_cluster():
    f = open(file)
    dict = {(): 0}
    for l in f:
        a=next(f)
        a2=next(f)
        a3=next(f)
        a3b=next(f)
        line=next(f)
        if "avin" in line:
            line=next(f)
        linesplit = line.strip().split(":")
        tup = tuple(int(i) for i in linesplit[0].split(","))
        acc = float(linesplit[1])
        dict[tup] = 99.27 - acc
        print(tup, acc)

    f.close()
    return dict

############################33
# just a regular but with tensot workd

# test accuracy: 93.23 %
# tensor([8])
# test accuracy: 98.57 %
# tensor([9])
# test accuracy: 83.25 %
# tensor([0, 1])

def readdata():

    f = open(file)
    dict = {(): 0}
    for line in f:
        if "tensor" in line:
            combination=line[8:-3] #tensor([1, 2, 4, 5, 7, 8, 9])
            #print(combination)
            tup = tuple(int(i) for i in combination.split(","))
            accuracy_line=next(f)
            acc=float(accuracy_line[15:20])
            dict[tup]=100.0-acc
    return dict

#

# tensor([62])
# Test Lossds: 0.001 | Acc: 100.000% (10000/10000)
# 62: 100.00
#
# tensor([63])
# Test Lossds: 0.001 | Acc: 99.990% (9999/10000)
# 63: 99.99
#
# tensor([0, 1])
# Test Lossds: 0.001 | Acc: 99.970% (9997/10000)
# 0,1: 99.97
#
# tensor([0, 2])
# Test Lossds: 0.001 | Acc: 100.000% (10000/10000)
# 0,2: 100.00

#########################################
#
def readdata_vggformat():

    f = open(file)
    dict = {(): 0}
    for line in f:
        if "tensor" in line:
            line=next(f)
            if "Test Lossds" in line:
                line=next(f)
                linesplit = line.strip().split(":")
                tup = tuple(int(i) for i in linesplit[0].split(","))
                acc = float(linesplit[1])
                dict[tup] = 100.0 - acc
                print(tup, acc)
    f.close()
    return dict

    # f = open(file)
    # dict = {(): 0}
    # for line in f:
    #     if "tensor" in line:
    #         combination=line[8:-3] #tensor([1, 2, 4, 5, 7, 8, 9])
    #         #print(combination)
    #         tup = tuple(int(i) for i in combination.split(","))
    #         accuracy_line=next(f)
    #         acc=float(accuracy_line[26:31])
    #         dict[tup]=100.0-acc
    # return dict

# treanforms data into the standard dict from readdata_notsampled method

# tensor([1, 2, 4, 5, 6, 8, 9])
# test accuracy: 99.18 %
# test accuracy: 99.18 %
# test accuracy: 99.12 %
# ('Averaged accuracy: ', 99.16000000000001)
# tensor([1, 2, 4, 5, 7, 8, 9])
# test accuracy: 99.19 %
# test accuracy: 99.08 %
# test accuracy: 99.14 %

import re

def readdata_averaged():

    f = open(file)
    dict = {(): 0}
    for line in f:
        if "tensor" in line:
            combination=line[8:-3] #tensor([1, 2, 4, 5, 7, 8, 9])
            #print(combination)
            tup = tuple(int(i) for i in combination.split(","))
            next(f); next(f); next(f); accuracy_line=next(f)
            aver_acc=re.findall("\d+\.\d+", accuracy_line)
            aver_acc=np.round(float(aver_acc[0]),2)
            dict[tup]=aver_acc
    return dict


########################################################3
# READ DATA AND COMPUTE SV
# sampled, we take random samples of combinations  and random numbner from that comination and compute sv
# to see what was its contribution

def readdata_sampled_compute_sv():
    f = open(file, "r+")
    dict = {(): 0}

    # for index in range(5):
    #     line = f.next()
    #     print ("Line No %d - %s" % (index, line))

    shapley_samples=np.zeros(100)
    shapley_samples_num=np.zeros(100)
    samples_num=0
    for line in f:
        samples_num+=1
        num=int(line)
        print(num)
        line=f.readline()
        print(line)
        linesplit=line.strip().split(":")
        line2=f.readline()
        linesplit2=line2.strip().split(":")
        print("%s, %s" %(linesplit[1], linesplit2[1]))
        shap_diff=float(linesplit[1])-float(linesplit2[1])
        shapley_samples[num]+=shap_diff
        shapley_samples_num+=1

        #tup=tuple(int(i) for i in linesplit[0].split(","))
        #acc=float(linesplit[1])
        #dict[tup]=99.34-acc
        #print(tup, acc)
    shap_array=[]
    for i in range(len(shapley_samples)):
        print(i)
        shapley_value=shapley_samples[i]/shapley_samples_num[i]
        print(shapley_value)
        shap_array.append(shapley_value)

    np.set_printoptions(precision=7)
    np.set_printoptions(suppress=True)
    sorted_indices = np.argsort(shap_array)
    shap_array = np.array(shap_array)
    print(sorted_indices)
    print(shap_array[sorted_indices])
    print("Number of samples: "+ str(samples_num))
    print(",".join(str(i) for i in np.argsort(shap_array)))

#readdata_sampled_compute_sv()

#############################################
# COMPUTE SHAPLEY ONLY
# full shapley 1

#the characteristic function values are arbitrary for this toy example
def full_shapley_1():
    dict={(1,2,3): 6, (1,3) : 5, (2,3) : 5, (1,2): 2, (2,): 2, (3,) : 2, (1,): 1, ():0}
    elements=[1,2,3]

    for elem in elements:
        print(elem)
        sum=0
        for a in dict:
            if elem in a:
                elem_list=list(a)
                elem_list.remove(elem)
                a_removed=tuple(elem_list)

                sfact=math.factorial(len(a_removed))
                vsmin1fact=math.factorial(len(elements)-len(a_removed)-1)
                sum+=sfact*vsmin1fact*(dict[a]-dict[a_removed])

                #print(a, dict[a])
                #print(a_removed, dict[a_removed])
        shapley=sum/math.factorial(len(elements))
        print(shapley)

#########################################3
#shapley 1b

#computing a complete shapley value for an arbitrary dict

def full_shapley_1b(dict):

    elem_num = 10
    elements = np.arange(elem_num)
    shapleys=[]

    for elem in elements:
        print(elem)
        sum=0
        for a in dict:
            if elem in a:
                elem_list=list(a)
                elem_list.remove(elem)
                a_removed=tuple(elem_list)

                sfact=math.factorial(len(a_removed))
                vsmin1fact=math.factorial(len(elements)-len(a_removed)-1)
                sum+=sfact*vsmin1fact*(dict[a]-dict[a_removed])

                #print(a, dict[a])
                #print(a_removed, dict[a_removed])
        shapley=sum/math.factorial(len(elements))
        print(shapley)
        shapleys.append(shapley)

    print(np.argsort(shapleys)[::-1])
    print(np.sort(shapleys)[::-1])



###################################################################3
# full shapley 2

# def full_shapley_2():
#     #dict={(1,2,3): 6, (1,3) : 5, (1,2): 2, (2,3): 5, (2,): 2, (3,) : 2, (1,): 1, ():0}
#     elements=[1,2,3]
#
#
#     for elem in elements: # for each element we want to compute SV of
#         sum=0
#         permutations = itertools.permutations(elements)
#         for perm in permutations: # we look at all the permutations
#                 perm_list = list(perm)
#                 ind=perm.index(elem)
#                 del perm_list[ind+1:]
#                 perm_list.sort()
#                 perm_tuple=tuple(perm_list)
#                 perm_list.remove(elem)
#                 removed_perm_tuple=tuple(perm_list)
#                 val=dict[perm_tuple]-dict[removed_perm_tuple]
#                 sum+=val
#                 print(val)
#         shap=sum/math.factorial(len(elements))
#         print(shap)

####################################

#computing the partial shapley value for the subsets of size k
def full_shapley_2(dict):
    #dict={(1,2,3): 6, (1,3) : 5, (1,2): 2, (2,3): 5, (2,): 2, (3,) : 2, (1,): 1, ():0}
    elem_num=10
    elements=np.arange(elem_num)
    k=5 #the size of the subset

    shapleys=[]

    for elem in elements: # for each element we want to compute SV of
        coalitions_vals=[]; coalitions_vals_normal=[]
        print("elem: ", elem)
        sum=0
        permutations = itertools.permutations(elements)
        for perm in permutations: # we look at all the permutations
                perm_list = list(perm)
                ind=perm.index(elem)
                if ind==k:
                    del perm_list[ind+1:]
                    perm_list.sort() #sort to check characteristic function
                    perm_tuple=tuple(perm_list)
                    perm_list.remove(elem)
                    removed_perm_tuple=tuple(perm_list)
                    val=dict[perm_tuple]-dict[removed_perm_tuple]
                    sum+=val
                    coalitions_vals.append(val)
                    #print(val)
        #it is not part of the core shapley value computation NOTE
        #it just looks how the ch
        #for i in coalitions_vals:
        #    coalitions_vals_normal.append(i)
        #    coalitions_vals_normal.append(-i)
        #print("var: ", np.var(coalitions_vals_normal))
        #print("mean: ", np.mean(coalitions_vals_normal))
        #plt.hist(coalitions_vals)
        #plt.show()

        # number of permutations with the same group of nodes before and after ind
        repetitions_num = math.factorial(k)*math.factorial(elem_num-k-1)
        partial_shap=sum*repetitions_num #to jeszcze podzielic przez number of this unique sets before and after
        #partial_shap=sum/math.factorial(len(elements))
        partial_shap=partial_shap/math.factorial(len(elements))
        shap = sum / math.factorial(len(elements)-1)
        print("shap: ", partial_shap)
        shapleys.append(partial_shap)
    np.set_printoptions(precision=2)
    shsort=np.sort(shapleys)[::-1]
    argshsort=np.argsort(shapleys)[::-1]
    print(shsort)
    print(argshsort)
    for i in range(len(shsort)):
        print("%d (%.1f), " % (argshsort[i], shsort[i]), end='')
    #plt.hist(shapleys)
    #plt.show()

#example:

# Out[1]: (1, 2, 3, 4, 5, 6, 7, 8, 0, 9)
# perm_list
# Out[2]: [1, 2, 3, 4, 5, 6, 7, 8, 0]
# perm_list
# Out[3]: [1, 2, 3, 4, 5, 6, 7, 8]
# removed_perm_tuple
# Out[4]: (1, 2, 3, 4, 5, 6, 7, 8)
# perm_tuple
# Out[5]: (0, 1, 2, 3, 4, 5, 6, 7, 8)
# val
# Out[6]: 10.349999999999994
# dict[perm_tuple]
# Out[7]: 87.72
# dict[removed_perm_tuple]
# Out[8]: 77.37



###########################################################
# sampled shapley, "full" perms
# (in quotes because we may not have computed all the perms, but we compute them sequentially
# to get all of them, e.g. all perms of size 1, all perms of size 2, etc

#for each node we want to compute Shapley value:
#we get a random permutation and find that node (we count the subset from the beginning up to that node)
# remove it and chceck the difference if both the subsets are present

# works on such dics
# 8: 98.57
# 9: 99.02
# 0,1: 97.65
# 0,2: 98.83
# 0,3: 98.63

def shapley_samp(dict_passed, nodesnum, samples_num):
    print("Partial Random Shapley")
    dict = dict_passed

    #permutations = list(itertools.permutations(elements))
    shap_array=[]
    elements_num=nodesnum
    for elem in range(elements_num): # for each element we want to compute SV of
        sum=0
        dict_elems=0
        print(elem)
        for i in range(samples_num):
            perm=np.random.permutation(elements_num).tolist()
            #print(perm)
            # we look at all the permutations
            ind=perm.index(elem)
            del perm[ind+1:]
            perm.sort()
            perm_tuple=tuple(perm)
            perm.remove(elem)
            removed_perm_tuple=tuple(perm)
            if perm_tuple in dict and removed_perm_tuple in dict:
                val=dict[perm_tuple]-dict[removed_perm_tuple]
                sum+=val
                #print(val)
                dict_elems+=1
        #print("sum: %.2f, perms: %d" % (sum,dict_elems))
        shap=sum/dict_elems
        print("shap: %.2f" % shap)
        shap_array.append(shap)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    sorted_indices=np.argsort(shap_array)[::-1]
    sorted_array=np.sort(shap_array)[::-1]
    x_pos = [i for i, _ in enumerate(sorted_indices)]
    plt.rcParams.update({'font.size': 10})
    plt.bar(x_pos[:25], sorted_array[:25])
    plt.xticks(x_pos[:25], sorted_indices[:25])
    plt.xlabel("Filter numbers (layer: %s)" % layer)
    plt.ylabel("Shapley value")
    ax = plt.gca()
    #ax2=ax.twiny()
    #ax2.xaxis.set_label_position("bottom")
    #ax2.spines["bottom"].set_position(("axes", -0.15))
    labels = ax.get_xticklabels()
    #plt.locator_params(axis='x', nbins=50)
    #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    #plt.show()
    #plt.savefig(dirfile+"/shapley_%s.png" % layer)



    #savefig(dirfile)


    shap_array=np.array(shap_array)
    print(sorted_indices)
    print(shap_array[sorted_indices])
    shapley_rank=",".join(str(i) for i in sorted_indices)
    print(shapley_rank)
    #file_tosave.write("[%s],\n" %shapley_rank)


    return sorted_indices

##########################3
# check differences betwen different ks







###########################################################
# sampled permutation sampled shapley

# given characteristic function values (complete or incomplete) we simply sample permuations,
# and see the contribution of giving element for that permutation

def shapley_samp_perm_samp(dict_passed, elements_num, iterations):
    print("Partial nodes and permutations Random Shapley")
    dict = dict_passed

    # permutations = list(itertools.permutations(elements))
    shap_array = []
    for elem in range(elements_num):  # for each element we want to compute SV of
        sum = 0
        dict_elems = 0
        print(elem)
        for i in range(iterations):
            perm = np.random.permutation(elements_num).tolist()
            # print(perm)
            # we look at all the permutations
            ind = perm.index(elem)
            del perm[ind + 1:]
            perm.sort()
            perm_tuple = tuple(perm)
            perm.remove(elem)
            removed_perm_tuple = tuple(perm)
            if perm_tuple in dict and removed_perm_tuple in dict:
                val = dict[perm_tuple] - dict[removed_perm_tuple]
                sum += val
                # print(val)
                dict_elems += 1
        # print("sum: %.2f, perms: %d" % (sum,dict_elems))
        shap = sum / dict_elems
        print("shap: %.2f" % shap)
        shap_array.append(shap)
    print(np.argsort(shap_array))
    print(",".join(str(i) for i in np.argsort(shap_array)))

# FILES
vgg_layername=["c3"]

for layer in vgg_layername:

    #file="/home/kamil/Dropbox/Current_research/python_tests/results_compression/combinations/94.34/zeroing_0.2val/combinations_vgg_"+layer+".0.out"
    #file="/home/kamil/Dropbox/Current_research/python_tests/results_compression/combinations/fashionmnist/90.04/zeroing_trainval1_val0.2/combinations_fashionmnist_trainval1_val0.2_"+layer+"_.0.out"
    #file="/home/kamil/Dropbox/Current_research/python_tests/results_compression/combinations/99.27/zeroing_trainval1_val0.2/combinations_trainval1_val0.2_"+layer+"_.0.out"
    #file="/home/kamil/Dropbox/Current_research/python_tests/results_shapley/combinations/combinations_pruning_mnist_conv:10_conv:50_fc:800_fc:500_rel_bn_epo:103_acc:99.37/combinations_pruning_mnist_conv:10_conv:50_fc:800_fc:500_rel_bn_epo:103_acc:99.37_"+layer+".weight.txt"
    #file="/home/kamil/Dropbox/Current_research/python_tests/results_compression/combinations/99.06/zeroing/combinations_trainval0.9_"+layer+"_.0.out"
    #file = "/home/kamil/Dropbox/Current_research/python_tests/results_compression/combinations/99.27/additive_noise/combinations_9927_addnoise_"+layer+".0.out"
    file="/home/kamil/Desktop/Dropbox/Current_research/ranking/results_shapley/combinations/mnist/99.27/combinations_pruning_mnist_mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27/combinations_pruning_mnist_mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27_%s.weight.txt" % layer
    #file="/home/kamil/Dropbox/Current_research/python_tests/results_shapley/results/combinations_fashionmnist/combinations_pruning_mnist_fashionmnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:62_acc:90.04_%s.weight.txt" % layer
    #file="/home/kamil/Dropbox/Current_research/python_tests/results_shapley/results/vgg_93.92/combinations_pruning_cifar_vgg16_module.c3.weight.txt"
    #file="/home/kamil/Dropbox/Current_research/python_tests/results_shapley/results/vgg_93.92/tesst_vgg15.0.out"
    #dirfile=os.path.split(file)[0]
    #save_textfile=os.path.join(dirfile, "shapley.txt")
    #file_tosave=open(save_textfile, "a+")
    lenet_filternums={'c1': 10, 'c3': 20, 'c5': 100, 'f6': 25}
    #vgg_filternums={"c1":64, "c2":64, "c3":128, "c4":128, "c5":256, "c6":256, "c7":256, "c8":512, "c9":512, "c10":512, "c11":512, "c12":512, "c13":512, "l1":512, "l2":512}
    file ="/home/kamil/Desktop/Dropbox/Current_research/featimp_dp/code/combinations/model[0].txt"

#NOTSAMPLED

# to run
#1. change file name
#2. shapley_samp - choose number of filters

#dict=readdata_notsampled_clu0ster()


########################################
# parse and save the dictionaries of combinations

#import operator
#
# original_accuracy=93.25
# for i in range(0,13):
#     file = "/home/kamil/Dropbox/Current_research/python_tests/results_compression/combinations/93.25/zeroing/combinations_pruning_cifar_vgg16_module.c"+str(i+1)+".weight.txt"
#
#     dict=readdata_notsampled(original_accuracy)
#     # IMP sorted_indices_filters=shapley_samp(dict, vgg_filternums[i], 20000)
#
#
#     sorted_indices_filters = sorted(dict.items(), key=operator.itemgetter(1))
#     sorted_indices_filters=np.array(sorted_indices_filters)[:,0]
#     sorted_indices_filters=[i[0] for i in sorted_indices_filters if i]
#
#
#     name="results/vgg_"+str(original_accuracy)+"/shapley_"+str(original_accuracy)+"_vgg16_"+str(i+1)
#     np.save(name,  sorted_indices_filters)


###############################################



    original_accuracy=59#99.27
    dict=readdata_notsampled(original_accuracy)
    dict[tuple(np.arange(10))]=original_accuracy
    print(dict)

    #shapley_samp_perm_samp(dict, 20, 100)
    shapley_samp(dict, 20, 100)


    #sorted_indices_filters=shapley_samp(dict, lenet_filternums[layer], 10000) #IMP #approximation lenet
    #sorted_indices_filters=shapley_samp(dict, vgg_filternums[layer], 100000) #IMP

#if having only one value per node IMP
#sorted_indices_filters = sorted(dict.items(), key=operator.itemgetter(1))
#sorted_indices_filters=np.array(sorted_indices_filters)[:,0]
#sorted_indices_filters=[i[0] for i in sorted_indices_filters if i]

#sorted_indices_filters=shapley_samp(dict, 512, 20000) #IMP



#name="results/vgg_"+str(original_accuracy)+"/shapley_"+str(original_accuracy)+"_vgg16_14"
#np.save(name,  sorted_indices_filters)


#partial_rand_shapley(dict)
#partial_shapley(dict)

## SAMPLED

# to run
#1. change file name

#readdata_sampled_compute_sv()
