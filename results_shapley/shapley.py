import math
import itertools
import pickle
import random
import numpy as np

#file="/home/kamil/Dropbox/Current_research/python_tests/results_networktest/results/combinations_pruning_mnist_conv1:20_rel_bn.txt"
file="../results_networktest/results_copied/combinations_pruning_mnist_mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19_c3.weight.txt"
file="/home/kamil/Dropbox/Current_research/python_tests/results_networktest/results/combinations_pruning_mnist_mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19/combinations_pruning_mnist_mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19_c1.weight.txt"
file="/local_data/kadamczewski/Dropbox/Current_research/python_tests/results_networktest/results/combinations_pruning_mnist_mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27/combinations_pruning_mnist_mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27_c5.weight.txt"


#for i in range(1000):
#    line=next(f)


def readdata_notsampled():
    f = open(file)
    dict = {(): 0}
    for line in f:
        linesplit=line.strip().split(":")
        tup=tuple(int(i) for i in linesplit[0].split(","))
        acc=float(linesplit[1])
        dict[tup]=99.27-acc
        print(tup, acc)
    f.close()
    return dict

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
    print(np.argsort(shap_array))
    print("Number of samples: "+ str(samples_num))
    print(",".join(str(i) for i in np.argsort(shap_array)))

#readdata_sampled_compute_sv()

#############################################
# METHODS

#############################################################3
# full shplaey 1

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

###################################################################3
# full shapley 2
def full_shapley_2():
    dict={(1,2,3): 6, (1,3) : 5, (1,2): 2, (2,3): 5, (2,): 2, (3,) : 2, (1,): 1, ():0}
    elements=[1,2,3]


    for elem in elements: # for each element we want to compute SV of
        sum=0
        permutations = itertools.permutations(elements)
        for perm in permutations: # we look at all the permutations
                perm_list = list(perm)
                ind=perm.index(elem)
                del perm_list[ind+1:]
                perm_list.sort()
                perm_tuple=tuple(perm_list)
                perm_list.remove(elem)
                removed_perm_tuple=tuple(perm_list)
                val=dict[perm_tuple]-dict[removed_perm_tuple]
                sum+=val
                print(val)
        shap=sum/math.factorial(len(elements))
        print(shap)


###########################################################
# sampled shapley, "full" perms 1 (may not be all perms but not sampled)
def partial_shapley(dict_passed):
    print("Partial Shapley")
    dict = dict_passed
    elements = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    permutations = itertools.permutations(elements)
    #np.random_permutation(20)

    for elem in elements: # for each element we want to compute SV of
        sum=0
        dict_elems=0
        print(elem)
        #for perm in permutations: # we look at all the permutations
        #        perm_list = list(perm)

        for i in range(10):
            perm_list=np.random_permutation(20)

            ind=perm_list.index(elem)
            del perm_list[ind+1:]
            perm_list.sort()
            perm_tuple=tuple(perm_list)
            perm_list.remove(elem)
            removed_perm_tuple=tuple(perm_list)
            if perm_tuple in dict and removed_perm_tuple in dict:
                val=dict[perm_tuple]-dict[removed_perm_tuple]
                sum+=val
                #print(val)
                dict_elems+=1
        shap=sum/dict_elems
        print(shap)

###########################################################
# sampled shapley, "full" perms 2
def shapley_samp(dict_passed):
    print("Partial Random Shapley")
    dict = dict_passed

    #permutations = list(itertools.permutations(elements))
    shap_array=[]
    elements_num=25
    for elem in range(elements_num): # for each element we want to compute SV of
        sum=0
        dict_elems=0
        print(elem)
        for i in range(10000):
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
    print(np.argsort(shap_array))
    print(",".join(str(i) for i in np.argsort(shap_array)))

    ###########################################################
    # sampled permutation sampled shapley 
    def shapley_samp_perm_samp(dict_passed):
        print("Partial nodes and permutations Random Shapley")
        dict = dict_passed

        # permutations = list(itertools.permutations(elements))
        shap_array = []
        elements_num = 10
        for elem in range(elements_num):  # for each element we want to compute SV of
            sum = 0
            dict_elems = 0
            print(elem)
            for i in range(10000):
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



#dict=readdata_notsampled()
#shapley_samp(dict)
#partial_rand_shapley(dict)

readdata_sampled_compute_sv()
