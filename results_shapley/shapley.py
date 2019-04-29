import math
import itertools
import pickle
import random
import numpy as np

file="/home/kamil/Documents/data/combinations_pruning_rel_bn_c3.weight.txt"
file="/home/kamil/Dropbox/Current_research/python_tests/results_networktest/results/combinations_pruning_fashionmnist_rel_bn_c1.weight_0.txt"

f= open(file)
dict={():0}
#for i in range(1000):
#    line=next(f)
for line in f:
    linesplit=line.strip().split(":")
    tup=tuple(int(i) for i in linesplit[0].split(","))
    acc=float(linesplit[1])
    dict[tup]=99.34-acc
    print(tup, acc)

###############################################################3
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
# partial shapley 2
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
# partial sampled shapley 2
def partial_rand_shapley(dict_passed):
    print("Partial Random Shapley")
    dict = dict_passed

    #permutations = list(itertools.permutations(elements))
    shap_array=[]
    for elem in range(20): # for each element we want to compute SV of
        sum=0
        dict_elems=0
        print(elem)
        for i in range(1000):
            perm=np.random.permutation(20).tolist()
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
    print(np.argsort(shap_array), -1)



partial_rand_shapley(dict)

