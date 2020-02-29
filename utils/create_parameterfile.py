from sklearn.model_selection import  ParameterGrid

file = open("parameters.txt", 'a+')

pruned_architectures=ParameterGrid({'c1':[3, 4, 5, 6, 7, 10], 'c3': [4, 6, 8, 10, 12, 20], 'f5': [20, 30, 40, 50, 60, 100], 'f6': [5, 10, 15, 20, 25]})

for pruned_arch in pruned_architectures:
    file.write("%i, %i, %i, %i\n" % (pruned_arch['c1'], pruned_arch['c3'], pruned_arch['f5'], pruned_arch['f6']))