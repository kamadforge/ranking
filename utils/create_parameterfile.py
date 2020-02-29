from sklearn.model_selection import  ParameterGrid

file = open("parameters.txt", 'w')

pruned_architectures=ParameterGrid({'c1':[4, 5, 6], 'c3': [6,7,8], 'f5': [35, 40, 45, 50], 'f6': [15, 17, 20]})

for pruned_arch in pruned_architectures:
    file.write("%i, %i, %i, %i\n" % (pruned_arch['c1'], pruned_arch['c3'], pruned_arch['f5'], pruned_arch['f6']))