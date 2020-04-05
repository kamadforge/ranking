#creates a parameter file for grid search for different compression methods for vgg

from sklearn.model_selection import  ParameterGrid

file = open("parameters_vgg_full.txt", 'w')

25,25,65,80,201,158,159,460,450,490,470,465,465,450

pruned_architectures=ParameterGrid({'c1':[25], 'c2':[25], 'c3':[65], 'c4':[80], 'c5':[201], 'c6':[158, 138], 'c7':[159, 139], 'c8':[460, 440], 'c9':[450, 430], 'c10':[490], 'c11':[470, 450], 'c12':[465, 445], 'c13':[465, 445], 'c14':[450]})
#pruned_architectures=ParameterGrid({'c1':[25], 'c2':[25], 'c3':[65], 'c4':[80], 'c5':[201], 'c6':[158, 178], 'c7':[159, 179], 'c8':[460, 480], 'c9':[450, 470], 'c10':[490], 'c11':[470, 490], 'c12':[465, 485], 'c13':[465, 485], 'c14':[450, 480]})


for pruned_arch in pruned_architectures:
    file.write("%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i\n" % (pruned_arch['c1'], pruned_arch['c2'], pruned_arch['c3'], pruned_arch['c4'], pruned_arch['c5'], pruned_arch['c6'], pruned_arch['c7'], pruned_arch['c8'], pruned_arch['c9'], pruned_arch['c10'], pruned_arch['c11'], pruned_arch['c12'], pruned_arch['c13'], pruned_arch['c14'] ))