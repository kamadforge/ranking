import numpy as np

filename='../results/shapley_switches.txt'

file=open(filename)

array=[]
for i in range(10):
    array.append([])

iter=0
for line in file:
    iter+=1
    if iter % 101 == 1:
        index=int(line)
    else:
        array[index].append(float(line))

switch_shap=np.mean(np.array(array), 1)
print(switch_shap)

sort=np.sort(switch_shap)[::-1]
argsort=np.argsort(switch_shap)[::-1]

print(sort)
print(argsort)

