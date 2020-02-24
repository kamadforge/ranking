#read files from results_comparison/vgg/94.34/

file1_path="vgg/94.34/cmp_fisher.txt"
file2_path="vgg/94.34/cmp_l1l2filt.txt"

file1=open(file1_path)
file2=open(file2_path)

acc_dict={}


for line in file2:
    if "l1" in line:
        next(file2)
        arch_line=next(file2)
        arch=[int(a) for a in arch_line[1:-2].split(",")]
        for i in range(17):
            next(file2)
        acc_line=next(file2)
        print(acc_line[25:31])
        acc=float(acc_line[25:31])
        acc_dict[arch_line[1:-2]] = []
        acc_dict[arch_line[1:-2]].append(('l1', acc))

    if "l2" in line:
        next(file2)
        arch_line = next(file2)
        arch = [int(a) for a in arch_line[1:-2].split(",")]
        for i in range(17):
            next(file2)
        acc_line = next(file2)
        print(acc_line[25:31])
        acc = float(acc_line[25:31])
        acc_dict[arch_line[1:-2]].append(('l2', acc))

    if "filter" in line:
        next(file2)
        arch_line = next(file2)
        arch = [int(a) for a in arch_line[1:-2].split(",")]
        for i in range(18):
            next(file2)
        acc_line = next(file2)
        print(acc_line[25:31])
        acc = float(acc_line[25:31])
        acc_dict[arch_line[1:-2]].append(('filter', acc))

print("end file2")

for line in file1:
    if "fisher" in line:
        next(file1)
        arch_line=next(file1)
        arch=[int(a) for a in arch_line[1:-2].split(",")]
        for i in range(19):
            next(file1)
        acc_line=next(file1)
        acc=float(acc_line[25:31])
        acc_dict[arch_line[1:-2]].append(('fisher', acc))

end=1
print("end file1")

end=1

bestmeth_arr={'filter':0,'fisher':0,'l1':0,'l2':0}
for elem in acc_dict:
    if len(acc_dict[elem])==4:
        print(elem)
        best_acc=0
        best_meth=""
        for tuple in acc_dict[elem]:
            if tuple[1]>best_acc:
                best_acc=tuple[1]
                best_meth=tuple[0]
            print(tuple)
            print(best_acc, best_meth)
        bestmeth_arr[best_meth]+=1

print(bestmeth_arr)



