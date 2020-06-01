from sklearn.model_selection import ParameterGrid

params=ParameterGrid({"dataset": ["xor", "orange_skin","alternating","syn4", "syn5", "syn6"], "lr": [0.1, 0.01, 0.001, 0.0001], "epochs":[5,10,20,50,100]})

params=ParameterGrid({"dataset": ["xor", "orange_skin","additive_nonlinear", "syn4", "syn5", "syn6"], "lr": [0.1, 0.01], "epochs":[5,10,20,50], "num_Dir_samples": [20,50,100]})

file=open("parameters.txt", "w")
for p in params:
    file.write(f"{p['dataset']} {p['lr']} {p['epochs']} {p['num_Dir_samples']}\n")