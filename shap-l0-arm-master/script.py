import subprocess

# for i in range(0, 10):
#     #node_str="node=%i" % i
#     subprocess.call(["python", "main.py", "train", "--model", "ARMLeNet5", "--dataset", "mnist", "--lambas", "[.1,.1,.1,.1]", "--optimizer", "adam", "--lr", "0.001", "--max_epoch", "15",  "--node", str(i), "--ar" ])


for i in range(0, 10):
    with open("results/shapley_switches.txt", 'a') as file:
        file.write(str(i)+"\n")

    for m in range(100): #number of MC simulations
        #node_str="node=%i" % i
        subprocess.call(["python", "main.py", "test", "--model", "ARMLeNet5", "--dataset", "mnist", "--lambas", "[.1,.1,.1,.1]", "--load_file", "checkpoints/ARMLeNet5_2019-06-19 14:27:03/0.model",   "--node", str(i), "--use_t_in_testing", "False", "--node_remove", "False", "--ar" ])

    # subprocess.call(["python", "main.py", "test", "--model", "ARMLeNet5", "--dataset", "mnist", "--lambas", "[.1,.1,.1,.1]", "--load_file", "checkpoints/ARMLeNet5_2019-06-19 14:27:03/0.model",   "--node", str(i), "--use_t_in_testing", "False", "--node_remove", "False", "--ar" ])
    #
