import subprocess

for i in range(0, 10):
    #node_str="node=%i" % i
    subprocess.call(["python", "main.py", "train", "--model", "ARMLeNet5", "--dataset", "mnist", "--lambas", "[.1,.1,.1,.1]", "--optimizer", "adam", "--lr", "0.001", "--max_epoch", "15",  "--node", str(i), "--ar" ])