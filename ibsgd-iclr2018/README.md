# Code for On the Information Bottleneck Theory of Deep Learning

* (KER)`demo.py` is a simple script showing how to compute MI between X and Y, where Y = f(X) + Noise.

* (KER) `IBnet_ComputeMI.ipynb` is a jupyter notebook that loads the data files created by `IBnet_SaveActivations.ipynb`, computes MI values, and does the infoplane plots and SNR plots. Note, for the full results of the paper, MI values and SNR was averaged for 50 distinct trials; MI and SNR for individual runs can vary substantially.

* (KER) `IBnet_SaveActivations.ipynb` is a jupyter notebook that recreates the network and data from https://github.com/ravidziv/IDNNs and saves activations, weight norms, &c. for each epoch for a single trial.

* (KER)  kde - kernel density estimation done for something

* (PYT) kde_torch - the pytorch equivof kde

* (KER) loggingreporter.py - logs the numbers from the training, etc.

* (KER) `MNIST_ComputeMI.ipynb` is a jupyter notebook that loads the data files, computes MI values, and does the infoplane plots and SNR plots for data created using `MNIST_SaveActivations.ipynb`.

* (ker (required)/PYT) `MNIST_ComputeMI_torch.ipynb` is a jupyter notebook that loads the data files, computes MI values, and does the infoplane plots and SNR plots for data created using `MNIST_SaveActivations.ipynb`.

* (ker/PYT) `MNIST_ComputeMI_torch_self.ipynb` is a jupyter notebook that loads the data files, computes MI values, and does the infoplane plots and SNR plots for data created using `MNIST_SaveActivations.ipynb`.


* (KER) `MNIST_SaveActivations.ipynb` is a jupyter notebook that trains on MNIST and saves (in a data directory) activations when run on test set inputs (as well as weight norms, &c.) for each epoch.

* (BOTH) simplebinme.py

* (KER) utils.py



Andrew Michael Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan Daniel Tracey, David Daniel Cox, On the Information Bottleneck Theory of Deep Learning, *ICLR 2018*.
