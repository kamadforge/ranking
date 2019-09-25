from __future__ import print_function
#%load_ext autoreload
#%autoreload 2
import os, pickle
from collections import defaultdict, OrderedDict


import numpy as np
#import keras.backend as K

import kde_torch
import simplebinmi
import pdb

import utils
trn, tst = utils.get_mnist()

# Which measure to plot
infoplane_measure = 'upper'
#infoplane_measure = 'bin'

DO_SAVE        = True    # Whether to save plots or just show them
DO_LOWER       = True    # (infoplane_measure == 'lower')   # Whether to compute lower bounds also
DO_BINNED      = True    #(infoplane_measure == 'bin')     # Whether to compute MI estimates based on binning

MAX_EPOCHS = 20      # Max number of epoch for which to compute mutual information measure
# MAX_EPOCHS = 1000
COLORBAR_MAX_EPOCHS = 10000

# Directories from which to load saved layer activity
ARCH = '1024-20-20-20'
#ARCH = '20-20-20-20-20-20'
#ARCH = '32-28-24-20-16-12'
#ARCH = '32-28-24-20-16-12-8-8'
DIR_TEMPLATE = '%%s_%s'%ARCH

# Functions to return upper and lower bounds on entropy of layer activity
#K what is entropy of a layer, computes
noise_variance = 1e-1                    # Added Gaussian noise variance



#kde.entropy_estimator_kl(activity, noise_variance)

#Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder

def entropy_func_upper(activity):
    return kde_torch.entropy_estimator_kl(activity, noise_variance)

def entropy_func_lower(activity):
    return kde_torch.entropy_estimator_bd(activity, noise_variance)

#entropy_func_upper = K.function([Klayer_activity,], [kde.entropy_estimator_kl(Klayer_activity, noise_variance),])
#entropy_func_lower = K.function([Klayer_activity,], [kde.entropy_estimator_bd(Klayer_activity, noise_variance),])

# nats to bits conversion factor
nats2bits = 1.0/np.log(2)


# Save indexes of tests data for each of the output classes
saved_labelixs = {}
for i in range(10):
    #tst.y: array([7, 2, 1, ..., 4, 5, 6] # labels for each example
    saved_labelixs[i] = tst.y == i
    #saved labels[0] contains the index numbers (chosen from 1 to 10000) where the label is 0 (True if it is, False if it son), similarly saved_labels[1], etc.

#share of ecah label among all the labels
labelprobs = np.mean(tst.Y, axis=0)

PLOT_LAYERS    = None     # Which layers to plot.  If None, all saved layers are plotted

# Data structure used to store results
measures = OrderedDict()
measures['relu'] = {}
measures['tanh'] = {}

#####################################

for activation in measures.keys():
    cur_dir = 'rawdata/' + DIR_TEMPLATE % activation
    if not os.path.exists(cur_dir):
        print("Directory %s not found" % cur_dir)
        continue

    # Load files saved during each epoch, and compute MI measures of the activity in that epoch
    # for each epoch of the training get the test activations

    print('*** Doing %s ***' % cur_dir)
    for epochfile in sorted(os.listdir(cur_dir)):
        if not epochfile.startswith('epoch'):
            continue

        fname = cur_dir + "/" + epochfile
        with open(fname, 'rb') as f:
            d = pickle.load(f) #K file with the activations
            #K d has many keys
            # but we are looking only at d['data']['activity_tst'][lndx] where lndx is the layer index, it has 5 layers here


        epoch = d['epoch']
        if epoch in measures[activation]:  # Skip this epoch if its already been processed
            continue  # this is a trick to allow us to rerun this cell multiple times)

        if epoch > MAX_EPOCHS:
            continue

        print("Doing", fname)

        num_layers = len(d['data']['activity_tst'])

        if PLOT_LAYERS is None:
            PLOT_LAYERS = []
            for lndx in range(num_layers):
                # if d['data']['activity_tst'][lndx].shape[1] < 200 and lndx != num_layers - 1:
                PLOT_LAYERS.append(lndx)

        cepochdata = defaultdict(list)

        # here we compute entropies or mutual informations for each layer
        for lndx in range(num_layers):
            activity = d['data']['activity_tst'][lndx]  # K activity for each layer, they're of size 1024x10000, 20x10000, etc.
            # pdb.set_trace()

            # Compute marginal entropies
            h_upper = entropy_func_upper([activity, ])
            if DO_LOWER:
                h_lower = entropy_func_lower([activity, ])

            # Layer activity given input. This is simply the entropy of the Gaussian noise
            hM_given_X = kde_torch.kde_condentropy(activity, noise_variance)

            # Compute conditional entropies of layer activity given output
            hM_given_Y_upper = 0.
            for i in range(10):
                hcond_upper = entropy_func_upper([activity[saved_labelixs[i], :], ])
                hM_given_Y_upper += labelprobs[i] * hcond_upper

            if DO_LOWER:
                hM_given_Y_lower = 0.
                for i in range(10):
                    hcond_lower = entropy_func_lower([activity[saved_labelixs[i], :], ])
                    hM_given_Y_lower += labelprobs[i] * hcond_lower

            # # It's also possible to treat the last layer probabilistically. Here is the
            # # code to do so. Should only be applied when lndx == num_layers - 1

            # ps = activity.mean(axis=0)
            # h_lower = h_upper = sum([-p*np.log(p) for p in ps if p != 0])

            # x = -activity * np.log(activity)
            # x[activity == 0] = 0.
            # hM_given_X = np.mean(x.sum(axis=1))

            # hM_given_Y=0.
            # for i in range(10):
            #     ixs = tst.y[::subsample] == i
            #     ps = activity[ixs,:].mean(axis=0)
            #     hcond = sum([-p*np.log(p) for p in ps if p != 0])
            #     prob = np.mean(ixs)
            #     hM_given_Y += l * hcond
            # hM_given_Y_lower = hM_given_Y_upper = hM_given_Y
            # del hM_given_Y

            # K why do we subtract
            cepochdata['MI_XM_upper'].append(nats2bits * (h_upper - hM_given_X))
            cepochdata['MI_YM_upper'].append(nats2bits * (h_upper - hM_given_Y_upper))
            cepochdata['H_M_upper'].append(nats2bits * h_upper)

            pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
            cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])
            if DO_LOWER:  # Compute lower bounds
                cepochdata['MI_XM_lower'].append(nats2bits * (h_lower - hM_given_X))
                cepochdata['MI_YM_lower'].append(nats2bits * (h_lower - hM_given_Y_lower))
                cepochdata['H_M_lower'].append(nats2bits * h_lower)
                pstr += ' | lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                cepochdata['MI_XM_lower'][-1], cepochdata['MI_YM_lower'][-1])

            if DO_BINNED:  # Compute binner estimates
                binxm, binym = simplebinmi.bin_calc_information2(saved_labelixs, activity, 0.5)
                cepochdata['MI_XM_bin'].append(nats2bits * binxm)
                cepochdata['MI_YM_bin'].append(nats2bits * binym)
                pstr += ' | bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                cepochdata['MI_XM_bin'][-1], cepochdata['MI_YM_bin'][-1])

            print('- Layer %d %s' % (lndx, pstr))

        measures[activation][epoch] = cepochdata

############################################

#% matplotlib
#inline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style('darkgrid')

# PLOT_LAYERS = [0,1,2,3,4] # [1,2,3]
# PLOT_LAYERS = [0,1,2,3]
# PLOT_LAYERS = [0,1,2,3]
plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 2)
for actndx, (activation, vals) in enumerate(measures.items()):
    epochs = sorted(vals.keys())
    if not len(epochs):
        continue

    plt.subplot(gs[0, actndx])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        xmvalsU = np.array([vals[epoch]['H_M_upper'][layerid] for epoch in epochs])
        if DO_LOWER:
            xmvalsL = np.array([vals[epoch]['H_M_lower'][layerid] for epoch in epochs])
        plt.plot(epochs, xmvalsU, label='Layer %d' % layerid)
        # plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(activation)
    plt.ylabel('H(M)')

    plt.subplot(gs[1, actndx])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        # for epoch in epochs:
        #    print('her',epoch, measures[activation][epoch]['MI_XM_upper'])
        xmvalsU = np.array([vals[epoch]['MI_XM_upper'][layerid] for epoch in epochs])
        if DO_LOWER:
            xmvalsL = np.array([vals[epoch]['MI_XM_lower'][layerid] for epoch in epochs])
        plt.plot(epochs, xmvalsU, label='Layer %d' % layerid)
        # plt.errorbar(epochs, (xmvalsL + xmvalsU)/2,xmvalsU - xmvalsL, label='Layer %d'%layerid)
    plt.xscale('log')
    plt.ylabel('I(X;M)')

    plt.subplot(gs[2, actndx])
    for lndx, layerid in enumerate(PLOT_LAYERS):
        ymvalsU = np.array([vals[epoch]['MI_YM_upper'][layerid] for epoch in epochs])
        if DO_LOWER:
            ymvalsL = np.array([vals[epoch]['MI_YM_lower'][layerid] for epoch in epochs])
        plt.plot(epochs, ymvalsU, label='Layer %d' % layerid)
    plt.xscale('log')
    plt.ylabel('MI(Y;M)')

    if DO_BINNED:
        plt.subplot(gs[3, actndx])
        for lndx, layerid in enumerate(PLOT_LAYERS):
            hbinnedvals = np.array([vals[epoch]['MI_XM_bin'][layerid] for epoch in epochs])
            plt.semilogx(epochs, hbinnedvals, label='Layer %d' % layerid)
        plt.xlabel('Epoch')
        # plt.ylabel("H'(M)")
        plt.ylabel("I(X;M)bin")
        # plt.yscale('log')

    if actndx == 0:
        plt.legend(loc='lower right')

plt.tight_layout()

if DO_SAVE:
    plt.savefig('plots/summaries', bbox_inches='tight')


########

print("info")

max_epoch = max((max(vals.keys()) if len(vals) else 0) for vals in measures.values())
sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
sm._A = []

fig = plt.figure(figsize=(10, 5))
for actndx, (activation, vals) in enumerate(measures.items()):
    epochs = sorted(vals.keys())
    if not len(epochs):
        continue
    plt.subplot(1, 2, actndx + 1)
    for epoch in epochs:
        c = sm.to_rgba(epoch)
        xmvals = np.array(vals[epoch]['MI_XM_' + infoplane_measure])[PLOT_LAYERS]
        ymvals = np.array(vals[epoch]['MI_YM_' + infoplane_measure])[PLOT_LAYERS]

        plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
        plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in PLOT_LAYERS], edgecolor='none', zorder=2)

    plt.ylim([0, 3.5])
    plt.xlim([0, 14])
    plt.xlabel('I(X;M)')
    plt.ylabel('I(Y;M)')
    plt.title(activation)

cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
plt.colorbar(sm, label='Epoch', cax=cbaxes)
plt.tight_layout()


if DO_SAVE:
    plt.savefig('plots/' + DIR_TEMPLATE % ('infoplane_' + ARCH), bbox_inches='tight')
