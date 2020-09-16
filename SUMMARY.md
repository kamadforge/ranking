COMPRESSION

- vgg_main2_tomodule - turning nn.Seqeuential to Module to train, loads the model, computes combinations, does everthing that
netwokr pruning for lenet plus vis and training from scratch
- lenet_network_pruning -
    # for four layer lenet:
    # -loads the models
    # -evaluates the model
    #- prunes the model
    # -retrains the model#
    # imports from and can compute combinations (and can also get combinations  add additive noise)
    - lenet_network_pruning_with combinations
- vgg_computeComb - computes the combinations with the pruned model for vgg
- magnitude_pruning.py - takes a model and computes the rank based on the weight magnitutes, that is using KL and L2 norms

functional:

- plot_compression_acc_curve.py - takes arrays of accuracies for different compression levels and plots

trials:
- iterative_pruning.py - attempts to implement iterative pruning
- distributions - tries pytorch distributions



old (possibly in archive_workingcode):

main  - original file
main2 - copied VGG to the file so that we don't need to import(it's using nn.Sequential)
main2.py gets 93.47% accuracy
main2_tomodule gets 93.89%
vgg_main2_tomodule_vggv1.py - older architecture with one less Linear and two more conv layers

-------------------------------------------------------------------------------


SWITCHES:

for vgg
-main2vgg_script_vggswitch.py -  a script to run a file multiple times for different arguments
-main2vgg_switch.py - runs switches but for vgg
-main2vgg_switch_v1.py - runs switches but for vgg but for the older version with one less fc and two more conv
-mnist.3L.conv.gpu_switch - run the program for switches and saves the swicthes values to the pt file
-plot_switches - makes a plot out of the pt file saved in the mnist_3L.conv.gpu_switch

for lenet
-results_switch/mnist.3L.conv.gpu_switch_working_really_workingversion.py


Steps:

1. Need to pretrain a model without switches

2. Load the pretrain model and run the training for a few epochs

The S (switch) vector gives you the probability distribution of neuron importances.

Example:

Use pretrained models provided on github:
"models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"
"models/fashionmnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:62_acc:90.04"


Notes:

to custom architecture add the switch function like that

output=self.c5(output)
if layer=='f5':
    output, Sprime = self.switch_func(output)
output=self.f6(output)

The function:

   def switch_func(self, output):
        phi = f.softplus(self.parameter)
        S = phi / torch.sum(phi)
        Sprime = S
        for i in range(len(Sprime)):
            output[:, i] *= Sprime[i].expand_as(output[:, i])
        return output, Sprime


to run the compression for switches, currently, we need to first run the files switches, lenet5_conv_gpu_switch_working_Feb20_pointest and lenet5_conv_gpu_switch_working_Feb20_integral
and then run the compression by loading the files.
For all the layers you run it with script_vgg_switch.py


-------------------------------------------

COMPARISON

ranking/cluster_output  - contains the files from cluster, e.g. the outputs of the pruning and retraining file for different methods (as e.g. in vgg_main2_module.py). it contains different architectures tested for different methods.
note: for vgg it seems that the best method is not in all teh parameters grid dearch (parameters_vgg_all.txt) but from testing in parameters_vgg2.txt

-----------

WIDE RES NET

pytorch-prunes-master