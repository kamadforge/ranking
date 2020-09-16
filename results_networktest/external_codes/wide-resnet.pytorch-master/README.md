

### WideResNet

Along with other settings, the default value for num_epoch=200 can be changed in the config.py


#### 1. To train the base network run:

python main_prune.py

The checkpoint is saved in the folder checkpoints


#### 2. To prune and retrain

python main_prune.py --resume --prune True --arch 75,85,80,80,159,159,154,159,315,315,314,316 --model_path checkpoint/cifar10_pruned_65,75,80,70,159,149,154,149,305,315,304,306_acc_tensor(93.0900)/wide-resnet-28x10.t7



#### 3. Then compute the switch vectors

python main_switch.py

It is enough to run it for even 1 iteration, 3-5 are recommended.