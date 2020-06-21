### VGG



#### 1. To train the base network run:

```
python results_compression/vgg_main2_tomodule.py
```


#### 2. To resume a model and check its accuracy

In model argument, put the path to the desired model trained in Step 1.

```
python results_compression/vgg_main2_tomodule.py --resume=True --model=results_compression/checkpoint/ckpt_vgg16_59.51.t7
```

#### 3. To resume a model and prune it 

We prune a given number of neurons which are deemed unimportant by a selected method

`--model` - put the path to the desired model trained in Step 1.

`--arch` - put the number of neurons to be pruned at each layer (for all 14 layers)

`--method` - l1 or l2 or switch

```
python results_compression/vgg_main2_tomodule.py --prune_bool=True --model=results_compression/checkpoint/ckpt_vgg16_59.51.t7 --method=l1 --arch=1,2,3,4,5,6,7,8,9,10,11,12,13,14

```


#### 4. To resume a model and prune it and retrain it

```
python results_compression/vgg_main2_tomodule.py --prune_bool=True --retrain=True --model=results_compression/checkpoint/ckpt_vgg16_59.51.t7 --method=l1 --arch=1,2,3,4,5,6,7,8,9,10,11,12,13,14
```


#### 5. Switches

To run the switch pruning, in addition to selection method `switch` (as in step 4), you can specify additional parameters


```
python results_compression/vgg_main2_tomodule.py --model=results_compression/checkpoint/ckpt_vgg16_94.34.t7 --method switch --switch_epochs 5 --switch_train True --prune_bool True --retrain=True


```

`--switch_epochs` - number of epochs to train parameters, 1 epoch is enough but 5-8 is recommended.

`--switch_train` - once the switch_train is run once, we remove this tag, and the script will load the previously trained switches
