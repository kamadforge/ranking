### VGG

For all experiments

```
cd results_compression
```

#### 1. To train the base network run:

```
python vgg_main2_tomodule.py
```


#### 2. To resume a model and check its accuracy

In model argument, put the path to the desired model trained in Step 1.

```
python vgg_main2_tomodule.py --resume=True --model=checkpoint/ckpt_vgg16_59.51.t7
```

#### 3. To resume a model and prune it 

We prune a given number of neurons which are deemed unimportant by a selected method

`--model` - put the path to the desired model trained in Step 1.

`--arch` - put the number of neurons to be pruned at each layer (for all 14 layers)

`--method` - l1 or l2 

```
python vgg_main2_tomodule.py --prune_bool=True --model=checkpoint/ckpt_vgg16_59.51.t7 --method=l1 --arch=1,2,3,4,5,6,7,8,9,10,11,12,13,14

```


#### 4. To resume a model and prune it and retrain it

```
python vgg_main2_tomodule.py --prune_bool=True --retrain=True --model=checkpoint/ckpt_vgg16_59.51.t7 --method=l1 --arch=1,2,3,4,5,6,7,8,9,10,11,12,13,14
```

