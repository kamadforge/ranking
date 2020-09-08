#!/bin/bash
#SHOULD BE CALLED FROM RESULTS_COMPRESSION directory
EXP_NAME="CIFAR100_"
counter=1
while [ $counter -le 10 ]
do
	CUDA_VISIBLE_DEVICES="7" python vgg_main2_tomodule.py --run_name "${EXP_NAME}$counter" --dataset "CIFAR-100" --scratch_training_numepochs 500
	#echo "$counter"
	((counter++))
done
echo All done
