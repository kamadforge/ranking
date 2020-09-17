#!/bin/bash
#SHOULD BE CALLED FROM RESULTS_COMPRESSION DIRECTORY
counter=1
while [ $counter -le 1 ]
do
	CUDA_VISIBLE_DEVICES="7" python vgg_main2_tomodule.py --resume=True --run_name "l2_0_$counter" --dataset "CIFAR-100"
	#echo "$counter"
	((counter++))
done
echo All done
