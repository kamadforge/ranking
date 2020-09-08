#!/bin/bash
#SHOULD BE CALLED FROM RESULTS_COMPRESSION DIRECTORY
counter=1
while [ $counter -le 1 ]
do
	CUDA_VISIBLE_DEVICES="4" python vgg_main2_tomodule.py --resume=True --run_name "l2_1_$counter"
	#echo "$counter"
	((counter++))
done
echo All done
