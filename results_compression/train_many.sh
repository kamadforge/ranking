#!/bin/bash
counter=11
while [ $counter -le 13 ]
do
	CUDA_VISIBLE_DEVICES="7" python vgg_main2_tomodule.py --run_name "$counter"
	#echo "$counter"
	((counter++))
done
echo All done
