
network='vgg16'

input_shape = (1, 28,28) # Format:(channels, rows,cols)

if network=='lenet5':

    filters_num=[3,18,284,283]
    filters_num=[7,13,208,16]
    filters_num=[3,12,192,500]
    filters_num=[3,8,128,499]
    filters_num=[2,7,112,478]
    filters_num=[5,10,76,16]
    filters_num=[8,13,88,13]
    filters_num=[5,7,45,20]



    conv_filters= [[1,filters_num[0],5,5], [2,2], [filters_num[0], filters_num[1], 5,5], [2,2], [filters_num[2]], [filters_num[3]], [10]]  # Format: (num_filters, channels, rows, cols)
    strides = [1, 2, 1, 2]
    paddings = [0, 0, 0, 0]
    activation = 'relu'

elif network=='vgg16':
    filter_nums=[64,64,128,128,256,256,256,512,512,512,512,512,512,512,512,  512, 512]
    filter_nums=[34,34,68,68,75,106,101,92,102,92,92,67,67,62,62,  512, 512]
    #filter_nums=[39,39,63,48,55,98,97,52,62,22,42,47,47,42,62,  512, 512]
    #filter_nums=[63,64,128,128,245,155,63,26,24,24,20,14,12,11,11,  512, 512]
    #filter_nums=[51,62,125,128,228,129,38,13,9,6,5,6,6,6,20,   512, 512]




    conv_filters=[[3,filter_nums[0],3,3],  [filter_nums[0],filter_nums[1],3,3],
                  [filter_nums[1],filter_nums[2],3,3], [filter_nums[2],filter_nums[3],3,3],
                  [filter_nums[3],filter_nums[4],3,3], [filter_nums[4],filter_nums[5],3,3],  [filter_nums[5],filter_nums[6],3,3],
                  [filter_nums[6],filter_nums[7],3,3], [filter_nums[7],filter_nums[8],3,3],  [filter_nums[8],filter_nums[9],3,3],
                  [filter_nums[9],filter_nums[10],3,3],
                  [filter_nums[10],filter_nums[11],3,3],  [filter_nums[11],filter_nums[12],3,3], [filter_nums[12],filter_nums[13],3,3],
                  [filter_nums[13],filter_nums[14],3,3],
                  [filter_nums[15]], [filter_nums[16]], [10]]

    #
    strides=[1,1,2,1,1,2,1,1,1,2,1,1,1,1,2,1,1,1,1,2]
    paddings=[1,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0]


total_flops=0
total_params=0
#with this formyula for alexnmet the number is like computed in the internet 105415200


for i in range(1, len(conv_filters)+1): #i is the number of the layer

    conv_filter=conv_filters[i-1]



    if len(conv_filter)==4:
        print("\nconv layer")
        stride = strides[i - 1]
        padding = paddings[i - 1]
        #############CONV LAYERS
        #per one filter application, how many flops
        # 1a vector_length
        # num of input maps * filter_wisth * filter_height
        n = conv_filter[1] * conv_filter[2] * conv_filter[3]
        # 1b general defination for number of flops (n: multiplications and n-1: additions)
        #flops_per_instance = n + (n-1)
        flops_per_instance = n


        #how many times we apply filter in an image
        # 2a for rows
        num_instances_per_filter_rows = (( input_shape[1] - conv_filter[2] + 2*padding) / stride ) + 1
        #print("num_instances_per_filter_rows(vert): %d" % num_instances_per_filter_rows)
        # 2b multiplied by columns
        num_instances_per_filter_cols = (( input_shape[2] - conv_filter[3] + 2*padding) / stride ) + 1
        #print("num_instances_per_filter_cols (hor): %d" % num_instances_per_filter_cols)

        num_instances_per_filter=num_instances_per_filter_rows*num_instances_per_filter_cols


        #1 *2 flops for one output filter aplied to all input feature maps
        flops_per_filter = num_instances_per_filter * flops_per_instance

        # multiply with number of output filters
        total_flops_per_layer = flops_per_filter * conv_filter[0]

        # if activation == 'relu':
        #     # Here one can add number of flops required
        #     # Relu takes 1 comparison and 1 multiplication
        #     # Assuming for Relu: number of flops equal to length of input vector
        #     total_flops_per_layer += conv_filter[0]*input_shape[1]*input_shape[2]
        #

        input_shape=(conv_filter[0], num_instances_per_filter_rows, num_instances_per_filter_cols)

        print(input_shape)

        total_flops+=total_flops_per_layer
        print("flops per layer: {:,}, total flops: {:,}".format(total_flops, total_flops_per_layer))


        #########################################

        params=(conv_filter[0]*conv_filter[2]*conv_filter[3]+1)*conv_filter[1]

        total_params+=params
        print("params: %d, total: %d" % (params, total_params))

        ######################


    elif len(conv_filter)==2: # pooling layer
        stride = strides[i - 1]
        padding = paddings[i - 1]
        print("\npooling layer")
        # how many times we apply filter in an image
        # 2a for rows
        num_instances_per_filter_rows = ((input_shape[1] - conv_filter[0] + 2 * padding) / stride) + 1
        #print("num_instances_per_filter_rows(vert): %d", num_instances_per_filter_rows)
        # 2b multiplied by columns
        num_instances_per_filter_cols = ((input_shape[2] - conv_filter[1] + 2 * padding) / stride) + 1
        #print("num_instances_per_filter_cols (hor): %d", num_instances_per_filter_cols)

        input_shape=(conv_filter[0], num_instances_per_filter_rows, num_instances_per_filter_cols)
        print(input_shape)

    elif len(conv_filter)==1:
        print("\n fc layer")
        if len(input_shape)==3:
            total_flops_per_layer=conv_filter[0]*num_instances_per_filter_rows*num_instances_per_filter_cols
        else:
            total_flops_per_layer=conv_filter[0]*input_shape[0]

        print(total_flops_per_layer)

        total_flops += total_flops_per_layer

        input_shape=[conv_filter[0]]
        print(input_shape)

        print("flops per layer: {:,}, total flops: {:,}".format( total_flops_per_layer, total_flops ))

        ############################
        if len(input_shape) == 3:
            params = conv_filter[0] * (num_instances_per_filter_rows * num_instances_per_filter_cols+1)
        else:
            params = conv_filter[0] * (input_shape[0]+1)


        total_params += params
        print("params: {:,}, total: {:,}".format(params, total_params))


print("\nSummary: ")
print("total flops: {:,}".format(total_flops))
print("total params: {:,} ".format( total_params))
######################3FULLY CONNECTED
