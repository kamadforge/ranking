input_shape = (3, 227,227) # Format:(channels, rows,cols)


conv_filters = [(96,3,11,11), (3,3), (256, 96, 5,5)]  # Format: (num_filters, channels, rows, cols)

stride = [4]
padding = [0]
activation = 'relu'

total_flops=0
#with this formyula for alexnmet the number is like computed in the internet 105415200


for i in range(1, len(conv_filters)+1): #i is the number of the layer

    conv_filter=conv_filters[i-1]

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
    print("num_instances_per_filter_rows(vert): %d",  num_instances_per_filter_rows)
    # 2b multiplied by columns
    num_instances_per_filter_cols = (( input_shape[2] - conv_filter[2] + 2*padding) / stride ) + 1
    print("num_instances_per_filter_cols (hor): %d", num_instances_per_filter_cols)

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

    input_layer=(conv_filter[0], num_instances_per_filter_rows, num_instances_per_filter_cols)

    print(total_flops_per_layer)

    total_flops+=total_flops_per_layer

print("total flops: %d" % total_flops)
######################3FULLY CONNECTED
