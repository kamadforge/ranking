import numpy as np


# https://stackoverflow.com/questions/28232235/how-to-calculate-the-number-of-parameters-of-convolutional-neural-networks


# actually the original archicture was 6,16,120,84
# https: // github.com / activatedgeek / LeNet - 5 / blob / master / lenet.py


def get_difference(arr1, arr2):
    new_arr = []
    for i in range(len(arr1)):
        new_arr.append(arr1[i] - arr2[i])
    return new_arr


def main(network_arg, dataset_arg, filters_num_arg):
    network = network_arg
    dataset = dataset_arg
    filter_nums = [int(a) for a in filters_num_arg.split(",")]
    print(filter_nums)

    if network == 'lenet5':

        input_shape = (1, 28, 28)  # Format:(channels, rows,cols)



        # for i1 in [2, 3, 4, 5, 6, 7]:
        #     for i2 in [4, 6, 8, 10, 12]:
        #         for i3 in [20, 30, 40, 50, 60]:
        #             for i4 in [5, 10, 15, 20]:
        #                 for method in methods:
        #                     print("\n\n %s \n" % method)
        #                     prune(False, i1, i2, i3, i4, write, save)
        #                 print("\n*********************************\n\n")
        # filters_num = [3, 4, 20, 10]

        if 1:
            if 1:
                if 1:
                    if 1:
                        # for i1 in [2, 3, 4, 5, 6, 7]:
                        #     for i2 in [4, 6, 8, 10, 12]:
                        #         for i3 in [20, 30, 40, 50, 60]:
                        #             for i4 in [5, 10, 15, 20]:

                        # filters_num=[i1,i2,i3,i4]

                        print("\n**************************\n")
                        print(filter_nums, '\n', '-' * 20)
                        conv_filters = [[1, filter_nums[0], 5, 5], [2, 2], [filter_nums[0], filter_nums[1], 5, 5],
                                        [2, 2], [filter_nums[2]], [filter_nums[3]],
                                        [10]]  # Format: (num_filters, channels, rows, cols)
                        strides = [1, 2, 1, 2]
                        paddings = [0, 0, 0, 0]
                        activation = 'relu'
                        layers_type = ['C', 'P', 'C', 'P', 'FC', 'FC', 'FC']
                        get_flops_params(layers_type, conv_filters, paddings, strides, input_shape, network)

    elif network == 'vgg16':


        if dataset == 'cifar':
            input_shape = (1, 32, 32)
            layers_type = ['C', 'C', 'P', 'C', 'C', 'P', 'C', 'C', 'C', 'P', 'C', 'C', 'C', 'P', 'C', 'C', 'C', 'P',
                           'FC',
                           'FC']

        elif dataset == 'imagenet':
            input_shape = (1, 224, 224)  # Format:(channels, rows,cols)
            layers_type = ['C', 'C', 'P', 'C', 'C', 'P', 'C', 'C', 'C', 'P', 'C', 'C', 'C', 'P', 'C', 'C', 'C', 'P',
                           'FC',
                           'FC', 'FC']
            filter_nums = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096, 1000]




        if dataset == 'cifar':
            conv_filters = [[3, filter_nums[0], 3, 3], [filter_nums[0], filter_nums[1], 3, 3], [2, 2],
                            [filter_nums[1], filter_nums[2], 3, 3], [filter_nums[2], filter_nums[3], 3, 3], [2, 2],
                            [filter_nums[3], filter_nums[4], 3, 3], [filter_nums[4], filter_nums[5], 3, 3],
                            [filter_nums[5], filter_nums[6], 3, 3], [2, 2],
                            [filter_nums[6], filter_nums[7], 3, 3], [filter_nums[7], filter_nums[8], 3, 3],
                            [filter_nums[8], filter_nums[9], 3, 3], [2, 2],
                            [filter_nums[9], filter_nums[10], 3, 3], [filter_nums[10], filter_nums[11], 3, 3],
                            [filter_nums[11], filter_nums[12], 3, 3], [2, 2],
                            # [filter_nums[12],filter_nums[13],3,3],
                            # [filter_nums[13],filter_nums[14],3,3]]
                            # [filter_nums[15]], [filter_nums[16]], [10]]
                            [filter_nums[13]], [filter_nums[14]]]  # for cifar, which has fc 2 layers
            # [filter_nums[13]], [filter_nums[14]], [filter_nums[15]]] #for imagenet which jas 3 layers
        elif dataset == 'imagenet':
            conv_filters = [[3, filter_nums[0], 3, 3], [filter_nums[0], filter_nums[1], 3, 3], [2, 2],
                            [filter_nums[1], filter_nums[2], 3, 3], [filter_nums[2], filter_nums[3], 3, 3], [2, 2],
                            [filter_nums[3], filter_nums[4], 3, 3], [filter_nums[4], filter_nums[5], 3, 3],
                            [filter_nums[5], filter_nums[6], 3, 3], [2, 2],
                            [filter_nums[6], filter_nums[7], 3, 3], [filter_nums[7], filter_nums[8], 3, 3],
                            [filter_nums[8], filter_nums[9], 3, 3], [2, 2],
                            [filter_nums[9], filter_nums[10], 3, 3], [filter_nums[10], filter_nums[11], 3, 3],
                            [filter_nums[11], filter_nums[12], 3, 3], [2, 2],
                            # [filter_nums[12],filter_nums[13],3,3],
                            # [filter_nums[13],filter_nums[14],3,3]]
                            # [filter_nums[15]], [filter_nums[16]], [10]]
                            # [filter_nums[13]], [filter_nums[14]]]  # for cifar, which has fc 2 layers
                            [filter_nums[13]], [filter_nums[14]], [filter_nums[15]]]  # for imagenet which jas 3 layers

        # only for conv layers
        strides = [1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2]
        paddings = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]

        print("\n\n")
        print(filter_nums)
        print('-'.join([str(i) for i in filter_nums]))
        get_flops_params(layers_type, conv_filters, paddings, strides, input_shape, network)


    elif network == 'wrn':
        if dataset == 'cifar':
            input_shape = (1, 32, 32)
            layers_type = ['C', 'C', 'P', 'C', 'C', 'P', 'C', 'C', 'C', 'P', 'C', 'C', 'C', 'P', 'C', 'C', 'C', 'P',
                           'FC', 'FC']
            filter_nums = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 10]


def get_flops_params(layers_type, conv_filters, paddings, strides, input_shape, network):
    total_flops = 0
    total_params = 0
    # with this formyula for alexnmet the number is like computed in the internet 105415200

    for i in range(1, len(layers_type) + 1):  # i is the number of the layer

        conv_filter = conv_filters[i - 1]
        layer_type = layers_type[i - 1]

        if layer_type == 'C':
            # print("\nconv layer")
            stride = strides[i - 1]
            padding = paddings[i - 1]
            ############# CONV LAYERS
            # per one filter application, how many flops
            # 1a vector_length
            # num of input maps * filter_wisth * filter_height
            n = conv_filter[1] * conv_filter[2] * conv_filter[3]
            # 1b general defination for number of flops (n: multiplications and n-1: additions)
            # flops_per_instance = n + (n-1)
            flops_per_instance = n

            # how many times we apply filter in an image
            # 2a for rows
            num_instances_per_filter_rows = ((input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1
            # print("num_instances_per_filter_rows(vert): %d" % num_instances_per_filter_rows)
            # 2b multiplied by columns
            num_instances_per_filter_cols = ((input_shape[2] - conv_filter[3] + 2 * padding) / stride) + 1
            # print("num_instances_per_filter_cols (hor): %d" % num_instances_per_filter_cols)

            num_instances_per_filter = num_instances_per_filter_rows * num_instances_per_filter_cols

            ################################ FLOPS

            # 1 *2 flops for one output filter aplied to all input feature maps
            flops_per_filter = num_instances_per_filter * flops_per_instance

            # multiply with number of output filters
            total_flops_per_layer = flops_per_filter * conv_filter[0]

            # if activation == 'relu':
            #     # Here one can add number of flops required
            #     # Relu takes 1 comparison and 1 multiplication
            #     # Assuming for Relu: number of flops equal to length of input vector
            #     total_flops_per_layer += conv_filter[0]*input_shape[1]*input_shape[2]
            #

            input_shape = (conv_filter[0], num_instances_per_filter_rows, num_instances_per_filter_cols)
            print(input_shape)

            total_flops += total_flops_per_layer
            # print("flops per layer: {:,}, total flops: {:,}".format(total_flops, total_flops_per_layer))

            ######################################### PARAMETERS

            params = (conv_filter[0] * conv_filter[2] * conv_filter[3] + 1) * conv_filter[1]

            total_params += params
            print("params: {:,}, total: {:,}".format(params, total_params))

            ######################


        elif layer_type == 'P':  # pooling layer
            stride = strides[i - 1]
            padding = paddings[i - 1]
            # print("\npooling layer")
            # how many times we apply filter in an image
            # 2a for rows
            num_instances_per_filter_rows = ((input_shape[1] - conv_filter[0] + 2 * padding) / stride) + 1
            # print("num_instances_per_filter_rows(vert): %d", num_instances_per_filter_rows)
            # 2b multiplied by columns
            num_instances_per_filter_cols = ((input_shape[2] - conv_filter[1] + 2 * padding) / stride) + 1
            # print("num_instances_per_filter_cols (hor): %d", num_instances_per_filter_cols)

            input_shape = (input_shape[0], num_instances_per_filter_rows, num_instances_per_filter_cols)
            print(input_shape, "pool")
            print("")

        elif layer_type == 'FC':
            # print("\n fc layer")
            if len(input_shape) == 3:
                total_flops_per_layer = input_shape[0] * num_instances_per_filter_rows * num_instances_per_filter_cols * \
                                        conv_filter[0]
            else:
                total_flops_per_layer = conv_filter[0] * input_shape[0]

            # print(total_flops_per_layer)

            total_flops += total_flops_per_layer

            # input_shape=[conv_filter[0]]
            # print(input_shape, "fc")

            # print("flops per layer: {:,}, total flops: {:,}".format( total_flops_per_layer, total_flops ))

            ############################
            if len(input_shape) == 3:
                params = input_shape[0] * num_instances_per_filter_rows * num_instances_per_filter_cols * conv_filter[
                    0] + conv_filter[0]
            else:
                params = conv_filter[0] * (input_shape[0] + 1)

            input_shape = [conv_filter[0]]
            print(input_shape, "fc")
            total_params += params
            print("params: {:,}, total: {:,}".format(params, total_params))

    print("\nSummary: ")
    print("total flops: {:,}".format(total_flops))
    print("total params: {:,} ".format(total_params))
    if network == 'vgg16':
        compression_rate = 4, 982, 474 / total_params
    elif network == 'lenet5':
        compression_rate = 25226 / total_params
    print('Compression rate: ', compression_rate)
    ###################### 3FULLY CONNECTED

    return total_params


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="lenet5", choices=["vgg16, lenet5"])
    parser.add_argument("--data", default="cifar", choices=["cifar"])
    # parser.add_argument("--arch", default="64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 10")
    parser.add_argument("--arch", default="6, 7, 35, 17")
    args = parser.parse_args()


    main(args.network, args.data, args.arch)

    # filters_num=[6,16,120,84] #original lenet
    # filters_num=[3,18,284,283]
    # filters_num=[7,13,208,16]
    # filters_num=[3,12,192,500]
    # filters_num=[3,8,128,499]
    # filters_num=[2,7,112,478]
    # filters_num=[5,10,76,16]
    # filters_num=[8,13,88,13]
    # filters_num=[5,7,45,20]
    # filters_num=[6,8,40,20] #IS 150
    # filters_num = [5, 8, 45, 15]  # IS point
    # filters_num = [6, 7, 35, 17]  # IS joint

# filters_num = [10, 20, 100, 25]

# widerresnet 28 10 36.5M

# filter_nums = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 10]
# filter_nums = np.array(filter_nums1)-np.array([25,25,  65,80,  201,158,159,  460,450,490,  470,465,465,   450, 10])
# filter_nums=[59, 59,  88, 88,   236, 216, 176,  432, 352, 472,  472, 352, 432,  352, 10] #0.6
# filter_nums=[59, 59,  108, 118,  236, 176, 216, 472, 472, 432,  352, 432, 472, 432, 10] #0.7
# filter_nums=[59, 59, 118, 118, 236, 246, 236, 492, 472, 492, 492, 472, 472, 492, 432]
# filter_nums=[59, 59, 88, 88, 236, 216, 176, 382, 322, 252, 252, 352, 262, 262, 352]
# filter_nums=[59, 59, 88, 88, 236, 216, 136, 282, 262, 212, 212, 352, 262, 262, 352]
# filter_nums=[59, 59, 118, 118, 246, 246, 246, 492, 492, 492, 502, 502, 502, 502, 502]

# filters_removed=[25,25,65,80,201,168,169,460,450,490,470,465,465,465,0]
# print('*'*300)
# filter_nums=get_difference(filter_all_nums, filters_removed)

# filter_nums=[39, 39,   63, 48,   55, 88, 87,   52, 62, 22,   42, 47, 47,   47, 10] #IS (150)
# filter_nums=[64,64,128,128,256,256,256,512,512,512,512,512,512,512,512,  512, 512]
# filter_nums=[34,34,   68,68,  75,106,101,  92,102,92,  92,67,67, 62,10]
# filter_nums=[39,39,63,48,55,98,97,52,62,22,42,47,47,42,62,  10]
# filter_nums=[63,64, 128,128, 245,155,63, 26,24,20, 14,12,11,  15,10] #BC-GNJ # przed ostatnia i przed przed ostatnia wybierze mniejsza (after double notation luizos) - removed 11
# filter_nums=[51,62,  125,128,  228,129,38,  13,9,6,  5,6,6,  20,10] #BC-GHS  # przed ostatnia i przed przed ostatnia wybierze mniejsza (after double notation luizos) - removed 6
# filter_nums = [40, 58,   127, 122,   248, 251, 245,    430, 289, 180,    118, 170, 337,  504, 10 ] #local 0.01, global 0.01
# filter_nums = [27, 57,   125, 122,   236, 244, 246,   340, 127, 77,    89, 52, 380,    414, 10]
# filter_nums=[44, 58,  127, 122,  248, 251, 246,  468, 410, 375,  369, 394, 502,  504, 10] #0%
# filter_nums=[44, 58, 126, 122, 238, 242, 236, 449, 395, 360, 356, 378, 481, 484] #5%
# filter_nums=[44, 58, 118, 113, 229, 229, 226, 428, 370, 347, 334, 361, 456, 457] #10%
# filter_nums=[44, 58, 119, 113, 228, 230, 227, 429, 377, 336, 335, 359, 544, 406]
