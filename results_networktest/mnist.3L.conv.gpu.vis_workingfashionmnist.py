#it's the same test as for mnist.#L.py but with conv layers (con lenet)

#transforms the input data

import torch
import torch.optim as optim
from torch import nn, optim
import torch.nn.functional as f

import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import csv
import pdb
import matplotlib.pyplot as plt

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"
#print(device)


##############################3
# CHANGE
# dataset naem
# dataset itself
# model file loaded


###################################################
# DATA

dataset="fashionmnist"
filename="nmnist_test_conv_full_test.txt"
BATCH_SIZE = 100
# Download or load downloaded MNIST dataset
# shuffle data at every epoch
train_loader = torch.utils.data.DataLoader(
datasets.FashionMNIST('data', train=True, download=True,
                    #transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    #datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    datasets.FashionMNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=False)


#########################################################
# PARAMS

early_stopping=35


##########################################################
# NETWORK


class Lenet(nn.Module):
    def __init__(self, nodesNum1, nodesNum2, nodesFc1, nodesFc2):
        super(Lenet, self).__init__()

        self.nodesNum2=nodesNum2

        # network without bn

        # self.c1=nn.Conv2d(1, nodesNum1, 5)
        # self.s1=nn.MaxPool2d(2)
        # self.c2=nn.Conv2d(nodesNum1,nodesNum2,5)
        # self.s2=nn.MaxPool2d(2)
        # self.fc1=nn.Linear(nodesNum2*4*4, nodesFc1)
        # self.fc2=nn.Linear(nodesFc1,nodesFc2)
        # self.output=nn.Linear(nodesFc2,10)

        #network with bn

        self.c1 = nn.Conv2d(1, nodesNum1, 5)
        self.s2 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(nodesNum1)
        self.c3 = nn.Conv2d(nodesNum1, nodesNum2, 5)
        self.s4 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(nodesNum2)
        self.c5 = nn.Linear(nodesNum2 * 4 * 4, nodesFc1)
        self.f6 = nn.Linear(nodesFc1, nodesFc2)
        self.output = nn.Linear(nodesFc2, 10)


    def forward(self, x, vis, epoch, i):

        # x=x.view(-1,784)
        # output=f.relu(self.fc1(x))
        # output=self.bn1(output)
        # output=f.relu(self.fc2(output))
        # output=self.bn2(output)
        # output=self.fc3(output)
        # return output

        #x=x.view(-1,784)
        #if vis:
        #    mm=x.cpu().detach().numpy()
        #    plt.imshow(mm[1, 0, :, :]) #only one channel
        #    plt.savefig("results_networktest/vis/filters/orig/orig_epoch%d_batch%i" % (epoch, i))

        #######################################
        # without bn

        # output = self.c1(x)
        # if vis:
        #     mm = output.cpu().detach().numpy()
        #     plt.imshow(mm[1, 2, :, :])  # showing 2nd channel (example of a channel)
        #     plt.savefig("results_networktest/vis/filters/conv1/conv1_epoch%d_batch%i" % (epoch, i))
        #
        # output = self.s1(output)
        # output = self.c2(output)
        # # if vis:
        # #    mm=output.cpu().detach().numpy()
        # #    plt.imshow(mm[1,2,:,:]) #showing 2nd channel (example of a channel)
        # #    plt.savefig("results_networktest/vis/filters/conv2/conv2_epoch%d_batch%i" % (epoch, i))
        #
        # output = self.s2(output)
        # output = output.view(-1, self.nodesNum2 * 4 * 4)
        # output = self.fc1(output)
        # output = self.fc2(output)
        # return output

        #####################################3
        #with bn and relu

        output=self.c1(x)



        if vis:
            for filter_num in range(10):
                mm=output.cpu().detach().numpy()
                #fig,ax = plt.subplots(1)

                #ax.imshow(mm[1,filter_num,:,:], cmap="gray", aspect='normal')
                plt.imshow(mm[1,filter_num,:,:], cmap="gray") #showing 2nd channel (example of a channel)
                #fig.subplots_adjust(bottom=0)
                #fig.subplots_adjust(top=1)
                #fig.subplots_adjust(right=1)
                #fig.subplots_adjust(left=0)
                #plt.axis('off')
                # fig.set_size_inches(5, 5)
                # # fig.set_size_inches(w, h)
                # ax = fig.add_axes([-5, -5, 0.8, 0.8])
                # #ax = plt.Axes(fig, [0., 0., 1, 1.])
                # ax.set_axis_off()
                # fig.add_axes(ax)
                # Some random scatterpoint data

                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                #plt. savefig("filename.pdf", bbox_inches='tight', pad_inches=0)


                # sol 2
                # plt.gca().set_axis_off()
                # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                #                     hspace=0, wspace=0)
                # plt.margins(0, 0)
                #plt.savefig("myfig.pdf")

                # # 1 solution
                # # Creare your figure and axes
                # fig, ax = plt.subplots(1)
                #
                # # Set whitespace to 0
                # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                #
                # # Display the image
                # ax.imshow(mm[1,filter_num,:,:], cmap="gray",extent=(0, 1, 1, 0)) #showing 2nd channel (example of a channel)
                # #plt.imshow(mm[1,filter_num,:,:], cmap="gray",extent=(0, 1, 1, 0)) #showing 2nd channel (example of a channel)
                #
                # # Turn off axes and set axes limits
                # ax.axis('tight')
                # ax.axis('off')
                #
                # plt.show()
                plt.savefig("vis/feature_maps/%s/90.04/trial_1noaxes/conv1/conv1_batch%d_filternum%d_epoch%d" % (dataset, i, filter_num, epoch), bbox_inches='tight', pad_inches=0)

                #plt.savefig("vis/feature_maps/%s/90.04/trial_1noaxes/conv1/conv1_batch%d_filternum%d_epoch%d" % (dataset, i, filter_num, epoch))


            # filter_num=5
            # plt.imshow(mm[1, filter_num, :, :], cmap="gray")  # showing 2nd channel (example of a channel)
            # plt.savefig("vis/feature_maps/%s/trial_3/conv1/conv1_batch%d_filternum%d_epoch%d" % (dataset, i, filter_num, epoch))
            # filter_num=14
            # plt.imshow(mm[1, filter_num, :, :], cmap="gray")  # showing 2nd channel (example of a channel)
            # plt.savefig("vis/feature_maps/%s/trial_3/conv1/conv1_batch%d_filternum%d_epoch%d" % (dataset, i, filter_num, epoch))
            # filter_num = 13
            # plt.imshow(mm[1, filter_num, :, :], cmap="gray")  # showing 2nd channel (example of a channel)
            # plt.savefig( "vis/feature_maps/%s/trial_2/conv1/conv1_batch%d_filternum%d_epoch%d" % (dataset, i, filter_num, epoch))
            # filter_num = 1
            # plt.imshow(mm[1, filter_num, :, :], cmap="gray")  # showing 2nd channel (example of a channel)
            # plt.savefig("vis/feature_maps/%s/trial_3/conv1/conv1_batch%d_filternum%d_epoch%d" % (dataset, i, filter_num, epoch))
            # filter_num = 11
            # plt.imshow(mm[1, filter_num, :, :], cmap="gray")  # showing 2nd channel (example of a channel)
            # plt.savefig("vis/feature_maps/%s/trial_3/conv1/conv1_batch%d_filternum%d_epoch%d" % (dataset, i, filter_num, epoch))
            # filter_num = 8
            # plt.imshow(mm[1, filter_num, :, :], cmap="gray")  # showing 2nd channel (example of a channel)
            # plt.savefig("vis/feature_maps/%s/trial_3/conv1/conv1_batch%d_filternum%d_epoch%d" % (dataset, i, filter_num, epoch))

        output = f.relu(self.s2(output))
        output = self.bn1(output)

        output=self.c3(output)
        # if vis:
        #    mm=output.cpu().detach().numpy()
        #    filter_num = 33
        #    plt.imshow(mm[1, filter_num, :, :], cmap="gray")  # showing 2nd channel (example of a channel)
        #    plt.savefig("vis/feature_maps/mnist/trial_3/conv2/conv2_batch%d_filternum%d_epoch%d" % (i, filter_num, epoch))
        #    filter_num = 41
        #    plt.imshow(mm[1, filter_num, :, :], cmap="gray")  # showing 2nd channel (example of a channel)
        #    plt.savefig("vis/feature_maps/mnist/trial_3/conv2/conv2_batch%d_filternum%d_epoch%d" % (i, filter_num, epoch))
        #    filter_num = 27
        #    plt.imshow(mm[1, filter_num, :, :], cmap="gray")  # showing 2nd channel (example of a channel)
        #    plt.savefig("vis/feature_maps/mnist/trial_3/conv2/conv2_batch%d_filternum%d_epoch%d" % (i, filter_num, epoch))
        #    filter_num = 49
        #    plt.imshow(mm[1, filter_num, :, :], cmap="gray")  # showing 2nd channel (example of a channel)
        #    plt.savefig("vis/feature_maps/mnist/trial_3/conv2/conv2_batch%d_filternum%d_epoch%d" % (i, filter_num, epoch))


        output = f.relu(self.s4(output))
        output = self.bn2(output)

        output=output.view(-1, self.nodesNum2*4*4)
        output=self.c5(output)
        output=self.f6(output)
        return output





###################################################
# RUN

def run_experiment(train, nodesNum1, nodesNum2, nodesFc1, nodesFc2):

    net=Lenet(nodesNum1, nodesNum2, nodesFc1, nodesFc2).to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer=optim.Adam(net.parameters(), lr=0.001)

    ##########################################
    # EVALUATE

    def evaluate(epoch):
        # print('Prediction when network is forced to predict')
        net.eval()
        correct = 0
        total = 0
        for j, data in enumerate(test_loader):
            # pdb.set_trace()
            images, labels = data
            images =images.to(device)
            predicted_prob = net.forward(images, True, epoch, j) # images.view(-1,28*28)

            # _,prediction = torch.max(predicted_prob.data, 1)
            # #prediction = prediction.cpu().numpy()

            predicted=np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
            total += labels.size(0)
            correct += (predicted == labels.numpy()).sum().item()

        # print(str(correct) +" "+ str(total))
        # pdb.set_trace()
        accuracy=100 * float(correct) / total
        print("accuracy: %.2f %%" % (accuracy))
        return accuracy

    #########################
    # TEST ONLY
    if train==False:
        #path = "models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"
        path = "models/fashionmnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:62_acc:90.04"
        net.load_state_dict(torch.load(path)["model_state_dict"])
        accuracy=evaluate(-1)

        return accuracy, -1 #-1 epoch means it is just the test

    ###################################
    # TRAINING

    if train==True:
        stop=0; epoch=0; best_accuracy=0; entry=np.zeros(3)
        while (stop<early_stopping):
            epoch=epoch+1
            for i, data in enumerate(train_loader):

                inputs, labels=data
                inputs, labels=inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs=net(inputs, False, epoch, i)
                loss=criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                #if i % 100==0:
                #    print (i)(
                #   print (loss.item())
            #print (i)
            print (loss.item())
            accuracy=evaluate(epoch)
            print ("Epoch " +str(epoch)+ " ended.")



            ############### visualize filters

            if (accuracy<=best_accuracy):
                stop=stop+1
                entry[2]=0
            else:
                best_accuracy=accuracy
                print("Best updated")
                stop=0
                entry[2]=1
            print("\n")
            #write
            entry[0]=accuracy; entry[1]=loss
            with open(filename, "a+") as file:
                file.write(",".join(map(str, entry))+"\n")

            ########################################

        name="results_networktest/vis/weights_mnist.3L.conv"
        torch.save(net.state_dict(), name)

         #visualize filters
        for name, param in net.named_parameters():
           print(param.shape)
           print(name)
           if (name=="c1.weight" or name=="c2.weight"):
               for i1 in range(param.shape[0]):
                   for i2 in range(param.shape[1]):
                       plt.imshow(param[i1][i2].cpu().detach().numpy(), cmap='gray')
                       #plt.show()
                       plt.savefig("results_networktest/vis/filters/%s_gray/%d_%d" % (name, i1, i2))

        return best_accuracy, epoch


#########################################################################
train=False

if train==False:
    accuracy, num_epochs = run_experiment(train, 10, 20, 100, 25)
    print("accuracy: %.2f" % accuracy)

else:
    print("\n\n NEW EXPERIMENT:\n")

    #single run
    sum_average=0
    for i in range(1): #number of iterations
        with open(filename, "a+") as file:
            file.write("\nInteration: "+ str(i)+"\n")
            print("\nIteration: "+str(i))
        best_accuracy, num_epochs=run_experiment(train, 10, 50, 800, 25)
        sum_average+=best_accuracy
        average_accuracy=sum_average/(i+1)




        with open(filename, "a+") as file:
            file.write("\nAv accuracy: %.2f, best accuracy: %.2f\n" % (average_accuracy, best_accuracy))
        print("\nAv accuracy: %.2f, best accuracy: %.2f\n" % (average_accuracy, best_accuracy))

        

#multiple runs

# for i1 in range(1,20):        
#     for i2 in range(1,20):
#         with open(filename, "a+") as file:
#             file.write("\n\nnumber of hidden nodes 1: "+str(i1)+", hidden nodes 2: " +str(i2)+"\n")
#             print("\n\nnumber of hidden nodes 1: "+str(i1)+", hidden nodes 2: " +str(i2)+"\n")
    
#         best_accuracy, num_epochs=run_experiment(i1, i2)
#         with open(filename, "a+") as file:
#             file.write("Best accuracy: %.2f in %d epochs" % (best_accuracy, num_epochs-early_stopping))
#             print("Best accuracy: %.2f in %d epochs" % (best_accuracy, num_epochs-early_stopping))