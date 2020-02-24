# it's the same test as for mnist.#L.py but with conv layers (con lenet)
# it's also a gpu version which add extra gpu support to the previous version of mnist.3L.conv.py (which wa deleted and this version was named after this)

<<<<<<< HEAD

######################
# it creates examples of the adversarial - it shows them
# it creates the while dataset fo adversarial examples

=======
# transforms the input data
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78

import torch
from torch import nn, optim
import torch.nn.functional as f
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable



import torch.utils.data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import numpy as np
import csv
import pdb
<<<<<<< HEAD
#import cv2
=======
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
print(device)

###################################################
# DATA

dataset = "FashionMNIST"
<<<<<<< HEAD
adversarial_dataset=True
adversarial_attack_when_evaluating_image = True
evaluate_image_bool = False
trainval_perc = 1
BATCH_SIZE = 100 #changed to see one images at a time
=======
trainval_perc = 1
BATCH_SIZE = 1 #changed to see one images at a time
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78
# Download or load downloaded MNIST dataset
# shuffle data at every epoch

trainval_dataset = datasets.__dict__[dataset]('data', train=True, download=True,
                                              # transform=transforms.Compose([transforms.ToTensor(),
                                              # transforms.Normalize((0.1307,), (0.3081,))]),
                                              transform=transforms.ToTensor())

train_size = int(trainval_perc * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Same for test data
test_loader = torch.utils.data.DataLoader(
    # datasets.MNIST('data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
    datasets.__dict__[dataset]('data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=False)

<<<<<<< HEAD
if adversarial_dataset:
    tensor_x = torch.load('data/FashionMNIST_adversarial/tensor_x.pt')
    tensor_y = torch.load('data/FashionMNIST_adversarial/tensor_y.pt')
    tensor_x=tensor_x.unsqueeze(1)

    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)  # create your datset
    test_loader = torch.utils.data.DataLoader(my_dataset)  # create your dataloader

    # for i, (m, n) in enumerate(my_dataloader):
    #     print(i)
    #     print(m)

=======
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78

##########################################################
# NETWORK


class Lenet(nn.Module):
    def __init__(self, nodesNum1, nodesNum2, nodesFc1, nodesFc2):
        super(Lenet, self).__init__()

        self.nodesNum2 = nodesNum2

        self.c1 = nn.Conv2d(1, nodesNum1, 5)
        self.s2 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(nodesNum1)
        self.c3 = nn.Conv2d(nodesNum1, nodesNum2, 5)
        self.s4 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(nodesNum2)
        self.c5 = nn.Linear(nodesNum2 * 4 * 4, nodesFc1)
        self.f6 = nn.Linear(nodesFc1, nodesFc2)
        self.output = nn.Linear(nodesFc2, 10)

        self.drop_layer = nn.Dropout(p=0.5)

    def forward(self, x):
        # x=x.view(-1,784)
        # output=f.relu(self.fc1(x))
        # output=self.bn1(output)
        # output=f.relu(self.fc2(output))
        # output=self.bn2(output)
        # output=self.fc3(output)
        # return output

        # x=x.view(-1,784)
        output = self.c1(x)
        output = f.relu(self.s2(output))
        output = self.bn1(output)
        output = self.drop_layer(output)
        output = self.c3(output)

        output = f.relu(self.s4(output))
        output = self.bn2(output)
        output = self.drop_layer(output)
        output = output.view(-1, self.nodesNum2 * 4 * 4)

        output = self.c5(output)
        output = self.f6(output)
        return output


###################################################
# RUN

def run_experiment(early_stopping, nodesNum1, nodesNum2, nodesFc1, nodesFc2):
    resume = True
    training = False

    net = Lenet(nodesNum1, nodesNum2, nodesFc1, nodesFc2).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer=optim.Adam(net.parameters(), lr=0.001)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    ######################################
    # LOADING MODEL/RESUME

    def load_model():
        # path="models/fashionmnist_conv:20_conv:50_fc:800_fc:500_rel_bn_trainval1.0_epo:11_acc:90.01"
        # path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
        if dataset == "MNIST":
            path = "models/mnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo540_acc99.27"
        elif dataset == "FashionMNIST":
            path = "models/fashionmnist_conv10_conv20_fc100_fc25_rel_bn_drop_trainval_modelopt1.0_epo62_acc90.04"
        # path="models/conv:10_conv:50_fc:800_fc:500_rel_bn_epo:103_acc:99.37""
        # path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:11_switch_acc:99.15"
        # path="/home/kamil/Dropbox/Current_research/python_tests/Dir_switch/models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:2_acc:98.75"

        net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['model_state_dict'],
                            strict=False)
        # path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_trainval_modelopt1.0_epo:309_acc:99.19"
        # path="models/mnist_conv:10_conv:20_fc:100_fc:25_rel_bn_drop_trainval_modelopt1.0_epo:540_acc:99.27"

<<<<<<< HEAD

    # EVALUATE

    def visualize(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob):
        x = x.squeeze(0)  # remove batch dimension # B X C H X W ==> C X H X W
        #x = x.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1)).numpy()  # reverse of normalization op- "unnormalize", multiply by std and add mean
        x=x.detach().numpy()
        #x = np.transpose(x, (1, 2, 0))  # C X H X W  ==>   H X W X C
        x = np.clip(x, 0, 1)

        x_adv = x_adv.squeeze(0)
        x_adv=x_adv.detach().numpy()
        #x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1)).numpy()  # reverse of normalization op
        #x_adv = np.transpose(x_adv, (1, 2, 0))  # C X H X W  ==>   H X W X C
        x_adv = np.clip(x_adv, 0, 1)

        x_grad = x_grad.squeeze(0).numpy()
        x_grad = np.transpose(x_grad, (1, 2, 0))
        x_grad = x_grad.squeeze(2)
        x_grad = np.clip(x_grad, 0, 1)

        figure, ax = plt.subplots(1, 3, figsize=(18, 8))
        ax[0].imshow(x)
        ax[0].set_title('Clean Example', fontsize=20)

        ax[1].imshow(x_grad)
        ax[1].set_title('Perturbation', fontsize=20)
        ax[1].set_yticklabels([])
        ax[1].set_xticklabels([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        ax[2].imshow(x_adv)
        ax[2].set_title('Adversarial Example', fontsize=20)

        ax[0].axis('off')
        ax[2].axis('off')

        ax[0].text(1.1, 0.5, "+{}*".format(round(epsilon, 3)), size=15, ha="center",
                   transform=ax[0].transAxes)

        ax[0].text(0.5, -0.13, "Prediction: {}\n Probability: {}".format(clean_pred, clean_prob), size=15, ha="center",
                   transform=ax[0].transAxes)

        ax[1].text(1.1, 0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

        ax[2].text(0.5, -0.13, "Prediction: {}\n Probability: {}".format(adv_pred, adv_prob), size=15, ha="center",
                   transform=ax[2].transAxes)

        plt.show()

    def adversarial(image, class_least_likely):

        method =1
        y_target = Variable(torch.LongTensor([class_least_likely[0]]), requires_grad=False).to(device)  # 9= ostrich
        alpha = 0.025
        num_steps = 10
=======
    ##########################################
    # EVALUATE

    # def evaluate():
    #     net.eval()
    #     correct = 0
    #     total = 0
    #     for j, data in enumerate(test_loader):
    #         images, labels = data
    #         images =images.to(device)
    #         predicted_prob = net.forward(images) #images.view(-1,28*28)
    #
    #         predicted=np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
    #         total += labels.size(0)
    #         correct += (predicted == labels.numpy()).sum().item()
    #
    #     accuracy=100 * float(correct) / total
    #     print("accuracy: %.2f %%" % (accuracy))
    #     return accuracy
    # looks same as below
    ########################################################

    adversarial_attack=True
    # EVALUATE

    def adversarial(image):

        method =2
        y_target = Variable(torch.LongTensor([9]), requires_grad=False).to(device)  # 9= ostrich
        alpha = 0.025
        num_steps = 5
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78

        # img_variable.data = image_tensor  # in previous method we assigned it to the adversarial img
        image_tensor = Variable(image, requires_grad=True)  # convert tensor into a variable

        img_variable_temp = image_tensor  # in previous method we assigned it to the adversarial img

        if method==1:
<<<<<<< HEAD
            eps = 0.25
=======
            epsilon = 0.25
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78

            img_variable_temp = image_tensor  # in previous method we assigned it to the adversarial img

            for i in range(num_steps):
                zero_gradients(img_variable_temp)
                output = net.forward(img_variable_temp)
                loss = torch.nn.CrossEntropyLoss()
                loss_cal = loss(output, y_target)
                loss_cal.backward()

                x_grad = alpha * torch.sign(img_variable_temp.grad.data)

                adv_temp = img_variable_temp.data - x_grad

                total_grad = adv_temp - image_tensor
<<<<<<< HEAD
                total_grad = torch.clamp(total_grad, -eps, eps)
=======
                total_grad = torch.clamp(total_grad, -epsilon, epsilon)
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78
                x_adv = image_tensor + total_grad
                img_variable_temp.data = x_adv

            output_adv = net.forward(img_variable_temp)
<<<<<<< HEAD
            x_adversarial=img_variable_temp
=======
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78
            #x_adv_pred = labels[torch.max(output_adv.data, 1)[1][0]]
            output_adv_softmax = f.softmax(output_adv, dim=1)
            #softm=torch.nn.functional.softmax(predicted_prob)

<<<<<<< HEAD
            advers_bestclass_pred_prob = round(float(torch.max(output_adv_softmax.data, 1)[0][0]) * 100, 4)

            advers_predicted_class = np.argmax(output_adv_softmax.cpu().detach().numpy(), axis=1)

            print("\nAdversarial\n", advers_predicted_class, advers_bestclass_pred_prob, output_adv)
=======
            x_adv_pred_softmax_max = round(float(torch.max(output_adv_softmax.data, 1)[0][0]) * 100, 4)

            predicted_class = np.argmax(output_adv_softmax.cpu().detach().numpy(), axis=1)

            print("\nAdversarial\n", predicted_class, x_adv_pred_softmax_max, output_adv)
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78

        if method==2:

            for i in range(num_steps):
                zero_gradients(img_variable_temp)
                output = net.forward(img_variable_temp)
                loss = torch.nn.CrossEntropyLoss()
                loss_cal = loss(output, y_target)

                loss_cal.backward()

                x_grad = alpha * torch.sign(img_variable_temp.grad.data)

                eps = 0.02
                x_grad = torch.sign(img_variable_temp.grad.data)  # calculate the sign of gradient of the loss func (with respect to input X) (adv)
                x_adversarial = img_variable_temp.data + eps * x_grad  # find adv example using formula shown above
                output_adv = net.forward(Variable(x_adversarial))  # perform a forward pass on adv example
<<<<<<< HEAD
                advers_predicted_class = int(torch.max(output_adv.data, 1)[1][0])  # classify the adv example
                op_adv_probs = f.softmax(output_adv, dim=1)  # get probability distribution over classes
                advers_bestclass_pred_prob = round(float((torch.max(op_adv_probs.data, 1)[0][0]) * 100),4)  # find probability (confidence) of a predicted class

            print("Adversarial: ", advers_predicted_class, advers_bestclass_pred_prob, op_adv_probs)

        return image_tensor, x_adversarial, x_grad, eps, advers_predicted_class, advers_bestclass_pred_prob
=======
                x_adv_pred = int(torch.max(output_adv.data, 1)[1][0])  # classify the adv example
                op_adv_probs = f.softmax(output_adv, dim=1)  # get probability distribution over classes
                adv_pred_prob = round(float((torch.max(op_adv_probs.data, 1)[0][0]) * 100),4)  # find probability (confidence) of a predicted class

            print("Adversarial: ", x_adv_pred, op_adv_probs, adv_pred_prob)
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78

            # adv_temp = img_variable.data - x_grad
            # total_grad = adv_temp - image_tensor
            # total_grad = torch.clamp(total_grad, -epsilon, epsilon)
            # x_adv = image_tensor + total_grad
            # img_variable.data = x_adv

        # output_adv = inceptionv3.forward(img_variable)
        # x_adv_pred = labels[torch.max(output_adv.data, 1)[1][0]]
        # output_adv_probs = F.softmax(output_adv, dim=1)
        # x_adv_pred_prob = round((torch.max(output_adv_probs.data, 1)[0][0]) * 100, 4)

<<<<<<< HEAD
    def evaluate_image(images, classes, ind):
=======
    def evaluate_image(images, classes):
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78
        # print('Prediction when network is forced to predict')
        net.eval()
        correct = 0
        total = 0
        images = images.to(device)
<<<<<<< HEAD
        predicted_output = net.forward(images)  # images.view(-1,28*28)

        x_pred_prob=torch.nn.functional.softmax(predicted_output)
        bestclass_pred_prob= round(float((torch.max(x_pred_prob, 1)[0][0]) * 100),4)

        x_pred = np.argmax(predicted_output.cpu().detach().numpy(), axis=1)
        x_least_likely=np.argmin(predicted_output.cpu().detach().numpy(), axis=1)
        print("\nGT\n", classes)
        print("\nNormal\n", x_pred, torch.max(x_pred_prob, 1), x_pred_prob)

        if adversarial_attack_when_evaluating_image:
            image_tensor, x_adversarial, x_grad, eps, x_adv_pred, adv_bestclass_pred_prob=adversarial(images, x_least_likely)

        #visualize(image_tensor[0][0], x_adversarial[0][0], x_grad, eps, x_pred, x_adv_pred, bestclass_pred_prob, adv_bestclass_pred_prob)

        path_save="data/FashionMNIST_adversarial/fashionmnist_"+str(ind)
        np.save(path_save, x_adversarial[0][0].detach().numpy(), 'a')
        #plt.imsave(path_save, x_adversarial[0][0].detach().numpy())

=======
        predicted_prob = net.forward(images)  # images.view(-1,28*28)

        softm=torch.nn.functional.softmax(predicted_prob)

        predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
        print("\nGT\n", classes)
        print("\nNormal\n", predicted, torch.max(softm, 1), predicted_prob)

        if adversarial_attack:
            adversarial(images)
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78
        #total += labels.size(0)
        #correct += (predicted == labels.numpy()).sum().item()
        # print(str(correct) +" "+ str(total))
        # pdb.set_trace()
        #accuracy = 100 * float(correct) / total
        #print("test accuracy: %.2f %%" % (accuracy))
        return -1

    def evaluate():
        # print('Prediction when network is forced to predict')
        net.eval()
        correct = 0
        total = 0
        for j, data in enumerate(test_loader):
            images, labels = data
            images = images.to(device)
            predicted_prob = net.forward(images)  # images.view(-1,28*28)
            predicted = np.argmax(predicted_prob.cpu().detach().numpy(), axis=1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels.numpy()).sum().item()
        # print(str(correct) +" "+ str(total))
        # pdb.set_trace()
        accuracy = 100 * float(correct) / total
        print("test accuracy: %.2f %%" % (accuracy))
        return accuracy

<<<<<<< HEAD
    #######################################################################

=======
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78
    if resume:
        load_model()


<<<<<<< HEAD
        if evaluate_image_bool:
            #evaluate images
            labels=[]
            for j, (images, classes) in enumerate(test_loader):
                #if j<10:
                if 1:
                    evaluate_image(images, classes, j)
                    labels.append(np.array(classes.numpy()))
            np.save("data/FashionMNIST_adversarial/labels.npy", labels, 'a')

        else:
            evaluate()

#        np.save("data/FashionMNIST_adversarial/labels.npy", 'a')


=======

        #evaluate images
        for j, (images, classes) in enumerate(test_loader):
            if j<10:
                evaluate_image(images, classes)

    def visualize(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob):
        x = x.squeeze(0)  # remove batch dimension # B X C H X W ==> C X H X W
        x = x.mul(torch.FloatTensor(std).view(3, 1, 1)).add(
            torch.FloatTensor(mean).view(3, 1,
                                         1)).numpy()  # reverse of normalization op- "unnormalize", multiply by std and add mean
        x = np.transpose(x, (1, 2, 0))  # C X H X W  ==>   H X W X C
        x = np.clip(x, 0, 1)

        x_adv = x_adv.squeeze(0)
        x_adv = x_adv.mul(torch.FloatTensor(std).view(3, 1, 1)).add(
            torch.FloatTensor(mean).view(3, 1, 1)).numpy()  # reverse of normalization op
        x_adv = np.transpose(x_adv, (1, 2, 0))  # C X H X W  ==>   H X W X C
        x_adv = np.clip(x_adv, 0, 1)

        x_grad = x_grad.squeeze(0).numpy()
        x_grad = np.transpose(x_grad, (1, 2, 0))
        x_grad = np.clip(x_grad, 0, 1)

        figure, ax = plt.subplots(1, 3, figsize=(18, 8))
        ax[0].imshow(x)
        ax[0].set_title('Clean Example', fontsize=20)

        ax[1].imshow(x_grad)
        ax[1].set_title('Perturbation', fontsize=20)
        ax[1].set_yticklabels([])
        ax[1].set_xticklabels([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        ax[2].imshow(x_adv)
        ax[2].set_title('Adversarial Example', fontsize=20)

        ax[0].axis('off')
        ax[2].axis('off')

        ax[0].text(1.1, 0.5, "+{}*".format(round(epsilon, 3)), size=15, ha="center",
                   transform=ax[0].transAxes)

        ax[0].text(0.5, -0.13, "Prediction: {}\n Probability: {}".format(clean_pred, clean_prob), size=15, ha="center",
                   transform=ax[0].transAxes)

        ax[1].text(1.1, 0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

        ax[2].text(0.5, -0.13, "Prediction: {}\n Probability: {}".format(adv_pred, adv_prob), size=15, ha="center",
                   transform=ax[2].transAxes)

        plt.show()

    #visualize(image_tensor, x_adversarial, x_grad, eps, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob)
>>>>>>> 40694c14a26d808b1e780e581505094fa4c9ca78



    def train():

        stop = 0;
        epoch = 0;
        best_accuracy = 0;
        entry = np.zeros(3);
        best_model = -1
        while (stop < early_stopping):
            epoch = epoch + 1
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # if i % 100==0:
                #    print (i)
                #   print (loss.item())
            # print (i)
            print(loss.item())
            accuracy = evaluate()
            print("Epoch " + str(epoch) + " ended.")

            if (accuracy <= best_accuracy):
                stop = stop + 1
                entry[2] = 0
            else:
                best_accuracy = accuracy
                print("Best updated")
                stop = 0
                entry[2] = 1
                best_model = net.state_dict()
                best_optim = optimizer.state_dict()
                torch.save({'model_state_dict': best_model, 'optimizer_state_dict': best_optim},
                           "models/%s_conv:%d_conv:%d_fc:%d_fc:%d_rel_bn_drop_trainval_modelopt%.1f_epo:%d_acc:%.2f" % (
                           dataset, conv1, conv2, fc1, fc2, trainval_perc, epoch, best_accuracy))

            print("\n")
            # write
            entry[0] = accuracy;
            entry[1] = loss
            with open(filename, "a+") as file:
                file.write(",".join(map(str, entry)) + "\n")
        return best_accuracy, epoch, best_model




    if training:
        best_accuracy, epoch, best_model = train()
        return best_accuracy, epoch, best_model
    else:
        return -1, -1, -1


print("\n\n NEW EXPERIMENT:\n")

########################################################
# PARAMS
early_stopping = 350
sum_average = 0;
conv1 = 10;
conv2 = 20;
fc1 = 100;
fc2 = 25
filename = "%s_test_conv_relu_bn_drop_trainval%.1f_conv:%d_conv:%d_fc:%d_fc:%d.txt" % (
dataset, trainval_perc, conv1, conv2, fc1, fc2)
filename = "%s_test_conv_relu_bn_drop_trainval%.1f_conv:%d_conv:%d_fc:%d_fc:%d.txt" % (
dataset, trainval_perc, conv1, conv2, fc1, fc2)

######################################################
# single run  avergaed pver n iterations

for i in range(1):
    with open(filename, "a+") as file:
        file.write("\nInteration: " + str(i) + "\n")
        print("\nIteration: " + str(i))
    best_accuracy, num_epochs, best_model = run_experiment(early_stopping, conv1, conv2, fc1, fc2)
    sum_average += best_accuracy
    average_accuracy = sum_average / (i + 1)

    with open(filename, "a+") as file:
        file.write("\nAv accuracy: %.2f, best accuracy: %.2f\n" % (average_accuracy, best_accuracy))
    print("\nAv accuracy: %.2f, best accuracy: %.2f\n" % (average_accuracy, best_accuracy))
    # torch.save(best_model, filename_model)

# multiple runs

# for i1 in range(1,20):
#     for i2 in range(1,20):
#         with open(filename, "a+") as file:
#             file.write("\n\nnumber of hidden nodes 1: "+str(i1)+", hidden nodes 2: " +str(i2)+"\n")
#             print("\n\nnumber of hidden nodes 1: "+str(i1)+", hidden nodes 2: " +str(i2)+"\n")

#         best_accuracy, num_epochs=run_experiment(i1, i2)
#         with open(filename, "a+") as file:
#             file.write("Best accuracy: %.2f in %d epochs" % (best_accuracy, num_epochs-early_stopping))
#             print("Best accuracy: %.2f in %d epochs" % (best_accuracy, num_epochs-early_stopping))