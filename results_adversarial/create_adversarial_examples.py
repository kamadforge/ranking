# code from
# https://savan77.github.io/blog/imagenet_adv_examples.html

# import required libs
import torch
import torch.nn
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import requests, io
import matplotlib.pyplot as plt
from torch.autograd import Variable

visualize_images = False

inceptionv3 = models.inception_v3(pretrained=True)  # download and load pretrained inceptionv3 model
inceptionv3.eval();

url = "https://savan77.github.io/blog/images/ex4.jpg"  # tiger cat #i have uploaded 4 images to try- ex/ex2/ex3.jpg
response = requests.get(url)
img = Image.open(io.BytesIO(response.content))
# img.show()

# mean and std will remain same irresptive of the model you use
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Next, we will preprocess our input image using the function created above.


image_tensor = preprocess(img)  # preprocess an i
image_tensor = image_tensor.unsqueeze(0)  # add batch dimension.  C X H X W ==> B X C X H X W

# Next, we will classify this image using pre-trained inceptionv3 model that we just loaded.

img_variable = Variable(image_tensor, requires_grad=True)  # convert tensor into a variable

output = inceptionv3.forward(img_variable)
label_idx = torch.max(output.data, 1)[1][0]  # get an index(class number) of a largest element
print(label_idx)

# Next, we will create a dictionary to map index to imagenet class. ImageNet has 1000 classes.

labels_link = "https://savan77.github.io/blog/files/labels.json"
labels_json = requests.get(labels_link).json()
labels = {int(idx): label for idx, label in labels_json.items()}
x_pred = labels[int(label_idx)]
print(x_pred)

# get probability dist over classes
output_probs = F.softmax(output, dim=1)
x_pred_prob = (torch.max(output_probs.data, 1)[0][0]) * 100
print(x_pred_prob)

#####################################################################################################################
# Fast Gradient Sign Method

# Let's say we have an input X, which is correctly classified by our model (M). We want to find an adversarial example X̂, which is perceptually indistinguishable from original input X, such that it will be misclassified by that same model (M). We can do that by adding an adversarial perturbation ( θ ) to the original input. Note that we want adversarial example to be indistinguishable from the original one. That can be achieved by constraining the magnitude of adversarial perturbation: ||X−X̂ ||∞⩽ϵ. That is, the L∞ norm should be less than epsilon. Here, L∞ denotes the maximum changes for all pixels in adversarial example. Fast Gradient Sign Method (FGSM) is a fast and computationally efficient method to generate adversarial examples. However, it usually has a lower success rate. The formula to find adversarial example is as follows:
# Xadv=X+ϵsign(∇XJ(X,Ytrue)
# Here,
# X = original (clean) input
# Xadv = adversarial input (intentionally designed to be misclassified by our model)
# ϵ = magnitude of adversarial perturbation
# ∇XJ(X,Ytrue) = gradient of loss function w.r.t to input (X)

y_true = 282  # tiger cat  ##change this if you change input image
target = Variable(torch.LongTensor([y_true]), requires_grad=False)
print(target)

# perform a backward pass in order to get gradients
loss = torch.nn.CrossEntropyLoss()
loss_cal = loss(output, target)
loss_cal.backward(
    retain_graph=True)  # this will calculate gradient of each variable (with requires_grad=True) and can be accessed by "var.grad.data"

# Following code cell computes the adversarial example using formula shown above.

eps = 0.02
x_grad = torch.sign(
    img_variable.grad.data)  # calculate the sign of gradient of the loss func (with respect to input X) (adv)
x_adversarial = img_variable.data + eps * x_grad  # find adv example using formula shown above
output_adv = inceptionv3.forward(Variable(x_adversarial))  # perform a forward pass on adv example
x_adv_pred = labels[int(torch.max(output_adv.data, 1)[1][0])]  # classify the adv example
op_adv_probs = F.softmax(output_adv, dim=1)  # get probability distribution over classes
adv_pred_prob = round(float((torch.max(op_adv_probs.data, 1)[0][0]) * 100),
                      4)  # find probability (confidence) of a predicted class

print(x_adv_pred)
print(adv_pred_prob)


# Finally, we generated an adversarial example which was misclassified by our model. Since Egyptian cat look somewhat similar to tiger cat, it's not that great. We still need to do better. Next, we define a function which visualizes original input, adversarial input and adversarial perturbation. This will give us a better understanding of how adversarial examples look like, are they indistinguishable from odiginal input or not.


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


if visualize_images:
    visualize(image_tensor, x_adversarial, x_grad, eps, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob)

# As you can see, the generated adversarial image is visually indistinguishable from the original image but inceptionv3 classifies it as Egyptian cat. Now, let us generate several adversarial images with different values of epsilon. Notice that as we increase the value of epsilon the adversarial image becomes distinguishable from the original one.

epsilon = [0.00088, 0.004, 0.01, 0.12, 0.55]

x_grad = torch.sign(img_variable.grad.data)
for i in epsilon:
    x_adversarial = img_variable.data + i * x_grad
    output_adv = inceptionv3.forward(Variable(x_adversarial))
    x_adv_pred = labels[int(torch.max(output_adv.data, 1)[1][0])]
    op_adv_probs = F.softmax(output_adv, dim=1)
    adv_pred_prob = round(float(torch.max(op_adv_probs.data, 1)[0][0]) * 100, 4)
    if visualize_images:
        visualize(image_tensor, x_adversarial, x_grad, i, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob)

# For very small value of epsilon, class doesn't change. But it decreases the probability. An alternative way is to use raw gradient (not sign) without any constraint (epsilon). It is called as Fast Gradient Value Method.


# One-step target class method

# As you might have noticed, FGSM finds perturbation which increases the loss for the true class and subsequently that leads to misclassification. Also, FGSM is a non-targeted method. We can easily convert it into targeted method by maximizing probability P(Ytarget|X)
# for some target class Ytarget. For a neural network with cross-entropy loss the formula will be:
# Xadv=X−ϵsign(∇XJ(X,Ytarget)
# The only change we need to make is that instead of adding perturbation to the original input, now we need to remove it from the original input. But, how to choose the target class? There are two ways. One is to use a random class as a target class. Another and recommended way is to use class to which our model assigned lowest probability. It is also known as least likely class. Here, we will use random class. ( I'm lazy :) )

# targeted class can be a random class or the least likely class predicted by the network
y_target = 288  # leopard
y_target = Variable(torch.LongTensor([y_target]), requires_grad=False)
print(y_target)

zero_gradients(img_variable)  # flush gradients
loss_cal2 = loss(output, y_target)
loss_cal2.backward()

epsilons = [0.002, 0.01, 0.15, 0.5]

x_grad = torch.sign(img_variable.grad.data)
for i in epsilons:
    x_adversarial = img_variable.data - i * x_grad
    output_adv = inceptionv3.forward(Variable(x_adversarial))
    x_adv_pred = labels[int(torch.max(output_adv.data, 1)[1][0])]
    op_adv_probs = F.softmax(output_adv, dim=1)
    adv_pred_prob = round(float(torch.max(op_adv_probs.data, 1)[0][0]) * 100, 4)
    if visualize_images:
        visualize(image_tensor, x_adversarial, x_grad, i, x_pred, x_adv_pred, x_pred_prob, adv_pred_prob)

# Seems like it doesn't work well here. You may try with other images and results might change. Let's move on to other method and see if it can generate better adversarial examples.
# Basic Iterative Method

# Basic Iterative Method is an extension of FGSM method where it applies FGSM multiple times with small step size. We initialize adversarial image as an original image and then take one step along the direction of gradient during each iteration.
# Xadv0=X
# XadvN+1=ClipX,ϵ(XadvN+αsign(∇XJ(XadvN,Ytrue))

# Here, ClipX,ϵ
# denotes clipping of input in the range of [X−ϵ,X+ϵ]

y_true = Variable(torch.LongTensor([282]), requires_grad=False)  # tiger cat
epsilon = 0.25
num_steps = 5
alpha = 0.025
# above three are hyperparameters


for i in range(num_steps):
    zero_gradients(img_variable)  # flush gradients
    output = inceptionv3.forward(img_variable)  # perform forward pass
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(output, y_true)
    loss_cal.backward()
    x_grad = alpha * torch.sign(img_variable.grad.data)  # as per the formula
    adv_temp = img_variable.data + x_grad  # add perturbation to img_variable which also contains perturbation from previous iterations
    total_grad = adv_temp - image_tensor  # total perturbation
    total_grad = torch.clamp(total_grad, -epsilon, epsilon)
    x_adv = image_tensor + total_grad  # add total perturbation to the original image
    img_variable.data = x_adv

# final adversarial example can be accessed at- img_variable.data

output_adv = inceptionv3.forward(img_variable)
x_adv_pred = labels[int(torch.max(output_adv.data, 1)[1][0])]  # classify adversarial example
output_adv_probs = F.softmax(output_adv, dim=1)
x_adv_pred_prob = round(float(torch.max(output_adv_probs.data, 1)[0][0]) * 100, 4)
# if visualize_images:
visualize(image_tensor, img_variable.data, total_grad, epsilon, x_pred, x_adv_pred, x_pred_prob,
          x_adv_pred_prob)  # class and prob of original ex will remain same

# Looks great! Inceptionv3 classified it as Egyptian cat with more confidence than it has for the "tiger cat".
# Iterative Target Class Method

# We can do the same for one-step target class method. As we show before, in order to make FGSM targeted we need to remove perturbation from original image. Here, we will do this for multiple iteration. Again note that it is recommended to use least likely class, but I am using random one. You can find least likely class by running y_LL = torch.min(output.data, 1)[1][0]

y_target = Variable(torch.LongTensor([9]), requires_grad=False)  # 9= ostrich
epsilon = 0.25
num_steps = 5
alpha = 0.025

img_variable.data = image_tensor  # in previous method we assigned it to the adversarial img

for i in range(num_steps):
    zero_gradients(img_variable)
    output = inceptionv3.forward(img_variable)
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(output, y_target)
    loss_cal.backward()
    x_grad = alpha * torch.sign(img_variable.grad.data)
    adv_temp = img_variable.data - x_grad
    total_grad = adv_temp - image_tensor
    total_grad = torch.clamp(total_grad, -epsilon, epsilon)
    x_adv = image_tensor + total_grad
    img_variable.data = x_adv

output_adv = inceptionv3.forward(img_variable)
x_adv_pred = labels[torch.max(output_adv.data, 1)[1][0]]
output_adv_probs = F.softmax(output_adv, dim=1)
x_adv_pred_prob = round((torch.max(output_adv_probs.data, 1)[0][0]) * 100, 4)
if visualize_images:
    visualize(image_tensor, img_variable.data, total_grad, epsilon, x_pred, x_adv_pred, x_pred_prob, x_adv_pred_prob)

# As you can see, the generated adversarial image is indistinguishale from the original image but our model classified it as an ostrich. I tried with several different target classes and it worked every time.

# The methods introduced here are very simple and efficient methods to generate adversarial examples. As you might have noticed, iterative methods produce better adversarial examples than one-step methods. There are other methods as well (DeepFool, Carlini & Wagner's Attack, Jacobian-based Saliency Map Attack, etc). There are also some methods (Defensive Distillation, Adversarial Training) to defend models against such attacks, which I will introduce in another notebook.